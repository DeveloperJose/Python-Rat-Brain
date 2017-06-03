# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import pickle
import random
import config

from multiprocessing.pool import ThreadPool

import logbook
logger = logbook.Logger(__name__)

# ****************************** SIFT Parameters
SIFT = cv2.xfeatures2d.SIFT_create(nfeatures=3, contrastThreshold=config.SIFT_CONTRAST_THRESHOLD, edgeThreshold=config.SIFT_EDGE_THRESHOLD, sigma=config.SIFT_SIGMA)

# ****************************** Matcher Parameters
FLANN = cv2.FlannBasedMatcher(config.FLANN_INDEX_PARAMS, config.FLANN_SEARCH_PARAMS)
BF = cv2.BFMatcher(normType=cv2.NORM_HAMMING)

class ImageInfo(object):
    def __init__(self, im, title, filename):
        self.im = im
        self.title = title
        self.filename = filename

class Match(object):
    def __init__(self, nissl_level, matches, ransac_results, im_results, original_moments, transformed_moments, hu_dist, isConvex):
        self.nissl_level = nissl_level
        self.matches = matches
        self.ransac_results = ransac_results
        self.im_results = im_results
        self.original_moments = original_moments
        self.transformed_moments = transformed_moments
        self.hu_dist = hu_dist
        self.isConvex = isConvex

        # Process variables
        self.H = ransac_results["homography"]
        self.inlier_count = ransac_results["inlier_count"]

        self.matches_count = len(matches)
        self.inlier_ratio = self.inlier_count / self.matches_count * 100

        self.homography_det = abs(np.linalg.det(self.H))
        self.svd = np.linalg.svd(self.H, compute_uv=False)
        self.cond_num = int(self.svd[0] / self.svd[-1])

    def comparison_key(self):
        return self.inlier_count
        #return self.inlier_count
        #return -((1/self.inlier_ratio) * (self.svd_ratio) * self.homography_det * self.dist)

    # "{0:.8f}".format
    def to_string_array(self):
        return np.array([
            "Plate #" + str(self.nissl_level),
            str(self.matches_count),
            str(self.inlier_count),
            "{0:.1f}".format(self.inlier_ratio),
            str(self.cond_num),
            str(self.homography_det),
            str(self.original_moments['m00']),
            str(self.transformed_moments['m00']),
            str(self.hu_dist),
            str(self.isConvex),
            str(self.ransac_results["total_error"]),
            str(self.ransac_results["min_error"]),
            str(self.ransac_results["max_error"])
            ])

def warp(im, points, disp_min, disp_max, disp_len = None, disp_angle = None):
    h, w = im.shape[:2]

    # Include the corners
    src_pts = np.array([[0, 0], [w, 0], [0, h], [w, h]])
    dst_pts = np.array([[0, 0], [w, 0], [0, h], [w, h]])

    for i in range(points):
        rand_x = random.uniform(0, w)
        rand_y = random.uniform(0, h)
        p = np.array([rand_x, rand_y])
        if disp_len is None:
            rand_len = random.uniform(disp_min, disp_max)
        else:
            rand_len = disp_len

        if disp_angle is None:
            rand_angle = np.deg2rad(random.uniform(0, 360))
        else:
            rand_angle = disp_angle

        p2_x = rand_x + np.cos(rand_angle) * rand_len
        p2_y = rand_y + np.sin(rand_angle) * rand_len
        p2 = np.array([p2_x, p2_y])

        if src_pts is None:
            src_pts = np.array([p])
        else:
            temp = np.vstack((src_pts, p))
            src_pts = temp

        if dst_pts is None:
            dst_pts = np.array([p2])
        else:
            temp = np.vstack((dst_pts, p2))
            dst_pts = temp

    from skimage.transform import warp, PiecewiseAffineTransform
    tform = PiecewiseAffineTransform()
    tform.estimate(src_pts, dst_pts)

    return warp(im, tform)

def im_read(filename, flags=cv2.IMREAD_COLOR):
    im = cv2.imread(filename, flags)

    if flags == cv2.IMREAD_COLOR:
        # Convert from BGR to RGB for some reason
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    return im

def im_write(filename, im):
    if len(im.shape) == 3:
        # Not grayscale
        # Convert back to the original BGR
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    cv2.imwrite(filename, im)

def nissl_load(nissl_level, color_flags=cv2.IMREAD_COLOR):
    filename = config.NISSL_PREFIX + str(nissl_level).zfill(config.NISSL_DIGITS) + config.NISSL_EXT
    path = os.path.join(config.NISSL_DIR, filename)

    if not os.path.exists(path):
        logger.error("Tried to load plate {0} but it wasn't found in path {1}", nissl_level, path)
        return None

    return im_read(path, color_flags)

def nissl_load_sift(nissl_level):
    filename = config.NISSL_PREFIX + str(nissl_level).zfill(config.NISSL_DIGITS) + ".sift"
    path = os.path.join(config.NISSL_DIR, filename)

    if not os.path.exists(path):
        logger.info("Creating SIFT for plate {0}", nissl_level)
        nissl = nissl_load(nissl_level, cv2.IMREAD_GRAYSCALE)
        if nissl is None:
            return None

        import scipy.misc as misc
        old_shape = nissl.shape
        reduction_percent = int(config.RESIZE_WIDTH/old_shape[0] * 100)
        nissl = misc.imresize(nissl, reduction_percent)
        logger.debug("Resized region from {0} to {1}", old_shape, nissl.shape)

        kp, des = extract_sift(nissl)
        temp = pickle_sift(kp, des)
        pickle.dump(temp, open(path, "wb"))
        return kp, des

    else:
        raw_sift = pickle.load(open(path, "rb"))
        kp, des = unpickle_sift(raw_sift)
        return kp, des

def match(im1, kp1, des1, im2, kp2, des2, plate=None):
    if config.MATCH_WITH_FLANN:
        matches = FLANN.knnMatch(des1, des2, k=2)
    else:
        matches = BF.knnMatch(des1, des2, k=2)

    # Apply Ratio Test
    good_matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * config.DISTANCE_RATIO]

    if len(good_matches) < config.MIN_MATCH_COUNT:
        logger.debug("Matches lower than threshold. {0} < {1}", len(good_matches), config.MIN_MATCH_COUNT)
        return None

    # For homography calculation
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    h, w = im1.shape[:2]
    corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

    # Calculate the homography using RANSAC
    #H, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=config.RANSAC_REPROJ_TRESHHOLD, maxIters=config.RANSAC_MAX_ITERS, confidence=config.RANSAC_CONFIDENCE)

    import ransac
    ransac_results = ransac.ransac(src_pts, dst_pts, corners)
    H = ransac_results["homography"]

    #if plate == 34:
    #    import pdb
    #    pdb.set_trace()

    # Check homography validity
    if H is None or len(H.shape) != 2:
        logger.debug("Couldn't get homography")
        return None

    # Calculate the homography determinant
    #det = np.linalg.det(H)

    # If it's too low for comfort, don't consider the match
    #if abs(det) < config.HOMOGRAPHY_DETERMINANT_THRESHOLD:
    #    logger.debug("Failed homography test")
    #    return None

    # Attempt to transform corners based on homography
    try:
        transformedCorners = cv2.perspectiveTransform(corners, H)
    except:
        logger.debug("Couldn't transform corners")
        return None

    # Only accept corners that remain convex after being transformed
    isConvex = cv2.isContourConvex(transformedCorners)
    if not isConvex and not config.ALLOW_NON_CONVEX_CORNERS:
        logger.debug("Not convex")
        return None

    # Get the moments
    original_moments = cv2.moments(corners)
    transformed_moments = cv2.moments(transformedCorners)

    # Get the 7 Hu invariant moments
    original_hu_moments = cv2.HuMoments(original_moments).flatten()
    transformed_hu_moments = cv2.HuMoments(transformed_moments).flatten()

    # Find the Euclidean distance between the moments
    hu_distance = np.linalg.norm(original_hu_moments - transformed_hu_moments)

    # Ignore moments that are too large
    if hu_distance > config.HU_DISTANCE_THRESHOLD:
        logger.debug("Failed hu moment test")
        return None

    # ******************** Image 1: All lines
    im_all_lines = cv2.drawMatches(im1, kp1, im2, kp2, good_matches, None, None, None, None, cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    # ******************** Image 2: Only inliers + corner rectangle
    # Get the inliers
    inlier_mask = ransac_results["inlier_mask"]

    # Draw a polygon on the second image joining the transformed corners
    im_corner_rectangle = cv2.polylines(im2, [np.int32(transformedCorners)], True, config.MATCH_RECT_COLOR, 2, cv2.LINE_AA)

    # Draw the lines between the 2 images connecting matches
    im_inliers = cv2.drawMatches(im1, kp1, im_corner_rectangle, kp2, good_matches, None, None, None, inlier_mask.tolist(), cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # ******************** Image 3: The original 4 points
    # Get the 4 points
    original_inlier_mask = ransac_results["original_inlier_mask"]

    im_original = cv2.drawMatches(im1, kp1, im2, kp2, good_matches, None, None, None, original_inlier_mask.tolist(), cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    original_pts = src_pts[original_inlier_mask]

    transformed_pts = None
    try:
        transformed_pts = cv2.perspectiveTransform(np.array([original_pts]), H).reshape(4, 2)
    except:
        logger.debug("Couldn't transform original 4 pts")
        im_original_2 = im2

    if transformed_pts is not None:
        import pylab as plt
        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.imshow(im2)
        axes.scatter(*zip(*transformed_pts), s=5, c='b')
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        from PIL import Image
        im_original_2 = np.array(Image.open(buf))

        buf.close()
        plt.close()

    # ******************** Image 4: Transformed overlay
    # Warp the first image onto the second image
    im_warp = cv2.warpPerspective(im1, H, (im2.shape[1],im2.shape[0]))

    # Find the pixels which are visible in the warped image
    overlay_mask = (im_warp != 0)

    # Create a copy of the second image
    im_overlay = im2.copy()

    # Overlay the warped image on top of the 2nd image
    # If the mask is grayscale the overlay will only be applied to the red channel
    if len(overlay_mask.shape) == 3: # Overlay colors
        im_overlay[overlay_mask] = im_warp[overlay_mask]
    else:
        im_overlay[overlay_mask, 0] = im_warp[overlay_mask]

    # End of image results
    im_result = (ImageInfo(im_all_lines, "All Lines", "all-lines"), ImageInfo(im_inliers, "Inliers Only", "inliers"), ImageInfo(im_original, "Original 4 Points", "orig-4"), ImageInfo(im_overlay, "Overlay", "overlay-warp"), ImageInfo(im_original_2, "4 points 2", ""))

    return Match(None, good_matches, ransac_results, im_result, original_moments, transformed_moments, hu_distance, isConvex)

def match_region_nissl(im_region, nissl_level):
    kp1, des1 = extract_sift(im_region)

    kp2, des2 = nissl_load_sift(nissl_level)
    im_nissl = nissl_load(nissl_level)

    m = match(im_region, kp1, des1, im_nissl, kp2, des2, nissl_level)
    if m is None:
        return None
    else:
        m.nissl_level = nissl_level
        logger.debug("Match accepted")
        return m

def match_sift_nissl(im_region, kp1, des1, nissl_level):
    logger.debug("{0} Matching {1}", "="*75, nissl_level)
    kp2, des2 = nissl_load_sift(nissl_level)
    im_nissl = nissl_load(nissl_level)

    m = match(im_region, kp1, des1, im_nissl, kp2, des2, nissl_level)

    if m is None:
        return None
    else:
        m.nissl_level = nissl_level
        return m

def extract_sift(im):
    if im.dtype != np.uint8:
            im = (im * 255).astype(np.uint8)

    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    # Use multithreading to split the affine detection work
    pool = ThreadPool(processes = cv2.getNumberOfCPUs())

    kp, des = affine_detect(SIFT, im, pool=pool)

    pool.close()
    pool.join()

    return kp, des


def affine_skew(tilt, phi, img, mask=None):
    '''
    affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai
    Ai - is an affine transform matrix from skew_img to img
    '''
    h, w = img.shape[:2]
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    A = np.float32([[1, 0, 0], [0, 1, 0]])
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c,-s], [ s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32( np.dot(corners, A.T) )
        x, y, w, h = cv2.boundingRect(tcorners.reshape(1,-1,2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if tilt != 1.0:
        s = 0.8*np.sqrt(tilt*tilt-1)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv2.resize(img, (0, 0), fx=1.0/tilt, fy=1.0, interpolation=cv2.INTER_NEAREST)
        A[0] /= tilt
    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv2.warpAffine(mask, A, (w, h), flags=cv2.INTER_NEAREST)
    Ai = cv2.invertAffineTransform(A)
    return img, mask, Ai

def affine_detect(detector, img, mask=None, pool=None):
    '''
    affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs
    Apply a set of affine transormations to the image, detect keypoints and
    reproject them into initial image coordinates.
    See http://www.ipol.im/pub/algo/my_affine_sift/ for the details.
    ThreadPool object may be passed to speedup the computation.
    '''
    params = [(1.0, 0.0)]
    for t in 2**(0.5*np.arange(1,6)):
        for phi in np.arange(0, 180, 72.0 / t):
            params.append((t, phi))

    def f(p):
        t, phi = p
        timg, tmask, Ai = affine_skew(t, phi, img)
        keypoints, descrs = detector.detectAndCompute(timg, tmask)
        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple( np.dot(Ai, (x, y, 1)) )
        if descrs is None:
            descrs = []
        return keypoints, descrs

    keypoints, descrs = [], []
    if pool is None:
        ires = map(f, params)
    else:
        ires = pool.map(f, params)

    for i, (k, d) in enumerate(ires):
        #logger.debug('affine sampling: %d / %d\r' % (i+1, len(params)))
        keypoints.extend(k)
        descrs.extend(d)

    return keypoints, np.array(descrs)

# https://isotope11.com/blog/storing-surf-sift-orb-keypoints-using-opencv-in-python
def pickle_sift(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        if i >= len(descriptors):
            break
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id, descriptors[i])
        i += 1
        temp_array.append(temp)
    return temp_array

def unpickle_sift(array):
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)