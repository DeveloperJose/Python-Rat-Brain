# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import pickle
import random
import config

from multiprocessing.pool import ThreadPool

SIFT = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.08, edgeThreshold=30, sigma=2)
FLANN = cv2.FlannBasedMatcher(config.FLANN_INDEX_PARAMS, config.FLANN_SEARCH_PARAMS)
BF = cv2.BFMatcher(normType=cv2.NORM_HAMMING)

class Match(object):
    def __init__(self, nissl_level, matches, H, mask, result, result2, area_ratio):
        self.nissl_level = nissl_level
        self.matches = matches
        self.largest_match = max(matches, key=lambda x:x.distance)
        self.H = H
        self.mask = mask
        self.inlier_count = mask.sum()
        self.result = result
        self.result2 = result2
        self.area_ratio = area_ratio

    def comparison_key(self):
        return self.inlier_count

    def to_string_array(self):
        svd = np.linalg.svd(self.H, compute_uv=False)
        arr = np.array([
            "Plate #" + str(self.nissl_level),
            str(len(self.matches)),
            str(self.inlier_count),
            str(svd[-1] / svd[0])
            ])

        return arr

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
        print("Plate ", nissl_level, "not found")
        return None

    return im_read(path, color_flags)

def nissl_load_sift(nissl_level):
    filename = config.NISSL_PREFIX + str(nissl_level).zfill(config.NISSL_DIGITS) + ".sift"
    path = os.path.join(config.NISSL_DIR, filename)

    if not os.path.exists(path):
        print("Creating SIFT for ", str(nissl_level))
        nissl = nissl_load(nissl_level, cv2.IMREAD_GRAYSCALE)
        if nissl is None:
            return None

        kp, des = extract_sift(nissl)
        temp = pickle_sift(kp, des)
        pickle.dump(temp, open(path, "wb"))
        return kp, des

    else:
        raw_sift = pickle.load(open(path, "rb"))
        kp, des = unpickle_sift(raw_sift)
        return kp, des

def match(im1, kp1, des1, im2, kp2, des2):
    if config.MATCH_WITH_FLANN:
        matches = FLANN.knnMatch(des1, des2, k=2)
    else:
        matches = BF.knnMatch(des1, des2, k=2)

    # Apply Ratio Test
    good_matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * config.DISTANCE_RATIO]

    if len(good_matches) < config.MIN_MATCH_COUNT:
        return None

    # For homography calculation
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Obtain the homography matrix
    H, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=50, maxIters=2000, confidence=0.99)
    matchesMask = mask.ravel().tolist()

    if H is None or len(H.shape) != 2:
        print("couldn't get homography")
        return None

    det = np.linalg.det(H)
    if(abs(det) > 10):
        print("det high", det)
        return None

    # Apply the perspective transformation to the source image corners
    h, w = im1.shape[:2]
    corners = np.float32([ [0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0] ]).reshape(-1, 1, 2)

    try:
        transformedCorners = cv2.perspectiveTransform(corners, H)

        original_area = cv2.contourArea(corners)
        transformed_area = cv2.contourArea(transformedCorners)
        area_ratio = transformed_area / original_area

        #if (area_ratio > 2 or area_ratio < 0.01):
            #print("horrible match from areas", "ratio: ", area_ratio)
            #return None

        # Draw a polygon on the second image joining the transformed corners
        im2 = cv2.polylines(im2, [np.int32(transformedCorners)], True, config.MATCH_RECT_COLOR, 2, cv2.LINE_AA)
    except:
        area_ratio = 0
        print("transformed corners failed")
        return None

    # SIFT line matching
    drawParameters = dict(matchColor=-1, singlePointColor=None, matchesMask=matchesMask, flags=2)
    result = cv2.drawMatches(im1, kp1, im2, kp2, good_matches, None, **drawParameters)

    # Warp the first image onto the second image
    im_out = cv2.warpPerspective(im1, H, (im2.shape[1],im2.shape[0]))
    good_mask = (im_out != 0)
    result2 = im2
    #print("r2", result2.shape, "im_out", im_out.shape, "gm", good_mask.shape)

    if len(good_mask.shape) == 3: # Color
        result2[good_mask] = im_out[good_mask]
    else:
        result2[good_mask, 0] = im_out[good_mask]

    return Match(None, good_matches, H, mask, result, result2, area_ratio)

def match_region_nissl(im_region, nissl_level):
    kp1, des1 = extract_sift(im_region)

    kp2, des2 = nissl_load_sift(nissl_level)
    im_nissl = nissl_load(nissl_level)

    m = match(im_region, kp1, des1, im_nissl, kp2, des2)
    if m is None:
        return None
    else:
        m.nissl_level = nissl_level
        return m

def match_sift_nissl(im_region, kp1, des1, nissl_level):
    kp2, des2 = nissl_load_sift(nissl_level)
    im_nissl = nissl_load(nissl_level)

    m = match(im_region, kp1, des1, im_nissl, kp2, des2)
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

    #kp, des = SIFT.detectAndCompute(im, None)
    print("Extracting SIFT")
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
        print('affine sampling: %d / %d\r' % (i+1, len(params)), end='')
        keypoints.extend(k)
        descrs.extend(d)

    print()
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