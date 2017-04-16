# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import pickle
import random
import config

from skimage.color import rgb2grey

SIFT = cv2.xfeatures2d.SIFT_create()
FLANN = cv2.FlannBasedMatcher(config.FLANN_INDEX_PARAMS, config.FLANN_SEARCH_PARAMS)
BF = cv2.BFMatcher(normType=cv2.NORM_HAMMING)

class Match(object):
    def __init__(self, nissl_level, matches, H, mask, result, result2):
        self.nissl_level = nissl_level
        self.matches = matches
        self.largest_match = max(matches, key=lambda x:x.distance)
        self.H = H
        self.mask = mask
        self.inlier_count = mask.sum()
        self.result = result
        self.result2 = result2

    def comparison_key(self):
        return self.inlier_count

    def to_string_array(self):
        arr = np.array([
            "Plate #" + str(self.nissl_level),
            str(len(self.matches)),
            str(self.inlier_count),
            str(self.largest_match.distance)
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
    H, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    matchesMask = mask.ravel().tolist()

    # Apply the perspective transformation to the source image corners
    h, w = im1.shape[:2]
    corners = np.float32([ [0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0] ]).reshape(-1, 1, 2)

    try:
        transformedCorners = cv2.perspectiveTransform(corners, H)
        # Draw a polygon on the second image joining the transformed corners
        im2 = cv2.polylines(im2, [np.int32(transformedCorners)], True, config.MATCH_RECT_COLOR, 2, cv2.LINE_AA)
    except:
        print("transformed corners failed")
        return None

    # SIFT line matching
    drawParameters = dict(matchColor=config.MATCH_LINE_COLOR, singlePointColor=None, matchesMask=matchesMask, flags=2)
    result = cv2.drawMatches(im1, kp1, im2, kp2, good_matches, None, **drawParameters)

    # Warp the first image onto the second image
    im_out = cv2.warpPerspective(im1, H, (im2.shape[1],im2.shape[0]))
    good_mask = (im_out != 0)
    result2 = im2
    print("r2", result2.shape, "im_out", im_out.shape, "gm", good_mask.shape)

    if len(good_mask.shape) == 3: # Color
        result2[good_mask] = im_out[good_mask]
    else:
        result2[good_mask, 0] = im_out[good_mask]

    return Match(None, good_matches, H, mask, result, result2)

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

def extract_sift(im):
    if im.dtype != np.uint8:
            im = (im * 255).astype(np.uint8)

    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    kp, des = SIFT.detectAndCompute(im, None)
    #star = cv2.xfeatures2d.SIFT_create()
    #brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    #kp = star.detect(im, None)
    #kp, des = brief.compute(im, kp)

    # Clustering test
    #from sklearn.cluster import KMeans
    #print("Clustering...")
    #km = KMeans(n_clusters=50)
    #km.fit(des)
    #print("Done")
    #des = km.cluster_centers_

    return kp, des

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