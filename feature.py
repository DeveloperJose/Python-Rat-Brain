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
BF = cv2.BFMatcher(normType=cv2.NORM_L2)

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

def warp(im, points, disp_min, disp_max):
    h, w = im.shape[:2]

    # Include the corners
    src_pts = np.array([[0, 0], [w, 0], [0, h], [w, h]])
    dst_pts = np.array([[0, 0], [w, 0], [0, h], [w, h]])

    for i in range(points):
        rand_x = random.uniform(0, w)
        rand_y = random.uniform(0, h)
        p = np.array([rand_x, rand_y])
        rand_len = random.uniform(disp_min, disp_max)
        rand_angle = np.deg2rad(random.uniform(0, 360))

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
    #H, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    #return cv2.warpPerspective(im, H, (w, h))

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
    filename = "Level-" + str(nissl_level).zfill(2) + config.NISSL_EXT
    path = os.path.join(config.NISSL_DIR, filename)

    if not os.path.exists(path):
        print("Plate ", nissl_level, "not found")
        return None

    return im_read(path, color_flags)

def nissl_load_sift(nissl_level):
    filename = "Level-" + str(nissl_level).zfill(2) + ".sift"
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

def match(im_region, nissl_level):
    # im_region is colored
    if len(im_region.shape) == 3:
        if im_region.dtype != np.uint8:
            im_region = (im_region * 255).astype(np.uint8)

        im_region_gray = cv2.cvtColor(im_region, cv2.COLOR_RGB2GRAY)

    kp1, des1 = extract_sift(im_region_gray)
    kp2, des2 = nissl_load_sift(nissl_level)

    if config.MATCH_WITH_FLANN:
        matches = FLANN.knnMatch(des1, des2, k=2)
    else:
        matches = BF.knnMatch(des1, des2, k=2)

    # Apply Ratio Test
    good_matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * config.DISTANCE_RATIO]

    if len(good_matches) > config.MIN_MATCH_COUNT:
        im_nissl = nissl_load(nissl_level)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        # Obtain the homography matrix
        H, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        ravel = mask.ravel()
        matchesMask = mask.ravel().tolist()

        count = int(config.DISTANCE_RATIO * len(ravel == 1))
        for i in range(len(matchesMask)):
            if ravel[i] == 1:
                matchesMask[i] = 0
                count -= 1

            if count <= 0:
                break


        # Apply the perspective transformation to the source image corners
        h, w = im_region_gray.shape
        corners = np.float32([ [0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0] ]).reshape(-1, 1, 2)

        try:
            transformedCorners = cv2.perspectiveTransform(corners, H)
        except:
            print("transformed corners failed")
            return None

        # Draw a polygon on the second image joining the transformed corners
        im_nissl = cv2.polylines(im_nissl, [np.int32(transformedCorners)], True, config.MATCH_RECT_COLOR, 2, cv2.LINE_AA)

        drawParameters = dict(matchColor=config.MATCH_LINE_COLOR, singlePointColor=None, matchesMask=matchesMask, flags=2)
        try:
            result = cv2.drawMatches(im_region, kp1, im_nissl, kp2, good_matches, None, **drawParameters)

            im_out = cv2.warpPerspective(im_region, H, (im_nissl.shape[1],im_nissl.shape[0]))
            good_mask = im_out != 0
            result2 = im_nissl
            result2[good_mask] = im_out[good_mask]
        except:
            print("draw matches failed")
            result = im_nissl
            result2 = im_nissl

        return Match(nissl_level, good_matches, H, mask, result, result2)

    else:
        return None

def extract_sift(im):
    kp, des = SIFT.detectAndCompute(im, None)

    # Clustering test
    #km = sklearn.cluster.KMeans(n_clusters=config.N_CLUSTERS)
    #km.fit(des)
    #des = km.cluster_centers_

    return kp, des

# https://isotope11.com/blog/storing-surf-sift-orb-keypoints-using-opencv-in-python
def pickle_sift(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
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