# -*- coding: utf-8 -*-
import cv2
import numpy as np
import sys

import config

SIFT = cv2.xfeatures2d.SIFT_create()
FLANN = cv2.FlannBasedMatcher(config.FLANN_INDEX_PARAMS, config.FLANN_SEARCH_PARAMS)

index = 0
def match(kp1, des1, kp2, des2, k):
    global index
    matches = FLANN.knnMatch(des2, des1, k)
        
    print ("matches", len(matches))
    matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * config.RATIO]
    
    print ("ratio test", len(matches))
    
    if len(matches) > config.MIN_MATCH_COUNT:
        largest_match = max(matches, key=lambda x:x.distance)
        return matches, largest_match.distance
    else:
        return [], sys.maxsize
    
    #p1 = np.float32([ kp1[m.trainIdx].pt for m in matches ])
    #p2 = np.float32([ kp2[m.queryIdx].pt for m in matches ])

    #H, status = cv2.findHomography(p1, p2, cv2.RANSAC, ransacReprojThreshold=5.0)
    #H2, status2 = cv2.findHomography(p2, p1, cv2.RANSAC, ransacReprojThreshold=5.0)
    
    #status = np.nonzero(status.ravel())
    #if status.sum() < config.MIN_MATCH_COUNT:
        #return []

def extract_sift(im, use_grayscale=False):
    if use_grayscale:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    return SIFT.detectAndCompute(im, None)

# https://isotope11.com/blog/storing-surf-sift-orb-keypoints-using-opencv-in-python        
def pickle_sift(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id, descriptors[i])     
        ++i
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