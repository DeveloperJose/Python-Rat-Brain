# -*- coding: utf-8 -*-
import cv2
import numpy as np
import sys
import sklearn.cluster

import config

SIFT = cv2.xfeatures2d.SIFT_create()
FLANN = cv2.FlannBasedMatcher(config.FLANN_INDEX_PARAMS, config.FLANN_SEARCH_PARAMS)
BF = cv2.BFMatcher(normType=cv2.NORM_L2)
class Match(object):
    def __init__(self, filename, matches):
        self.filename = filename
        self.matches = matches
        self.largest_match = min(matches, key=lambda x:x.distance)
        
    def comparison_key(self):
        return self.largest_match.distance
    
    def to_string_array(self):
        arr = np.array([
            self.filename,
            str(len(self.matches)),
            str(self.largest_match.distance),
            str(self.comparison_key())
            ])
        
        return arr

def match(filename, kp1, des1, kp2, des2, k):
    matches = FLANN.knnMatch(des2, des1, k)
    #matches = BF.knnMatch(des2, des1, k)

    # Apply Ratio Test
    matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * config.DISTANCE_RATIO]
    
    if len(matches) > config.MIN_MATCH_COUNT:    
        src_pts = np.float32([ kp2[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp1[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
        matchesMask = mask.ravel().tolist()
        
        if M is not None:
            pass
            #import pdb
            #pdb.set_trace()   
        return Match(filename, matches)
    else:
        return None

def extract_sift(im, use_grayscale=False):
    if use_grayscale:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    kp, des = SIFT.detectAndCompute(im, None)
    
    # Clustering test
    km = sklearn.cluster.KMeans(n_clusters=config.N_CLUSTERS)
    km.fit(des)
    
    return kp, km.cluster_centers_

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