# -*- coding: utf-8 -*-
import cv2
import os
import pickle
import numpy as np
from timeit import default_timer as timer

NISSL_DIR = "nissl"
NISSL_EXT = ".jpg"

# https://isotope11.com/blog/storing-surf-sift-orb-keypoints-using-opencv-in-python    
def unpickle_sift(array):
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)

# ***** Main Program
sift = cv2.xfeatures2d.SIFT_create()

im_region = cv2.imread("nissl/part.jpg")
gray = cv2.cvtColor(im_region, cv2.COLOR_BGR2GRAY)
kp1, des1 = sift.detectAndCompute(gray, None)

best_filename = None
best_count = -1
distance = 1000

for filename in os.listdir(NISSL_DIR):
    if filename.endswith(".sift"):
        print ("********** Processing", filename)
        path = os.path.join(NISSL_DIR, filename)
        raw_sift = pickle.load(open(path, "rb"))
        kp2, des2 = unpickle_sift(raw_sift)
        
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=100)   # or pass empty dictionary
        
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        
        matches = flann.knnMatch(des2,des1,k=2)
        
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
                
        print ("Matches", len(matchesMask))
        if len(matchesMask) > best_count:
            best_filename = path
            best_count = len(matchesMask)
                
print ("** Best Match:", best_filename, "Count:", best_count)