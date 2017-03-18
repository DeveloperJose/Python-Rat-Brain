# -*- coding: utf-8 -*-
import cv2
import os
import cPickle as pickle
import numpy as np

from timeit import default_timer as timer

NISSL_DIR = "nissl"
NISSL_EXT = ".jpg"

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

# ******************************************* Main Program
sift = cv2.xfeatures2d.SIFT_create()

program_start = timer()
start_time = None

for filename in os.listdir(NISSL_DIR):
    if filename.endswith(NISSL_EXT):
        path = os.path.join(NISSL_DIR, filename)
        sift_path = os.path.splitext(path)[0]+'.sift'
        
        print "----- Processing ", path
        start_time = timer()
        
        im = cv2.imread(path)
        mask = None
        kp, des = sift.detectAndCompute(im, mask)
        
        print "Extracted in ", (timer() - start_time), "s"
        
        start_time = timer()
        temp = pickle_sift(kp, des)
        pickle.dump(temp, open(sift_path, "wb"))
        
        print "Pickling took ", (timer() - start_time), "s"
        
print "-----\nDone! Took ", (timer() - program_start), "s"