# -*- coding: utf-8 -*-
import cv2
import os
import pickle
import numpy as np
import sys
from timeit import default_timer as timer

import config, feature

# ***** Main Program
im_region = cv2.imread("nissl/part.jpg")
kp1, des1 = feature.extract_sift(im_region, True)

best_filename = None
best_count = -1
best_distance = sys.maxsize
results = None

descriptors = None

BOW = cv2.BOWKMeansTrainer(100)

FLANN = cv2.FlannBasedMatcher(config.FLANN_INDEX_PARAMS, config.FLANN_SEARCH_PARAMS)

for filename in os.listdir(config.NISSL_DIR):
    if filename.endswith(".sift"):
        print ("********** Processing", filename)
        path = os.path.join(config.NISSL_DIR, filename)
        raw_sift = pickle.load(open(path, "rb"))
        kp2, des2 = feature.unpickle_sift(raw_sift)    
        
        #sift2 = cv2.xfeatures2d.SIFT_create()
        #cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
        
        #BOW.add(des2)
        
        FLANN.add(des2)
        
print ("Clustering BOW")
#dictionary = BOW.cluster()

print ("Training FLANN")
FLANN.train()

