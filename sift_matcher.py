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

for filename in os.listdir(config.NISSL_DIR):
    if filename.endswith(".sift"):
        print ("********** Processing", filename)
        path = os.path.join(config.NISSL_DIR, filename)
        raw_sift = pickle.load(open(path, "rb"))
        kp2, des2 = feature.unpickle_sift(raw_sift)    
        
        matches, largest_distance = feature.match(kp1, des1, kp2, des2, k=2)
                
        print ("Matches", len(matches))
        if len(matches) > best_count and largest_distance < best_distance:
            best_filename = path
            best_count = len(matches)
            best_distance = largest_distance
                
print ("** Best Match:", best_filename, "Count:", best_count, "Dis:", best_distance)