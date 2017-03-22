# -*- coding: utf-8 -*-
import cv2
import os
import pickle
import numpy as np
from timeit import default_timer as timer

import config, feature

program_start = timer()

for filename in os.listdir(config.NISSL_DIR):
    if filename.endswith(config.NISSL_EXT):
        path = os.path.join(config.NISSL_DIR, filename)
        sift_path = os.path.splitext(path)[0]+'.sift'
        
        print ("----- Processing ", path)
        im = cv2.imread(path)
        kp, des = feature.extract_sift(im, False)
        
        print ("* Descriptors:", len(des))
        temp = feature.pickle_sift(kp, des)
        pickle.dump(temp, open(sift_path, "wb"))
        
        print ("* Pickled and saved")

program_duration = timer() - program_start
print ("-----\nDone! Took ", program_duration, "s")