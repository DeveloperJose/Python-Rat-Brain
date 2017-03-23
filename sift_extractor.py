# -*- coding: utf-8 -*-
import cv2
import os
import pickle
import numpy as np
import sklearn.cluster
from timeit import default_timer as timer

import config, feature

program_start = timer()

for filename in os.listdir(config.NISSL_DIR):
    if filename.endswith(config.NISSL_EXT):
        path = os.path.join(config.NISSL_DIR, filename)
        sift_path = os.path.splitext(path)[0]+'.sift'
        cluster_path = os.path.splitext(path)[0]+'.cluster'
        
        print ("----- Processing ", path)
        im = cv2.imread(path, 0)
        kp, des = feature.extract_sift(im)
        
        #km = sklearn.cluster.KMeans(n_clusters=config.N_CLUSTERS)
        #km.fit(des)
        #print ("* Calculating cluster centers", config.N_CLUSTERS)
        
        print ("* Descriptors:", len(des))
        temp = feature.pickle_sift(kp, des)
        pickle.dump(temp, open(sift_path, "wb"))
        #pickle.dump(km.cluster_centers_, open(cluster_path, "wb"))
        
        print ("* Pickled and saved")
        break

program_duration = timer() - program_start
print ("-----\nDone! Took ", program_duration, "s")