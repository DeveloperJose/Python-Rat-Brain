# -*- coding: utf-8 -*-
import cv2
import os
import pickle
import numpy as np
import sys
import pylab as plt
import sklearn.cluster
from timeit import default_timer as timer

import config, feature

# ***** Main Program
im_region = cv2.imread("nissl/part.jpg")
kp1, des1 = feature.extract_sift(im_region, True)
#des1 = np.array([d/np.linalg.norm(d) for d in des1])
km = sklearn.cluster.KMeans(n_clusters=config.N_CLUSTERS)
km.fit(des1)

best_filename = None
best_count = -1
best_distance = sys.maxsize
results = None

descriptors = None
kmeans = None
centers = None

for filename in os.listdir(config.NISSL_DIR):
    if filename.endswith(".sift"):
        print ("********** Processing", filename)
        path = os.path.join(config.NISSL_DIR, filename)
        cluster_path = os.path.splitext(path)[0]+'.cluster'
        raw_sift = pickle.load(open(path, "rb"))
        centers2 = pickle.load(open(cluster_path, "rb"))
        
        kp2, des2 = feature.unpickle_sift(raw_sift)    
        
        m2 = feature.match(filename, kp1, km.cluster_centers_, kp2, centers2, k=2)
        
        #match = feature.match(filename, kp1, des1, kp2, des2, k=2)
        
        #if match is None:
            #continue
        
        #km = sklearn.cluster.KMeans(n_clusters=4)
        #km.fit(des2)
        #if results is None:
            #results = np.array([match])
            
            #kmeans = np.array([km])
            #centers = np.array([km.cluster_centers_])
        #else:
            #results = np.hstack((results, np.array([match])))
            #kmeans = np.hstack((kmeans, np.array([km])))
            #centers = np.hstack((centers, np.array([km.cluster_centers_])))
            
#results = sorted(results, key=lambda x:x.comparison_key())