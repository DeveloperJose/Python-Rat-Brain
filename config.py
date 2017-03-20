# -*- coding: utf-8 -*-
NISSL_DIR = "nissl"
NISSL_EXT = ".jpg"

FLANN_INDEX_KDTREE = 0
FLANN_INDEX_PARAMS = dict(algorithm = FLANN_INDEX_KDTREE,
                          trees = 5)
                          
FLANN_SEARCH_PARAMS = dict(checks = 200)
                    
N_FEATURES = 1000
MIN_MATCH_COUNT = 15
RATIO = 0.85