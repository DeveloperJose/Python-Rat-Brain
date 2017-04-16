# -*- coding: utf-8 -*-
FLANN_INDEX_KDTREE = 0
FLANN_INDEX_PARAMS = dict(algorithm=FLANN_INDEX_KDTREE,
                          trees=5)

FLANN_SEARCH_PARAMS = dict(checks=200)

#NISSL_DEFAULT_FILE = 'RBSC7-060.jpg'
#NISSL_DIR = 'atlas_paxinos_watson_25'
#NISSL_PREFIX = 'RBSC7-'
#NISSL_DIGITS = 3
#NISSL_COUNT = 161

NISSL_DEFAULT_FILE = 'Level-34.jpg'
NISSL_DIR = "atlas_swanson"
NISSL_PREFIX = "Level-"
NISSL_DIGITS = 2
NISSL_COUNT = 73

NISSL_EXT = ".jpg"

SAVE_REGION = True

# If false it will match using BruteForce
MATCH_WITH_FLANN = True

# Number of minimum good matches needed to compare descriptors
MIN_MATCH_COUNT = 1

# Neighbor distance ratio for the ratio test as per Lowe's SIFT paper
DISTANCE_RATIO = 0.8

# What percentage of matched keypoints should we draw? [0.0 - 1.0]
MATCH_KEYPOINTS_PERCENTAGE = 1

MATCH_RECT_COLOR = (0, 255, 255)
MATCH_LINE_COLOR = (0, 255, 0)