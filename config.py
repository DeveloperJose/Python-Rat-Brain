# -*- coding: utf-8 -*-
FLANN_INDEX_KDTREE = 0
FLANN_INDEX_PARAMS = dict(algorithm=FLANN_INDEX_KDTREE,
                          trees=5)

FLANN_SEARCH_PARAMS = dict(checks=200)

NISSL_DIR = "nissl"
NISSL_EXT = ".jpg"
NISSL_COUNT = 73

# If false it will match using BruteForce
MATCH_WITH_FLANN = True

# Number of minimum good matches needed to compare descriptors
MIN_MATCH_COUNT = 15

# Neighbor distance ratio for the ratio test as per Lowe's SIFT paper
DISTANCE_RATIO = 0.8

# What percentage of matched keypoints should we draw?
MATCH_KEYPOINTS_PERCENTAGE = 0.50

MATCH_RECT_COLOR = (0, 255, 255)
MATCH_LINE_COLOR = (0, 255, 0)