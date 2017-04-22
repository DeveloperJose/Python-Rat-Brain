# -*- coding: utf-8 -*-
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_PARAMS = dict(algorithm=FLANN_INDEX_KDTREE,
                          trees=5)

FLANN_SEARCH_PARAMS = dict(checks=200)

ATLAS = "SWANSON"

if (ATLAS == "SWANSON"):
    NISSL_DEFAULT_FILE = 'Level-34.jpg'
    NISSL_DIR = "atlas_swanson"
    NISSL_PREFIX = "Level-"
    NISSL_DIGITS = 2
    NISSL_COUNT = 73
else:
    NISSL_DEFAULT_FILE = 'RBSC7-060.jpg'
    NISSL_DIR = 'atlas_paxinos_watson_25'
    NISSL_PREFIX = 'RBSC7-'
    NISSL_DIGITS = 3
    NISSL_COUNT = 161

MULTITHREAD = False

LOGGER_FORMAT_STRING = (
    #u'[{record.time:%H:%M}] '
    u'[{record.channel}:] {record.level_name}: {record.message}'
)

# We will attempt to reduce the images to this size but maintain aspect ratio
RESIZE_WIDTH = 200

NISSL_EXT = ".jpg"

UI_WARP = False
UI_ANGLE = False

# Should we save the region you select in the program?
SAVE_REGION = False

# If false it will match using BruteForce
MATCH_WITH_FLANN = True

# Number of minimum good matches needed to compare descriptors
MIN_MATCH_COUNT = 1

# Neighbor distance ratio for the ratio test as per Lowe's SIFT paper
DISTANCE_RATIO = 0.85

MATCH_RECT_COLOR = (0, 255, 255)
MATCH_LINE_COLOR = (0, 255, 0)