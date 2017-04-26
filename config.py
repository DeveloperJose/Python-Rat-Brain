# -*- coding: utf-8 -*-
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_PARAMS = dict(algorithm=FLANN_INDEX_KDTREE,
                          trees=5)

FLANN_SEARCH_PARAMS = dict(checks=200)

ATLAS = "SWANSON"
NISSL_EXT = ".jpg"

if (ATLAS == "SWANSON"):
    NISSL_DEFAULT_FILE = 'Level-34.jpg'
    NISSL_DIR = "atlas_swanson"
    NISSL_PREFIX = "Level-"
    NISSL_DIGITS = 2
    NISSL_COUNT = 73
else:
    NISSL_DEFAULT_FILE = 'RBSC7-060.jpg'
    NISSL_DIR = 'atlas_pw'
    NISSL_PREFIX = 'RBSC7-'
    NISSL_DIGITS = 3
    NISSL_COUNT = 161

# Homography matrices whose determinant is lower will be discarded
HOMOGRAPHY_DETERMINANT_THRESHOLD = 0.001

HU_DISTANCE_THRESHOLD = 2

RANSAC_REPROJ_TRESHHOLD = 50
RANSAC_MAX_ITERS = 2000
RANSAC_CONFIDENCE = 0.99

SIFT_CONTRAST_THRESHOLD = 0.08
SIFT_EDGE_THRESHOLD = 30
SIFT_SIGMA = 2

# Should multithreading be used when matching?
# If True, the debug logs will not be in sequential order
MULTITHREAD = True

# The formatting string that will be used by the loggers
LOGGER_FORMAT_STRING = (
    #u'[{record.time:%H:%M}] '
    u'[{record.channel}:] {record.level_name}: {record.message}'
)

# We will attempt to reduce the image width to this size but maintain aspect ratio
RESIZE_WIDTH = 200

# Should the UI have region warping options?
UI_WARP = False

# Should the UI have region angle change options?
UI_ANGLE = False

# Should the UI show the region keypoints?
UI_SHOW_KP = False

# Should we save the region you select in the program?
UI_SAVE_REGION = False

# Should feature matching use the FLANN KDTree approach?
# If false, feature matching will be with BruteForce
MATCH_WITH_FLANN = True

# Number of minimum good matches needed to compare descriptors
MIN_MATCH_COUNT = 10

# Neighbor distance ratio for the ratio test as per Lowe's SIFT paper
DISTANCE_RATIO = 0.85

# The color of the rectangle overlayed in the matching
MATCH_RECT_COLOR = (0, 255, 255)