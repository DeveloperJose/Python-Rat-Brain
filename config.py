# -*- coding: utf-8 -*-
"""
=================================================
==================== Common Configuration
================================================= """
# Which will be the default atlas?
# Options: Swanson, PaxinosWatson
ATLAS = "Swanson"

# Should multithreading be used when matching?
# If True, the debug logs will not be in sequential order and PDB debugging will break
MULTITHREAD = False

# We will attempt to reduce the image width to this size but maintain aspect ratio
# Default: 200
RESIZE_WIDTH = 200

"""
=================================================
==================== Algorithm Selection
================================================= """
# Should we use our implemented RANSAC?
# If false, OpenCV RANSAC is used
NEW_RANSAC = True

"""
=================================================
==================== Matching
================================================= """
# Neighbor distance ratio for the ratio test as per Lowe's SIFT paper
DISTANCE_RATIO = 0.8

# Number of minimum good matches needed to compare descriptors
# You need at least 4 to be able to estimate a homography
MIN_MATCH_COUNT = 4

# Should feature matching use the FLANN KDTree approach?
# If false, feature matching will be with BruteForce
MATCH_WITH_FLANN = True

# The color of the rectangle overlayed in the matching
MATCH_RECT_COLOR = (0, 255, 255)

"""
=================================================
==================== Match Discarding
================================================= """
# Homography matrices whose absolute value determinant is lower will be discarded
# Default: 0.001
HOMOGRAPHY_DETERMINANT_THRESHOLD = 0.001

# When transforming corners using homography, should we allow non-convex shapes?
ALLOW_NON_CONVEX_CORNERS = False

# Moments larger than the threshold will be discarded
# Default: 2
HU_DISTANCE_THRESHOLD = 2e100

"""
=================================================
==================== RANSAC
================================================= """
# The higher the threshold, the lower the inliers
RANSAC_REPROJ_TRESHHOLD = 10

# How many RANSAC iterations should we perform?
# Default: 2000
RANSAC_MAX_ITERS = 2000

# Only OpenCV's RANSAC uses this
# Default: 0.99
RANSAC_CONFIDENCE = 0.99

"""
=================================================
==================== SIFT
================================================= """
# The larger the threshold, the less features are produced by the detector.
# Default: 0.08
SIFT_CONTRAST_THRESHOLD = 0.08

# The larger the threshold, the more features that are retained
# Default: 30
SIFT_EDGE_THRESHOLD = 10

# Sigma of Gaussian used by SIFT
# Reduce for images captured by a weak camera with soft lenses
# Default: 2
SIFT_SIGMA = 3

# Number of SIFT features to be extracted
# If 0, SIFT will decide the best number of features
SIFT_FEATURES = 0

# Larger means more features
SIFT_OCTAVE_LAYERS = 2

# Should we use ASIFT instead of regular SIFT?
USE_AFFINE = False

ASIFT_START = 0
ASIFT_END = 180
ASIFT_INC = 20.0

"""
=================================================
==================== Prototype User Interface
================================================= """
# Should the UI have region warping options?
UI_WARP = False

# Should the UI have region angle change options?
UI_ANGLE = False

# Should the UI show the region keypoints?
UI_SHOW_KP = True

# Should we save the region you select in the program?
UI_SAVE_REGION = False

# Should we save the results when they are selected?
UI_SAVE_RESULTS = True

"""
=================================================
==================== Logging
================================================= """
# The formatting string that will be used by the loggers
LOGGER_FORMAT_STRING = (
    #u'[{record.time:%H:%M}] '
    u'[{record.channel}:] {record.level_name}: {record.message}'
)

"""
=================================================
==================== FLANN-based Matcher
================================================= """
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_PARAMS = dict(algorithm=FLANN_INDEX_KDTREE,
                          trees=5)

FLANN_SEARCH_PARAMS = dict(checks=200)

"""
=================================================
==================== Atlas Automatic Set-Up
================================================= """
if (ATLAS.casefold == "Swanson".casefold):
    NISSL_DEFAULT_FILE = 'dataset/testing/region-34.jpg'
    NISSL_DIR = "dataset/atlas_swanson"
    NISSL_PREFIX = "Level-"
    NISSL_DIGITS = 2
    NISSL_COUNT = 73
    NISSL_EXT = ".jpg"
elif (ATLAS.casefold == "PaxinosWatson".casefold):
    NISSL_DEFAULT_FILE = 'dataset/atlas_pw/RBSC7-060.jpg'
    NISSL_DIR = 'dataset/atlas_pw'
    NISSL_PREFIX = 'RBSC7-'
    NISSL_DIGITS = 3
    NISSL_COUNT = 161
    NISSL_EXT = ".jpg"
else:
    raise Exception("Atlas: " + ATLAS + " is not supported")