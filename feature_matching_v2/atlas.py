# -*- coding: utf-8 -*-
import os
import cv2
import pickle
import logbook
logger = logbook.Logger(__name__)

import util
import config
import util_sift

def load_image(nissl_level, color_flags=cv2.IMREAD_COLOR):
    filename = config.NISSL_PREFIX + str(nissl_level).zfill(config.NISSL_DIGITS) + config.NISSL_EXT
    path = os.path.join(config.NISSL_DIR, filename)

    if not os.path.exists(path):
        logger.error("Tried to load plate {0} but it wasn't found in path {1}", nissl_level, path)
        return None

    return util.im_read(path, color_flags)

def load_sift(nissl_level):
    path = get_path(nissl_level, ext='.sift')

    if not os.path.exists(path):
        return __create_sift(nissl_level, path)
    else:
        return __load_sift(path)

def get_path(nissl_level, ext='.jpg'):
    filename = config.NISSL_PREFIX + str(nissl_level).zfill(config.NISSL_DIGITS) + ext
    return os.path.join(config.NISSL_DIR, filename)

def __create_sift(nissl_level, path):
    # Check if the plate image exists and load it
    logger.info("Creating SIFT for plate {0}", nissl_level)
    im_nissl = load_image(nissl_level, cv2.IMREAD_GRAYSCALE)

    if im_nissl is None:
        logger.debug("[Atlas.CreateSift] {} doesn't exist", path)
        return None

    #import scipy.misc as misc
    #old_shape = nissl.shape
    #reduction_percent = int(config.RESIZE_WIDTH/old_shape[0] * 100)
    #nissl = misc.imresize(nissl, reduction_percent)
    #logger.debug("Resized region from {0} to {1}", old_shape, nissl.shape)

    # Extract SIFT from the image and pickle it
    kp, des = util_sift.extract_sift(im_nissl)
    pickled_sift = util_sift.pickle_sift(kp, des)

    # Save the data to the path
    pickle.dump(pickled_sift, open(path, "wb"))
    return kp, des

def __load_sift(path):
    if not os.path.exists(path):
        logger.debug("[Atlas.LoadSift] {} doesn't exist", path)
        return None, None

    raw_sift = pickle.load(open(path, "rb"))
    kp, des = util_sift.unpickle_sift(raw_sift)
    return kp, des