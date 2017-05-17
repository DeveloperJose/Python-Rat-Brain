# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pylab as plt
import csv

import scipy.misc as misc

import feature
import timing
import config

import threading
from multiprocessing.pool import ThreadPool
import logbook
import sys
import os
import pickle
logger = logbook.Logger(__name__)
logbook.StreamHandler(sys.stdout, level=logbook.DEBUG).push_application()

def get_sift(atlas, level):
        if atlas == "SW":
            filename = 'Level-' + str(level).zfill(2) + ".sift"
            path = os.path.join('atlas_swanson', filename)
        else:
            filename = 'RBSC7-' + str(level).zfill(3) + ".sift"
            path = os.path.join('atlas_pw', filename)

        raw_sift = pickle.load(open(path, "rb"))
        kp, des = feature.unpickle_sift(raw_sift)
        return kp, des

def get_im(atlas, level):
    if atlas == "SW":
        filename = 'Level-' + str(level).zfill(2) + ".jpg"
        path = os.path.join('atlas_swanson', filename)

        return feature.im_read(path)
    else:
        filename = 'RBSC7-' + str(level).zfill(3) + ".jpg"
        path = os.path.join('atlas_pw', filename)

        im = feature.im_read(path)

        old_shape = im.shape
        reduction_percent = int(config.RESIZE_WIDTH/old_shape[0] * 100)
        im = misc.imresize(im, reduction_percent)
        logger.debug("Resized region from {0} to {1}", old_shape, im.shape)

        return im

pw_im = get_im("PW", 68)
pw_kp, pw_des = get_sift("PW", 68)
pw_kp_im = cv2.drawKeypoints(pw_im, pw_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

sw_im = get_im("SW", 33)
sw_kp, sw_des = get_sift("SW", 33)
sw_kp_im = cv2.drawKeypoints(sw_im, sw_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

match = feature.match(pw_im, pw_kp, pw_des, sw_im, sw_kp, sw_des)

if match is None:
    logger.debug("No matches :(")
else:
    logger.debug("Matches {}", match.matches_count)
    logger.debug("Inliers {}", match.inlier_count)

    figure = plt.figure(figsize=(12, 12))
    axes = plt.gca()
    axes.set_xticks(())
    axes.set_yticks(())
    plt.imshow(sw_kp_im)

    figure = plt.figure(figsize=(12, 12))
    axes = plt.gca()
    axes.set_xticks(())
    axes.set_yticks(())
    axes.imshow(pw_kp_im)

    figure = plt.figure(figsize=(12, 12))
    axes = plt.gca()
    axes.set_xticks(())
    axes.set_yticks(())
    axes.imshow(match.result)

    figure = plt.figure(figsize=(12, 12))
    axes = plt.gca()
    axes.set_xticks(())
    axes.set_yticks(())
    axes.imshow(match.result2)

    plt.show()