# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pylab as plt

import scipy.ndimage.filters as filters
import scipy.signal as signal
import scipy.misc as misc

import feature
import timing

import logbook
logger = logbook.Logger(__name__)

sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.08, edgeThreshold=30, sigma=5)
affine = True

def draw_kp(im):
    kp, des = feature.extract_sift(im)

    #kp = [k for k in kp if k.response > 0.09]
    #kp = [k for k in kp if k.size > 6]

    print("Keypoints: ", len(kp))
    im_kp = cv2.drawKeypoints(im, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(10,10))
    plt.imshow(im_kp)

    return kp, des


def match(im, atlas):
    kp, des = feature.extract_sift(im)
    kp2, des2 = feature.extract_sift(atlas)

    match = feature.match(im, kp, des, atlas, kp2, des2)
    print("Matches: ", len(match.matches))
    print("Inliers: ", match.inlier_count)

    for im_info in match.im_results:
         plt.figure(figsize=(10,10))
         plt.imshow(im_info.im)

    return match.H

filename = 'scripts_testing/region.jpg'
im_region = feature.im_read(filename)
im_region_gray = feature.im_read(filename, flags=cv2.IMREAD_GRAYSCALE)

im_atlas = feature.im_read('atlas_swanson/Level-33.jpg')

# *************** Resizing
#im_region = misc.imresize(im_region, (170, 310))
#im_region = misc.imresize(im_region, 10)

# *************** Convolution
#n = 6
#kernel = np.ones((n,n),np.float32)/(n**2)
#dst = cv2.filter2D(im_region,-1,kernel)

#draw_kp(feature.im_read('scripts_testing/region-34.jpg'))
#kp, des = draw_kp(im_region)
draw_kp(im_atlas)
#kp_min = min(kp, key=lambda x:x.size)
#kp_max = max(kp, key=lambda x:x.size)
#print("Min/Max:", kp_min.size, kp_max.size)

#timing.stopwatch()
#H = match(im_region, im_atlas)
#timing.stopwatch("Matching")