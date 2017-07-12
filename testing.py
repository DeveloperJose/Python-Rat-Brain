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

def save_atlas_sift():
    for i in range(1, 71):
        im = feature.nissl_load(i)
        kp, des = feature.nissl_load_sift(i)
        im_kp = cv2.drawKeypoints(im, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        feature.im_write('SIFT-' + str(i) + ".jpg", im_kp)

def draw_kp(im):
    kp, des = feature.extract_sift(im)

    #kp = [k for k in kp if k.response > 0.08]
    #kp = [k for k in kp if k.size > 6]

    print("Keypoints: ", len(kp))
    im_kp = cv2.drawKeypoints(im, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(6,6))
    plt.imshow(im_kp)

    return kp, des


def match(im, atlas):
    kp, des = feature.extract_sift(im)
    #kp2, des2 = feature.extract_sift(atlas)
    kp2, des2 = feature.nissl_load_sift(33)
    match = feature.match(im, kp, des, atlas, kp2, des2)

    if match is None:
        return None

    print("Matches: ", len(match.matches))
    print("Inliers: ", match.inlier_count)

    #for im_info in match.im_results:
    #    plt.figure(figsize=(6,6))
    #    plt.title(im_info.title)
    #    plt.imshow(im_info.im)

    return match

filename = 'scripts_testing/region.jpg'
im_region = feature.im_read(filename)
im_region_gray = feature.im_read(filename, flags=cv2.IMREAD_GRAYSCALE)

im_atlas = feature.im_read('atlas_swanson/Level-64.jpg')

# *************** Resizing
#im_region = misc.imresize(im_region, (170, 310))
#im_region = misc.imresize(im_region, 10)

# *************** Convolution
#n = 6
#kernel = np.ones((n,n),np.float32)/(n**2)
#dst = cv2.filter2D(im_region,-1,kernel)

#draw_kp(feature.im_read('scripts_testing/region-34.jpg'))
#draw_kp(im_region)
kp, des = draw_kp(im_atlas)
#print("KP: ", len(kp))
# x.size or x.response
#kp_min = min(kp, key=lambda x:x.response)
#kp_max = max(kp, key=lambda x:x.response)
#print("Min/Max:", kp_min.size, kp_max.response)

#timing.stopwatch()
match = match(im_region, im_atlas)
#timing.stopwatch("Matching")