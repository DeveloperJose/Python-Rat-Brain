# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pylab as plt
import feature

import scipy.ndimage.filters as filters
import scipy.signal as signal
import scipy.misc as misc



sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.08, edgeThreshold=30, sigma=2)
affine = True
#sift = cv2.xfeatures2d.SIFT_create()
def draw_kp(im):
    if affine:
        from multiprocessing.pool import ThreadPool
        pool=ThreadPool(processes = cv2.getNumberOfCPUs())
        kp, des = affine_detect(sift, im, pool=pool)
    else:
        kp = sift.detect(im)

    print("Keypoints: ", len(kp))
    im_kp = cv2.drawKeypoints(im, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(10,10))
    plt.imshow(im_kp)

def match(im):
    atlas = feature.im_read('testing/region-34.jpg')
    if affine:
        from multiprocessing.pool import ThreadPool

        pool=ThreadPool(processes = cv2.getNumberOfCPUs())
        kp, des = affine_detect(sift, im, pool=pool)

        kp2, des2 = affine_detect(sift, atlas, pool=pool)

    else:
        kp, des = sift.detectAndCompute(im, None)
        kp2, des2 = sift.detectAndCompute(atlas, None)

    match = feature.match(im, kp, des, atlas, kp2, des2)
    print("Matches: ", len(match.matches))
    print("Inliers: ", match.inlier_count)
    plt.figure(figsize=(10,10))
    plt.imshow(match.result)
    plt.figure(figsize=(10,10))
    plt.imshow(match.result2)

    return match.H


filename = 'testing/region-70-HQ.jpg'
im_region = feature.im_read(filename)
im_region_gray = feature.im_read(filename, flags=cv2.IMREAD_GRAYSCALE)

# *************** Resizing
im_region = misc.imresize(im_region, (170, 310))

# *************** Convolution
#n = 6
#kernel = np.ones((n,n),np.float32)/(n**2)
#dst = cv2.filter2D(im_region,-1,kernel)

#dst = im_region
#draw_kp(im_region)
#draw_kp(feature.im_read('testing/region-34.jpg'))
affine = True
draw_kp(feature.im_read('testing/region-34.jpg'))
H = match(im_region)


# ****************** Other tests
#plt.figure(figsize=(10,10))
#plt.imshow(im_region)

#im_region_gray = cv2.bilateralFilter(im_region_gray, 30, 100, 100)
#im_region_gray = cv2.equalizeHist(im_region_gray)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,4))
#im_region_gray = cv2.dilate(im_region_gray, kernel, iterations=2)
#im_region_gray = cv2.erode(im_region_gray, kernel, iterations=3)
#im_region_gray = cv2.morphologyEx(im_region_gray, cv2.MORPH_CLOSE, kernel)
#plt.figure(figsize=(10,10))
#plt.imshow(im_region_gray)

#kp1, des1 = feature.extract_sift(im_region_gray)
#print("Keypoints: ", len(kp1))
#im_kp = cv2.drawKeypoints(im_region_gray, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#plt.figure(figsize=(10,10))
#plt.imshow(im_kp)

#im_nissl = feature.nissl_load(34)
#im_nissl_gray = feature.nissl_load(34, cv2.IMREAD_GRAYSCALE)
#im_nissl = feature.im_read('testing/region-34.jpg')
#im_nissl_gray = feature.im_read('testing/region-34.jpg', cv2.IMREAD_GRAYSCALE)

#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 4))
#cv2.equalizeHist(im_nissl_gray)
#im_nissl_gray = cv2.morphologyEx(im_nissl_gray, cv2.MORPH_CLOSE, kernel)

#im_nissl_gray = filters.gaussian_filter(im_nissl_gray, 2)
#im_nissl = filters.median_filter(im_nissl, 5)
#kp2, des2 = feature.extract_sift(im_nissl_gray)
#im_nissl_kp = cv2.drawKeypoints(im_nissl_gray, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#plt.figure(figsize=(10,10))
#plt.imshow(im_nissl_kp)

#match = feature.match(im_region, kp1, des1, im_nissl, kp2, des2)
#print("Matches: ", len(match.matches))
#print("Inliers: ", match.inlier_count)
#plt.figure(figsize=(10,10))
#plt.imshow(match.result)
#plt.figure(figsize=(10,10))
#plt.imshow(match.result2)