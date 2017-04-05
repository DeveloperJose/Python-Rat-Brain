# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pylab as plt
import random

import feature

# Vars
warp_points = 5
warp_disp_min = 5
warp_disp_max = 20

im = feature.im_read('face.jpg')
h, w = im.shape[:2]

# Include the corners
points = np.array([[0, 0], [w, 0], [0, h], [w, h]])
points2 = np.array([[0, 0], [w, 0], [0, h], [w, h]])

for i in range(warp_points):
    rand_x = random.uniform(0, w)
    rand_y = random.uniform(0, h)
    p = np.array([rand_x, rand_y])
    rand_len = random.uniform(warp_disp_min, warp_disp_max)
    rand_angle = np.deg2rad(random.uniform(0, 360))

    p2_x = rand_x + np.cos(rand_angle) * rand_len
    p2_y = rand_y + np.sin(rand_angle) * rand_len
    p2 = np.array([p2_x, p2_y])

    if points is None:
        points = np.array([p])
    else:
        temp = np.vstack((points, p))
        points = temp

    if points2 is None:
        points2 = np.array([p2])
    else:
        temp = np.vstack((points2, p2))
        points2 = temp

figure = plt.figure(figsize=(8, 8))
axes = figure.add_subplot(111)
axes.set_xticks(())
axes.set_yticks(())
axes.imshow(im)

plot = axes.scatter(*zip(*points), c="b", s=5)
plot2 = axes.scatter(*zip(*points2), c="orange", s=5)

#H, mask = cv2.findHomography(points2, points, method=cv2.RANSAC, ransacReprojThreshold=5.0)
#im2 = cv2.warpPerspective(im, H, (w, h))
#plt.imshow(im2)

from skimage.transform import PiecewiseAffineTransform, warp, AffineTransform, SimilarityTransform
tform = PiecewiseAffineTransform()
tform.estimate(points, points2)

im2 = warp(im, tform)

plt.imshow(im2)
