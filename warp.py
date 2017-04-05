# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pylab as plt
import random

import feature

im = feature.im_read('nissl/Level-01.jpg')
h, w = im.shape[:2]

points = None
plots = []

rand_x = random.uniform(0, w)
rand_y = random.uniform(0, h)
p = np.array([rand_x, rand_y])

if points is None:
    points = np.array([p])
else:
    temp = np.vstack((points, p))
    points = temp

figure = plt.figure(figsize=(8, 8))
axes = figure.add_subplot(111)
axes.set_xticks(())
axes.set_yticks(())
axes.imshow(im)

plot = axes.scatter(*zip(*points), c="b", s=25)
plots.append(plot)

plt.show()