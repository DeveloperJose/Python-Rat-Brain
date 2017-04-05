# -*- coding: utf-8 -*-
from timeit import default_timer as timer
from PIL import Image
from scipy.ndimage import filters
import numpy as np
import pylab as plt

import config
import feature

im_nissl = feature.nissl_load(34)
plt.imshow(im_nissl)
corners = np.asarray(plt.ginput(n=4), dtype=np.float32)

im = np.array(im_nissl)
for corner in corners:
    x, y = [], []
    # Numpy slicing uses integers
    x.append(corner[0].astype(np.uint64))
    y.append(corner[1].astype(np.uint64))

    x1, x2, y1, y2 = min(x), max(x), min(y), max(y)
    im_region = im[y1:y2, x1:x2].copy()

feature.im_write(str(34) + '-part.jpg', im_region)