# Author: Jose G Perez
# Version 1.0
# Last Modified: January 31, 2018
import numpy as np
import pylab as plt

from util_matching import match
from util_sift import precompute_sift, load_sift

precompute_sift('S_BB_V1', 'PW_BB_V1')
s_im, s_label, s_kp, s_des = load_sift('S_BB_V1_SIFT.npz')
pw_im, pw_label, pw_kp, pw_des = load_sift('PW_BB_V1_SIFT.npz')
# =======******* Row Testing [S33 and all PW]
# PW68 is IDX:39
# matches = []
# s_idx = np.where(s_label == 33)[0][0]
# (im1, kp1, des1) = (s_im[s_idx], s_kp[s_idx], s_des[s_idx])
# for pw_idx in range(pw_im.shape[0]):
#     print(pw_idx, '/', pw_im.shape[0])
#     (im2, kp2, des2) = (pw_im[pw_idx], pw_kp[pw_idx], pw_des[pw_idx])
#     m = match(kp1, des1, kp2, des2)
#     matches.append(m)
#
# count = []
# for match in matches:
#     count.append(len(match))
matches = []
pw_idx = 33
(im2, kp2, des2) = (pw_im[pw_idx], pw_kp[pw_idx], pw_des[pw_idx])
for s_idx in range(s_im.shape[0]):
    print(s_idx, '/', s_im.shape[0])
    (im1, kp1, des1) = (s_im[s_idx], s_kp[s_idx], s_des[s_idx])
    m = match(kp1, des1, kp2, des2)
    matches.append(m)

count = []
for match in matches:
    count.append(len(match))
count_norm = np.array(count) / np.max(count)
count_im = (count_norm * 255).reshape(1, 73)
plt.gray()
plt.imshow(count_im)