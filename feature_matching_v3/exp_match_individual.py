# Author: Jose G Perez
# Version 1.0
# Last Modified: January 31, 2018
import numpy as np
import pylab as plt
import cv2

from util_sift import array_to_kp, precompute_sift, load_sift
from util_matching import match
from util_cv import match_to_cv
from util_visualization import imshow

precompute_sift('S_BB_V4', 'PW_BB_V4')
s_im, s_label, s_kp, s_des = load_sift('S_BB_V4_SIFT.npz')
pw_im, pw_label, pw_kp, pw_des = load_sift('PW_BB_V4_SIFT.npz')

#%%
s_idx = 32 # np.where(s_label == 33)[0][0]
pw_idx = 36 # np.where(pw_label == 50)[0][0]
(im1, kp1, des1) = (s_im[s_idx], s_kp[s_idx], s_des[s_idx])
(im2, kp2, des2) = (pw_im[pw_idx], pw_kp[pw_idx], pw_des[pw_idx])
matches = match(kp1,des1,kp2,des2)
# Convert to OpenCV objects for viewing
matches = match_to_cv(matches)
kp1 = array_to_kp(kp1)
kp2 = array_to_kp(kp2)
im_matches = cv2.drawMatches(im1, kp1, im2, kp2, matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# plt.gray()
str = '%s Matches Between S%s and PW%s' % (len(matches), s_label[s_idx], pw_label[pw_idx])
imshow(im_matches, str)