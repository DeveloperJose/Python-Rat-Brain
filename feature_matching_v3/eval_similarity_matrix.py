from util_sift import load_sift, array_to_kp, kp_to_array
from util_matching import match
from util_sm import load_sm
from util_cv import match_to_cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def display(s_idx, pw_idx):
    plt.figure()
    (im1, kp1, des1) = (s_im[s_idx], s_kp[s_idx], s_des[s_idx])
    (im2, kp2, des2) = (pw_im[pw_idx], pw_kp[pw_idx], pw_des[pw_idx])
    matches = match(kp1, des1, kp2, des2)
    # Convert to OpenCV objects for viewing
    matches = match_to_cv(matches)
    kp1 = array_to_kp(kp1)
    kp2 = array_to_kp(kp2)
    s_plate = s_label[s_idx]
    pw_plate = pw_label[pw_idx]
    im_matches = cv2.drawMatches(im1, kp1, im2, kp2, matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    s = '[CV] Matches between S%s & PW%s = %s' % (s_plate, pw_plate, str(len(matches)))
    print(s)
    plt.title(s)
    plt.imshow(im_matches)

#%% Load data
s_im, s_label, s_kp, s_des = load_sift('S_BB_V2_SIFT.npz')
pw_im, pw_label, pw_kp, pw_des = load_sift('PW_BB_V2_SIFT.npz')
pwi_im, pwi_label, pwi_kp, pwi_des = load_sift('PW_BB_V1_SIFT.npz')
sm_matches, sm_metric = load_sm('sm_v3', s_kp, pw_kp)

#%% Only use Nissl PWs=[''=/=
idxs = pw_label[pwi_label-1]-1
pw_im = pw_im[idxs]
pw_label = pw_label[idxs]
pw_kp = pw_kp[idxs]
pw_des = pw_des[idxs]
sm_matches = sm_matches[:,idxs]
sm_metric = sm_metric[:,idxs]

sm_matches = sm_matches.astype(np.uint8)

#%% Figure
fig, ax = plt.subplots()
ax.set_title('Similarity Matrix Visualization (v2)')
ax.imshow(sm_matches, cmap=plt.get_cmap('hot'))

fig2, ax2 = plt.subplots()
while True:
    plt.figure(1)
    p = plt.ginput(n=1, timeout=0)

    if len(p) == 0:
        break

    p = p[0]
    (x, y) = (p[0], p[1])
    rect = patches.Rectangle((np.floor(x)+0.5, np.floor(y)-0.5), 0.5, 0.5, color='blue')
    ax.add_patch(rect)

    s_idx = int(y)
    pw_idx = int(x)
    s_plate = s_label[s_idx]
    pw_plate = pw_label[pw_idx]

    s = 'Matches between S%s & PW%s = %s' % (s_plate, pw_plate, sm_matches[s_idx,pw_idx])
    ax.set_title(s)
    print(s, 'index', s_idx, pw_idx)

    plt.figure(2)
    (im1, kp1, des1) = (s_im[s_idx], s_kp[s_idx], s_des[s_idx])
    (im2, kp2, des2) = (pw_im[pw_idx], pw_kp[pw_idx], pw_des[pw_idx])
    matches = match(kp1, des1, kp2, des2)
    # Convert to OpenCV objects for viewing
    matches = match_to_cv(matches)
    kp1 = array_to_kp(kp1)
    kp2 = array_to_kp(kp2)
    im_matches = cv2.drawMatches(im1, kp1, im2, kp2, matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    s = '[CV] Matches between S%s & PW%s = %s' % (s_plate, pw_plate, str(len(matches)))
    print(s)
    ax2.set_title(s)
    ax2.imshow(im_matches)

plt.title("Finished")