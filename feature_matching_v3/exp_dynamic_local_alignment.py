#%% Experiment: Dynamic programming to align a sequence of plates to an atlas
# Author: Jose G. Perez
# Last Modified: March 28, 2020
import numpy as np
import pylab as plt
import cv2
import os
from timeit import default_timer as timer

import util_sm
import util_sift
import util_matching
import util_visualization
import bounding

PATH_CHARACTERS = ['', '↖', '↑', '←']

#%% Load atlas data
util_sift.precompute_sift('S_BB_V4', 'PW_BB_V4')
s_im, s_label, s_kp, s_des = util_sift.load_sift('S_BB_V4_SIFT.npz')
pw_im, pw_label, pw_kp, pw_des = util_sift.load_sift('PW_BB_V4_SIFT.npz')
sm_matches, sm_metric = util_sm.load_sm('sm_v4', s_kp, pw_kp)

#%% Load local sequence data
e1, e1_c = bounding.process_plate('experimental/18-016 LHA s1t6.tif', split=True)
e2, e2_c = bounding.process_plate('experimental/18-016 LHA s2t3.tif', split=True)
e3, e3_c = bounding.process_plate('experimental/18-016 LHA s3t3.tif', split=True)
e4, e4_c = bounding.process_plate('experimental/18-016 LHA s4t2.tif', split=True)

exp_filenames = ['s1t6', 's2t3', 's3t3', 's4t2']
exp = [e1_c, e2_c, e3_c, e4_c]
exp_kp = []
exp_des = []

if not os.path.isfile('EXP_SIFT.npz'):
    print("Precomputing and saving SIFT for experimental dataset")
    SIFT = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.05, edgeThreshold=100, sigma=2)
    for im_exp in exp:
        kp, des = SIFT.detectAndCompute(im_exp, None)
        kp = util_sift.kp_to_array(kp)

        exp_kp.append(kp)
        exp_des.append(des)

    exp_kp = np.asarray(exp_kp)
    exp_des = np.asarray(exp_des)

    np.savez_compressed('EXP_SIFT', kp=exp_kp, des=exp_des)
else:
    print("Loading pre-saved SIFT for experimental")
    data = np.load('EXP_SIFT.npz')
    exp_kp = data['kp']
    exp_des = data['des']

#%% Compute similarity matrix between atlas (PW) and experimental sequence
def perform_match(idx):
    global s_kp, s_des, pw_kp, pw_des, exp_kp, exp_des
    matches = []
    print('EXP IDX', idx)
    for pw_idx in range(pw_kp.shape[0]):
        matches.append(util_matching.match(exp_kp[idx], exp_des[idx], pw_kp[pw_idx], pw_des[pw_idx]))

    return np.array(matches)


if not os.path.isfile('EXP_SM.npz'):
    print("Creating atlas-experimental similarity matrix")
    time_start = timer()
    sm_exp = []
    for i in range(len(exp)):
        sm_exp.append(perform_match(i))

    sm_exp = np.array(sm_exp)
    duration = timer() - time_start
    duration_m = duration / 60
    print("Matching took %.3fs %.3fm" % (duration, duration_m))
    np.savez_compressed('EXP_SM', sm=sm_exp)
else:
    print("Loading pre-saved SM for atlas-experimental")
    data = np.load('EXP_SM.npz')
    sm_exp = data['sm']

sm_atlas_experimental = np.zeros_like(sm_exp, dtype=np.uint8)
for row in range(sm_exp.shape[0]):
    count = []
    for col in range(sm_exp.shape[1]):
        matches = sm_exp[row,col]
        count.append(len(matches))

    sm_atlas_experimental[row] = np.array(count)


#%% Dynamic programming algorithm (Sequence to Atlas / Local Alignment)
def local_alignment(sm, col_penalty, row_penalty):
    V = np.zeros((sm.shape[0]+1, sm.shape[1]+1))
    paths = np.zeros_like(V)
    V[:, 0] = V[0, :] = 0

    for i in range(1, V.shape[0]):
        for j in range(1, V.shape[1]):
            # 0 = Finish
            # 1 = Diagonal / Match
            # 2 = Up / Deletion
            # 3 = Left / Insertion
            choices = [0,
                       V[i-1, j-1] + sm[i-1, j-1],
                       V[i-1, j] - row_penalty,
                       V[i, j-1] - col_penalty,
                       ]

            idx = np.argmax(choices)
            paths[i,j] = idx
            V[i,j] = choices[idx]

    return V, paths.astype(np.uint8)


def reconstruct(sm, V, paths):
    # Visualization
    foreground = np.zeros((V.shape[0], V.shape[1], 3))
    background = V.astype(np.uint8)

    # For local alignment, pick the highest number in V
    max_value = np.amax(V)
    idxs = np.where(V == max_value)
    start_row_idx = idxs[0][0]
    start_col_idx = idxs[1][0]

    # Then backtrack until we hit a 0
    # We also keep backtracking until we reach the top-left corner
    row_idx = start_row_idx
    col_idx = start_col_idx
    score = V[row_idx, col_idx]

    while row_idx > 0 or col_idx > 0:
        # Paint the pixel we traveled to in the visualization
        value = paths[row_idx, col_idx]
        foreground[row_idx, col_idx] = [0, 0, 255]
        background[row_idx, col_idx] = 255

        # Navigate the dynamic programming table
        diagonal = (value == 1) or (value == 0 and col_idx >= 1 and row_idx >= 1)
        up = (value == 2) or (value == 0 and row_idx >= 1)
        left = (value == 3) or (value == 0 and col_idx >= 1)
        if diagonal:
            row_idx -= 1
            col_idx -= 1
        elif up:
            row_idx -= 1
        elif left:
            col_idx -= 1

    return util_visualization.overlay(background, foreground)


#%% Actual Experiment!
V, paths = local_alignment(sm_atlas_experimental, 1, 1)
im_overlay = reconstruct(sm_atlas_experimental, V, paths)

#%% Figure 1: Atlas-Experimental Similarity Matrix
util_visualization.imshow_detailed(
                np_arr=sm_atlas_experimental,
                title='Atlas-Experimental Similarity Matrix',
                axis_xlabel='PW Plate',
                axis_ylabel='Experimental Plate',
                xlabel_arr=['ε'] + pw_label,
                ylabel_arr=['ε'] + exp_filenames)

#%% Figure 2: DP Value Matrix (V)
util_visualization.imshow_detailed(
                np_arr=V,
                title='Atlas-Experimental Dynamic Programming Value Matrix',
                axis_xlabel='PW Plate',
                axis_ylabel='Experimental Plate',
                xlabel_arr=['ε'] + pw_label,
                ylabel_arr=['ε'] + exp_filenames)

#%% Figure 3: DP Paths Matrix (With arrows!)
util_visualization.imshow_detailed(
                np_arr=paths,
                title='Atlas-Experimental Dynamic Programming Path',
                axis_xlabel='PW Plate',
                axis_ylabel='Experimental Plate',
                xlabel_arr=['ε'] + pw_label,
                ylabel_arr=['ε'] + exp_filenames,
                value_to_str_func=lambda i, j, val: PATH_CHARACTERS[val])

#%% Figure 4: DP Path Backtracking Overlay
util_visualization.imshow_detailed(
                np_arr=im_overlay,
                title='Atlas-Experimental Dynamic Programming Overlay',
                axis_xlabel='PW Plate',
                axis_ylabel='Experimental Plate',
                xlabel_arr=['ε'] + pw_label,
                ylabel_arr=['ε'] + exp_filenames,
                value_to_str_func=lambda i, j, val: int(V[i, j]))