# Author: Jose G Perez
# Version 1.0
# Last Modified: April 9, 2018
import numpy as np
import pylab as plt
import cv2
import os
from timeit import default_timer as timer
from multiprocessing.pool import Pool
from skimage import color
from util_sm import load_sm, norm_sm, norm_prob_sm
from util_sift import precompute_sift, load_sift, array_to_kp, kp_to_array
from bounding import process_plate
from util_matching import match

#%% Load precomputed atlas information
precompute_sift('S_BB_V4', 'PW_BB_V4')
s_im, s_label, s_kp, s_des = load_sift('S_BB_V4_SIFT.npz')
pw_im, pw_label, pw_kp, pw_des = load_sift('PW_BB_V4_SIFT.npz')
sm_matches, sm_metric = load_sm('sm_v4', s_kp, pw_kp)

#%% Figure Set-Ups
pw_ticks_idxs = [0]
pw_ticks_vals = [pw_label[0]]
for x in range(len(pw_label)):
    try:
        diff = pw_label[x + 1] - pw_label[x]
        if diff > 1:
            pw_ticks_idxs.append(x)
            pw_ticks_vals.append(pw_label[x])
            # print("IDX: ", x, "DIFF:", diff)
    except:
        continue

pw_ticks_idxs.append(len(pw_label) - 1)
pw_ticks_vals.append(pw_label[-1])

def figure_reg():
    fig, ax = plt.subplots()
    ax.set_xlabel('PW Level')
    ax.set_ylabel('S Level')
    plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right')
    plt.xticks(pw_ticks_idxs, pw_ticks_vals)
    plt.yticks(np.arange(0, len(s_label)), np.arange(1, len(s_label) + 1))

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(8)

    return fig, ax

def figure_exp():
    fig, ax = plt.subplots()
    ax.set_xlabel('PW Plate')
    ax.set_ylabel('EXP Plate')
    plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right')
    plt.xticks(pw_ticks_idxs, pw_ticks_vals)
    plt.yticks([0, 1, 2, 3], ['s1t6', 's2t3', 's3t3', 's4t2'])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)

    return fig, ax

def figure_synth():
    fig, ax = plt.subplots()
    ax.set_ylabel('PW Plate')
    ax.set_xlabel('EXP Plate')
    plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right')
    plt.yticks(pw_ticks_idxs, pw_ticks_vals)
    plt.xticks([0, 1, 2, 3], ['s1t6', 's2t3', 's3t3', 's4t2'])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)

    return fig, ax

#%% Functions
def visualize_dir(dir):
    color_mask = np.zeros((dir.shape[0], dir.shape[1], 3), dtype=np.uint8)
    color_mask[:, :] = [255, 255, 255]
    for row in range(dir.shape[0]):
        for col in range(dir.shape[1]):
            val = dir[row, col]
            if val == 0: # Col-1
                color = [255, 0, 0]
            elif val == 1: # Diagonal
                color = [0, 0, 255]
            else: # Row-1
                color = [0, 255, 0]

            color_mask[row, col] = color

    return color_mask

def brute_force(sm):
    bg = sm.astype(np.uint8)
    color_mask = np.zeros((sm.shape[0], sm.shape[1], 3))

    for row in range(sm.shape[0]):
        best_col = sm[row].argmax()
        color_mask[row, best_col] = [0, 0, 255]
        bg[row, best_col] = 255

    img_color = np.stack((bg,) * 3, axis=2)
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1]
    im_overlay = color.hsv2rgb(img_hsv)
    return im_overlay

def brute_force_t(sm):
    bg = sm.astype(np.uint8)
    color_mask = np.zeros((sm.shape[0], sm.shape[1], 3))

    for col in range(sm.shape[1]):
        best_row = sm[:, col].argmax()
        color_mask[best_row, col] = [0, 0, 255]
        bg[best_row, col] = 255

    img_color = np.stack((bg,) * 3, axis=2)
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1]
    im_overlay = color.hsv2rgb(img_hsv)
    return im_overlay

def dynamic_prog(sm, col_penalty, row_penalty):
    ed = np.zeros((sm.shape[0]+1, sm.shape[1]+1))
    dir = np.zeros_like(ed)
    ed[:,0] = np.arange(ed.shape[0]) * -row_penalty
    ed[0,:] = np.arange(ed.shape[1]) * -col_penalty
    # ed[:,0] = ed[0,:] = 0

    for i in range(1,ed.shape[0]):
        for j in range(1,ed.shape[1]):
            choices = [ed[i,j-1] - col_penalty,  # 0 = top
                       ed[i-1,j-1] + sm[i-1,j-1],  # 1 = diagonal
                       ed[i-1,j] - row_penalty] # 2 = left
            idx = np.argmax(choices)
            dir[i,j]=idx
            ed[i,j]=choices[idx]
    return ed, dir.astype(np.uint8)

def dynamic_prog_breg(sm, pw_penalty, s_penalty, b_s4, b_pw3):
    ed = np.zeros((sm.shape[0]+1, sm.shape[1]+1))
    dir = np.zeros_like(ed)

    # Adjust bregma ranges so they are all negative
    # b_s4 = b_s4 - b_s4.max()
    # b_pw3 = b_pw3 - b_pw3.max()

    # Append 0th bregma
    # b_s4 = np.insert(b_s4, 0, 0.001)
    # b_pw3 = np.insert(b_pw3, 0, 0.001)
    alpha = 1

    ed[:,0] = np.arange(ed.shape[0]) * -s_penalty
    ed[0,:] = np.arange(ed.shape[1]) * -pw_penalty
    # ed[:,0] = ed[0,:] = 0

    for i in range(1,ed.shape[0]):
        for j in range(1,ed.shape[1]):
            diff = np.abs(b_s4[i-1] - b_pw3[i-1])
            choices = [ed[i,j-1] - pw_penalty, # 0 = top
                       ed[i-1,j-1] + (1 * sm[i-1,j-1]) - (alpha * diff) , # 1 = diagonal
                       ed[i-1,j] - s_penalty] # 2 = left
            idx = np.argmax(choices)
            dir[i,j]=idx
            ed[i,j]=choices[idx]
    return ed, dir.astype(np.uint8)

def bregma_space(dir, sm, b_s4, b_pw3):
    fig, ax = plt.subplots()
    ax.set_xlabel('S4 Bregma')
    ax.set_ylabel('PW3 Bregma')
    sidx = sm.shape[0] - 1
    pwidx = sm.shape[1] - 1
    while sidx >= 0 and pwidx >= 0:
        s_breg = b_s4[sidx]
        pw_breg = b_pw3[pwidx]

        ax.plot(s_breg, pw_breg, marker='o', markerSize=5, color='red')
        next_dir = dir[sidx, pwidx]
        if next_dir == 0:  # Left
            pwidx -= 1
        elif next_dir == 1:  # Diagonal
            sidx -= 1
            pwidx -= 1
        else:  # Up
            sidx -= 1

    return fig, ax

#%% Overlay Function
def overlay(dir, sm):
    color_mask = np.zeros((dir.shape[0],dir.shape[1],3))
    # bg = np.zeros((dir.shape[0],dir.shape[1]))
    # bg[1:,1:] = sm.astype(np.uint8)
    bg = sm.astype(np.uint8)

    sidx = sm.shape[0]-1
    pwidx = sm.shape[1]-1
    count = 0
    path = ['START']
    pairs = []
    while sidx >= 0 and pwidx >= 0:
        count += 1
        color_mask[sidx, pwidx] = [0, 0, 255]
        # bg[sidx-1, pwidx-1] = 255
        bg[sidx, pwidx] = 255
        next_dir = dir[sidx, pwidx]
        pairs.append([sidx, pwidx])
        if next_dir == 0: # Left
            pwidx -= 1
            path.append('L')
        elif next_dir == 1: # Diagonal
            sidx -= 1
            pwidx -= 1
            path.append('D')
        else: # Up
            sidx -= 1
            path.append('U')

    # Remove penalty row/col
    dir = dir[1:,1:]
    color_mask = color_mask[1:,1:,:]

    # PW8 S6, PW11 S11, PW42 S23, PW68 S33,
    if dir.shape[0] == 74:
        color_mask[np.where(s_label == 6), np.where(pw_label == 8)] = [255, 0, 0]
        bg[np.where(s_label == 6), np.where(pw_label == 8)] = 255

        color_mask[np.where(s_label == 11), np.where(pw_label == 11)] = [255, 0, 0]
        bg[np.where(s_label == 11), np.where(pw_label == 11)] = 255

        color_mask[np.where(s_label == 23), np.where(pw_label == 42)] = [255, 0, 0]
        bg[np.where(s_label == 23), np.where(pw_label == 42)] = 255

        color_mask[np.where(s_label == 33), np.where(pw_label == 68)] = [255, 0, 0]
        bg[np.where(s_label == 33), np.where(pw_label == 68)] = 255

    print("path", count, path)
    img_color = np.stack((bg,)*3,axis=2)
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1]
    im_overlay = color.hsv2rgb(img_hsv)
    return im_overlay, np.array(pairs)

#%% Load Bregma data
b_s4 = np.loadtxt('Bregma_S4.csv', dtype=np.float, delimiter=',')
b_pw3 = np.loadtxt('Bregma_PW3_M.csv', dtype=np.float, delimiter=',')

b_pw = []
for i in range(len(pw_label)):
    lbl = pw_label[i]
    idx = lbl-1
    b_pw.append(b_pw3[idx])

b_pw3 = np.array(b_pw)

#%% Brute Force Experiment
im_bf_overlay = brute_force(sm_matches)
fig, ax = figure_reg()
ax.set_title('Brute Force Overlay')
ax.imshow(im_bf_overlay)

#%% Dynamic Programming (Bregma Distances)
ed_b, dir_b = dynamic_prog_breg(sm_matches, pw_penalty=50, s_penalty=50, b_s4=b_s4, b_pw3=b_pw3)
im_dpb_overlay, pairs_b = overlay(dir_b, sm_matches)
im_dir_b = visualize_dir(dir_b)

fig, ax = figure_reg()
ax.set_title('Dynamic Prog (Bregma) - Overlay')
ax.imshow(im_dpb_overlay)

fig, ax = figure_reg()
ax.set_title('Dynamic Prog (Bregma) - Dir [Red=Col-1, Blue=Diagonal, Green=Row-1]')
ax.imshow(im_dir_b)

#%% Dynamic Programming (Regular)
mat = sm_matches
pw_penalty = 50
s_penalty = 50
ed, dir = dynamic_prog(mat, col_penalty=pw_penalty, row_penalty=s_penalty)
im_overlay, pairs = overlay(dir, mat)
im_dir = visualize_dir(dir)

fig, ax = figure_reg()
ax.set_title('Dynamic Prog (Regular) - Overlay')
ax.imshow(im_overlay)

fig, ax = figure_reg()
ax.set_title('Dynamic Prog (Regular) - Dir [Red=Col-1, Blue=Diagonal, Green=Row-1]')
ax.imshow(im_dir)

#%% Bregma Space
fig, ax = bregma_space(dir, mat, b_s4, b_pw3)
ax.set_title('Bregma Space (Regular DP)')

fig, ax = bregma_space(dir_b, sm_matches, b_s4, b_pw3)
ax.set_title('Bregma Space (Bregma Dist DP)')

#%% Experimental data init
e1, e1_c = process_plate('experimental/18-016 LHA s1t6.tif', split=True)
e2, e2_c = process_plate('experimental/18-016 LHA s2t3.tif', split=True)
e3, e3_c = process_plate('experimental/18-016 LHA s3t3.tif', split=True)
e4, e4_c = process_plate('experimental/18-016 LHA s4t2.tif', split=True)

exp = [e1_c, e2_c, e3_c, e4_c]
exp_kp = []
exp_des = []

if not os.path.isfile('EXP_SIFT.npz'):
    print("Precomputing and saving SIFT for experimental")
    SIFT = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.05, edgeThreshold=100, sigma=2)
    for im_exp in exp:
        kp, des = SIFT.detectAndCompute(im_exp, None)
        kp = kp_to_array(kp)

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

#%% Prepare to match with PW
def perform_match(idx):
    global s_kp, s_des, pw_kp, pw_des, exp_kp, exp_des
    matches = []
    print('EXP IDX', idx)
    for pw_idx in range(pw_kp.shape[0]):
        matches.append(match(exp_kp[idx], exp_des[idx], pw_kp[pw_idx], pw_des[pw_idx]))

    return np.array(matches)
    # np.savez_compressed(str(idx) + '-EM', m=matches)

#%% Match (Sequential)
if not os.path.isfile('EXP_SM.npz'):
    print("Creating EXP SM")
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
    print("Loading pre-saved SM for experimental")
    data = np.load('EXP_SM.npz')
    sm_exp = data['sm']

sm_exp_count = np.zeros_like(sm_exp)
for row in range(sm_exp.shape[0]):
    count = []
    for col in range(sm_exp.shape[1]):
        matches = sm_exp[row,col]
        count.append(len(matches))

    sm_exp_count[row] = np.array(count)

#%% Experimental matching with brute-force approach
bf_exp_overlay = brute_force(sm_exp_count)
fig, ax = plt.exp()
ax.set_title('Brute-Force Matching (SM-Exp)')
ax.imshow(bf_exp_overlay)

#%% Experimental matching with dynamic programming
ed_e, dir_e = dynamic_prog(sm_exp_count, col_penalty=10, row_penalty=10)
im_overlay_e, pairs_e = overlay(dir_e, sm_exp_count)
im_dir_e = visualize_dir(dir_e)

fig, ax = figure_exp()
ax.imshow(sm_exp_count.astype(np.uint8))
ax.set_title('Similarity Matrix - Experimental')

fig, ax = figure_exp()
ax.set_title('Dynamic Prog (Experimental) - Overlay')
ax.imshow(im_overlay_e)

fig, ax = figure_exp()
ax.set_title('Dynamic Prog (Experimental) - Dir [Red=Col-1, Blue=Diagonal, Green=Row-1]')
ax.imshow(im_dir_e)

#%% Synthetic
sm_synth = sm_exp_count
sm_synth[1,1] = 20

fig, ax = figure_synth()
ax.imshow(sm_synth.astype(np.uint8))
ax.set_title('Similarity Matrix - Synthetic')

#%% Synthetic Brute Force
bf_exp_overlay = brute_force(sm_synth)
fig, ax = figure_synth()
ax.set_title('Brute-Force Matching (SM-Exp)')
ax.imshow(bf_exp_overlay)

#%% Synthetic Dynamic Programming
ed_s, dir_s = dynamic_prog(sm_synth, col_penalty=10, row_penalty=10)
im_overlay_s, pairs_s = overlay(dir_s, sm_synth)
im_dir_s = visualize_dir(dir_s)

fig, ax = figure_synth()
ax.set_title('Dynamic Prog (Synthetic) - Overlay')
ax.imshow(im_overlay_s)

fig, ax = figure_synth()
ax.set_title('Dynamic Prog (Synthetic) - Dir [Red=Col-1, Blue=Diagonal, Green=Row-1]')
ax.imshow(im_dir_s)


#%% Match (Threaded)
# TODO: Required to be in main function
# time_start = timer()
#
# pool = Pool()
# idx = range(im_exp.shape[0])
#
# print('Beginning pool work: Matching experimental with atlas')
# pool.map(perform_match, idx)
# pool.close()
# pool.join()
#
# duration = timer() - time_start
# duration_m = duration / 60
# print("Program took %.3fs %.3fm" % (duration, duration_m))