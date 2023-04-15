#%% Load Data
import numpy as np
import pylab as plt
import sys
from util_visualization import imshow_matches
from util_sm import load_sm, norm_sm, norm_prob_sm
from util_sift import precompute_sift, load_sift
from skimage import color

b_s4 = np.loadtxt('Bregma_S4.csv', dtype=np.float, delimiter=',')
b_pw3 = np.loadtxt('Bregma_PW3_M.csv', dtype=np.float, delimiter=',')

s_im, s_label, s_kp, s_des = load_sift('S_BB_V2_SIFT.npz')
pw_im, pw_label, pw_kp, pw_des = load_sift('PW_BB_V2_SIFT.npz')
pwi_im, pwi_label, pwi_kp, pwi_des = load_sift('PW_BB_V1_SIFT.npz')
sm_matches, sm_metric = load_sm('sm_v3', s_kp, pw_kp)

# Only Nissl but using V2 dataset
b_pw3_im = []
b_pw3_label = []
b_pw3_kp = []
b_pw3_des = []
nop = [12, 14, 56, 76, 15, 17, 19, 21, 22, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59,
       61, 63, 65, 67, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91,93,95,97,99,101,103,105,107,109,111,113,115,117,119,
       121,123,125,127,129,131,133,135,137,139,141,143,144,145,147,149,151,153,155]
for plate in pw_label:
    if plate in nop:
        continue
    idx = plate - 1
    b_pw3_im.append(pw_im[idx])
    b_pw3_label.append(pw_label[idx])
    b_pw3_kp.append(pw_kp[idx])
    b_pw3_des.append(pw_des[idx])

pw_im = np.array(b_pw3_im)
pw_label = np.array(b_pw3_label)
pw_kp = np.array(b_pw3_kp)
pw_des = np.array(b_pw3_des)

b_sm_matches = []
b_sm_metric = []
for plate in pw_label:
    idx = plate - 1
    b_sm_matches.append(sm_matches[:,idx])
    b_sm_metric.append(sm_metric[:,idx])

sm_matches = np.array(b_sm_matches)
sm_metric = np.array(b_sm_metric)
# norm = norm_sm(sm_metric, 100)
norm = norm_prob_sm(sm_matches)

#%% Bregma Algorithm
atlas1 = b_s4
atlas2 = b_pw3
M = np.zeros((atlas1.shape[0]+1, atlas2.shape[0]))
M[1:,:] = np.inf
DIR = np.zeros_like(M)
DIR[:,:] = 2
for row in range(1, M.shape[0]):
    for col in range(M.shape[1]):
        if row > col:
            continue

        choices = [M[row,col-1], # Left
                   M[row-1,col-1]+abs(atlas1[row-1] - atlas2[col])] # Diagonal
        M[row][col] = min(choices)
        DIR[row][col] = np.argmin(choices)

M = M[1:,:]
DIR = DIR[1:,:]
#%% Path Backtracking
im = np.zeros((DIR.shape[0], DIR.shape[1], 3), dtype=np.uint8)
for row in range(DIR.shape[0]):
    for col in range(DIR.shape[1]):
        if row > col:
            im[row, col] = [0, 100, 50] # Dark Green
        elif DIR[row][col] == 0: # Left
            im[row,col] = [200, 0, 0] # Red
        elif DIR[row][col] == 1: # Diagonal
            im[row, col] = [0, 255, 0] # Green
        elif DIR[row][col] == 2: # Unused
            im[row, col] = [0, 100, 50] # Dark Green
        else:
            im[row, col] = [0, 0, 0] # Black

c = [148, 0, 211] # Purple
im[6-1][8-1] = c
im[11-1][11-1] = c
im[23-1][42-1] = c
im[33-1][68-1] = c
for row in range(DIR.shape[0]):
    col = np.argmin(M[row])
    # PW8 S6, PW11 S11, PW42 S23, PW68 S33,
    if (row == 6-1 and col == 8-1) or (row == 11-1 and col == 11-1) or (row == 23-1 and col == 42-1) or (row == 33-1 and col == 68-1):
        im[row][col] = [255, 255, 255] # White
    else:
        im[row][col] = [0, 0, 255] # Blue

fig, ax = plt.subplots()
ax.set_title("DIR - Best Path")
ax.imshow(im)