#%% Load Bregma Data
import numpy as np
import pylab as plt
from skimage import color
import sys

from util_visualization import imshow_matches
from util_sm import load_sm, norm_sm, norm_prob_sm
from util_sift import precompute_sift, load_sift

BREG = True

s_im, s_label, s_kp, s_des = load_sift('S_BB_V2_SIFT.npz')
pw_im, pw_label, pw_kp, pw_des = load_sift('PW_BB_V2_SIFT.npz')
pwi_im, pwi_label, pwi_kp, pwi_des = load_sift('PW_BB_V1_SIFT.npz')
sm_matches, sm_metric = load_sm('sm_v3', s_kp, pw_kp)

def norm_sm(sm, max_value=255, min_value=0):
    sm[sm == np.inf] = sys.maxsize
    im_result = np.zeros_like(sm)
    for idx in range(sm.shape[0]):
        x = sm[idx]
        norm = ((x - np.min(x))*(max_value-min_value)) / (np.max(x)-np.min(x)) + min_value
        im_result[idx] = (norm).reshape(1, sm.shape[1])
    return im_result

np.set_printoptions(edgeitems=5)
b_s4 = np.loadtxt('Bregma_S4.csv', dtype=np.float, delimiter=',')
b_pw3 = np.loadtxt('Bregma_PW3.csv', dtype=np.float, delimiter=',')



breg_1 = b_s4
breg_2 = b_pw3
# M = np.zeros((breg_1.shape[0]+1, breg_2.shape[0]))
M = np.zeros((breg_1.shape[0]+1, breg_2.shape[0]))
M[1:,:] = np.inf
DIR = np.zeros_like(M)
DIR[1:,:] = np.inf



#%% Dynamic programming
for row in range(1, M.shape[0]):
    for col in range(M.shape[1]):
        if row > col:
            continue
        if BREG:
            choices = [M[row,col-1], # Left
                       M[row-1,col-1]+abs(breg_1[row-1] - breg_2[col])] # Diagonal
            M[row][col] = min(choices)
            DIR[row][col] = np.argmin(choices)
        else:
            choices = [M[row,col-1], # Left
                        M[row-1,col-1] + sm_matches[row-1][col-1], # Diagonal
                        M[row-1][col-1] # Right
                       ]
            M[row][col] = max(choices)
            DIR[row][col] = np.argmax(choices)



#%% Overlap
bg = norm_sm(M, 255).astype(np.uint8)
color_mask = np.zeros((DIR.shape[0], DIR.shape[1], 3))
for row in range(1, M.shape[0]):
    for col in range(M.shape[1]):
        if row > col:
            color_mask[row][col] = [255, 0, 0]
d_row = DIR.shape[0] - 1
d_col = DIR.shape[1] - 1
count = 0
path = ['START']
pairs = []
if BREG:
    while d_row != 0 and d_col != 0:
        color_mask[d_row, d_col] = [0, 0, 255]
        bg[d_row, d_col] = 255
        next_dir = DIR[d_row, d_col]
        pairs.append([d_row, d_col])
        if next_dir == 0:
            d_col -= 1
            path.append('L')
        elif next_dir == 1:
            d_row -= 1
            d_col -= 1
            path.append('D')
else:
    while d_row != 0 and d_col != 0:
        color_mask[d_row, d_col] = [0, 0, 255]
        bg[d_row, d_col] = 255
        next_dir = DIR[d_row, d_col]
        pairs.append([d_row, d_col])
        if next_dir == 0:
            d_col -= 1
            path.append('L')
        elif next_dir == 1:
            d_row -= 1
            d_col -= 1
            path.append('D')
        else:
            d_row -= 1
            path.append("U")

#%% Path Figure
print("path", count, path)
img_color = np.stack((bg,) * 3, axis=2)
img_hsv = color.rgb2hsv(img_color)
color_mask_hsv = color.rgb2hsv(color_mask)
img_hsv[..., 0] = color_mask_hsv[..., 0]
img_hsv[..., 1] = color_mask_hsv[..., 1]
im_overlay = color.hsv2rgb(img_hsv)
plt.figure()
plt.title("Bregma Overlay: " + str(BREG))
plt.imshow(im_overlay)
plt.show()