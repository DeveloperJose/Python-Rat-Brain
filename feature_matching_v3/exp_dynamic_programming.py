# Author: Jose G Perez
# Version 1.0
# Last Modified: January 31, 2018
import numpy as np
from util_im import imshow_matches
from util_sm import load_sm, norm_sm
from util_sift import precompute_sift, load_sift

def dynamic_prog(sm, pw_penalty, s_penalty):
    ed = np.zeros((73, 89))
    ed[0:,:] = sm[0:,:] + pw_penalty
    ed[:,0] = sm[:,0] + s_penalty
    for i in range(1,73):
        for j in range(1,89):
            p1 = ed[i, j-1] + pw_penalty
            p2 = ed[i-1,j-1] + sm[i,j]
            p3 = ed[i-1][j] + s_penalty
            ed[i,j]=min(p1,p2,p3)
    return ed

def overlay(ed_matrix, sm):
    from skimage import color
    norm = norm_sm(sm).astype(np.uint8)
    color_mask = np.zeros((73,89,3))
    for sidx in range(73):
        pw_row = ed_matrix[sidx]
        best_pw = np.argmax(pw_row)
        color_mask[sidx, best_pw] = [0, 0, 255]
        norm[sidx, best_pw] = 225

    img_color = np.stack((norm,)*3,axis=2)
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1]
    im_overlay = color.hsv2rgb(img_hsv)
    return im_overlay

if __name__ == '__main__':
    precompute_sift('S_BB_V1', 'PW_BB_V1')
    s_im, s_label, s_kp, s_des = load_sift('S_BB_V1_SIFT.npz')
    pw_im, pw_label, pw_kp, pw_des = load_sift('PW_BB_V1_SIFT.npz')
    sm_matches, sm_metric = load_sm('sm_v2', s_kp, pw_kp)
    pw_penalty = 100
    s_penalty = 200
    ed = dynamic_prog(norm_sm(sm_metric, 100), pw_penalty, s_penalty)
    aoi = ed[32:35, 38:41]
    best_pw = s_label[np.argmax(ed,axis=0)]
    best_s = pw_label[np.argmax(ed,axis=1)]
    print("PW68 best match", best_pw[np.where(pw_label==68)])
    print("S33 best match", best_s[np.where(s_label==33)])
    im_overlay = overlay(ed, sm_metric)
    imshow_matches(im_overlay, 'SM V2 Metric Dyn')