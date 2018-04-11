# Author: Jose G Perez
# Version 1.0
# Last Modified: January 31, 2018
import numpy as np
from util_im import imshow_matches
from util_sm import load_sm, norm_sm, norm_prob_sm
from util_sift import precompute_sift, load_sift

precompute_sift('S_BB_V1', 'PW_BB_V1')
s_im, s_label, s_kp, s_des = load_sift('S_BB_V1_SIFT.npz')
pw_im, pw_label, pw_kp, pw_des = load_sift('PW_BB_V1_SIFT.npz')

def idx_to_plate(labels, plate):
    return np.where(labels == plate)

def dynamic_prog(sm, pw_penalty, s_penalty):
    ed = np.zeros((73+1, 89+1))
    dir = np.zeros_like(ed)
    ed[:,0] = np.arange(ed.shape[0]) * s_penalty
    ed[0,:] = np.arange(ed.shape[1]) * pw_penalty

    for i in range(1,ed.shape[0]):
        for j in range(1,ed.shape[1]):
            choices = [ed[i, j-1] - pw_penalty, # 0 = top
                       ed[i-1,j-1] + sm[i-1,j-1], # 1 = diagonal
                       ed[i-1][j] - s_penalty] # 2 = left
            idx = np.argmax(choices)
            dir[i,j]=idx
            ed[i,j]=choices[idx]
    return ed, dir

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

def error(best_pw, pw_plate, s_plate):
    # s_idx = int(np.argwhere(s_label == s_plate))
    pw_idx = int(np.argwhere(pw_label == pw_plate))
    pred_sidx = best_pw[pw_idx]
    pred_s = int(np.argwhere(s_label == pred_sidx))

    return abs(pred_s - s_plate)

if __name__ == '__main__':
    # best_pwp = 0
    # best_sps = 0
    # best_total = np.inf
    # for pw_penalty in range(0, 200):
    #     for s_penalty in range(0, 200):
    #         ed, ed2 = dynamic_prog(norm, pw_penalty=pw_penalty, s_penalty=s_penalty)
    #         best_pw = s_label[np.argmin(ed, axis=0)]
    #
    #         # PW8 S6, PW11 S11, PW42 S23, PW68 S33,
    #         e = error(best_pw, 68, 33) + \
    #             error(best_pw, 11, 11) + \
    #             error(best_pw, 42, 23) + \
    #             error(best_pw, 68, 33)
    #
    #         if e < best_total:
    #             print("New best total", e)
    #             best_total = e
    #             best_pwp = pw_penalty
    #             best_sps = s_penalty

    # best_pwp = 200
    # best_sps = 200
    # ed, ed2 = dynamic_prog(norm, pw_penalty=best_pwp, s_penalty=best_sps)
    # im_overlay = overlay(ed, norm)

    sm_matches, sm_metric = load_sm('sm_v2', s_kp, pw_kp)
    # norm = norm_sm(sm_metric, 100)
    norm = norm_prob_sm(sm_metric)

    import pylab as plt
    fig, axes = plt.subplots(nrows=2, ncols=2)
    plt.subplots_adjust(left=0.25, bottom=0.25)
    plt.set_cmap(plt.get_cmap('hot'))
    # axes.set_title('Dynamic')

    axes[0,0].set_title('Similarity Matrix [Metric]')
    axes[0,0].imshow(sm_metric)

    axes[0,1].set_title('SM Norm')
    axes[0,1].imshow(norm)

    axes[1,0].set_title('ED')
    axes[1,1].set_title('Overlay')

    # Sliders
    axcolor = 'lightgoldenrodyellow'
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    s_pwp = plt.Slider(axfreq, 'PW Penalty', 0, 2, valinit=np.mean(norm))
    s_sp = plt.Slider(axamp, 'S Penalty', 0, 2, valinit=np.mean(norm))

    def update(val):
        pw_penalty = s_pwp.val
        s_penalty = s_sp.val

        ed, ed2 = dynamic_prog(norm, pw_penalty=pw_penalty, s_penalty=s_penalty)
        im_overlay = overlay(ed, norm)

        axes[1,0].imshow(ed)
        axes[1,1].imshow(im_overlay)
        fig.canvas.draw_idle()

    s_pwp.on_changed(update)
    s_sp.on_changed(update)

    # fig.tight_layout()

    # imshow_matches(dynamic_prog(norm, pw_penalty=1, s_penalty=1)[1], '')
    # imshow_matches(overlay(dynamic_prog(sm_matches, 0.9, 0.1)[0], sm_matches), '')

    # aoi = ed[32:35, 38:41]
    # best_s = pw_label[np.argmin(ed,axis=1)]
    # print("PW68 best match", best_pw[np.where(pw_label==68)])
    # print("S33 best match", best_s[np.where(s_label==33)])

