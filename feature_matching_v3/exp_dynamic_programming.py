# Author: Jose G Perez
# Version 1.0
# Last Modified: January 31, 2018
import numpy as np
import pylab as plt
from skimage import color
from util_im import imshow_matches
from util_sm import load_sm, norm_sm, norm_prob_sm
from util_sift import precompute_sift, load_sift

precompute_sift('S_BB_V4', 'PW_BB_V4')
s_im, s_label, s_kp, s_des = load_sift('S_BB_V4_SIFT.npz')
pw_im, pw_label, pw_kp, pw_des = load_sift('PW_BB_V4_SIFT.npz')
sm_matches, sm_metric = load_sm('sm_v4', s_kp, pw_kp)

def idx_to_plate(labels, plate):
    return np.where(labels == plate)

def dynamic_prog(sm, pw_penalty, s_penalty):
    ed = np.zeros((sm.shape[0]+1, sm.shape[1]+1))
    dir = np.zeros_like(ed)
    ed[:,0] = np.arange(ed.shape[0]) * -s_penalty
    ed[0,:] = np.arange(ed.shape[1]) * -pw_penalty
    # ed[:,0] = ed[0,:] = 0

    for i in range(1,ed.shape[0]):
        for j in range(1,ed.shape[1]):
            choices = [ed[i,j-1] - pw_penalty, # 0 = top
                       ed[i-1,j-1] + sm[i-1,j-1], # 1 = diagonal
                       ed[i-1,j] - s_penalty] # 2 = left
            idx = np.argmax(choices)
            dir[i,j]=idx
            ed[i,j]=choices[idx]
    return ed, dir.astype(np.uint8)
    # return ed, dir.astype(np.uint8)

def get_pairs(dir):
    sidx = dir.shape[0]-1
    pwidx = dir.shape[1]-1
    pairs = []
    while sidx > 0 and pwidx > 0:
        next_dir = dir[sidx, pwidx]
        pairs.append([sidx, pwidx])
        if next_dir == 0:
            sidx -= 1
        elif next_dir == 1:
            sidx -= 1
            pwidx -= 1
        else:
            pwidx -= 1
    return np.array(pairs)

def pair_metric(sm_metric, pairs):
    best6_pw = get_best_pw(sm_metric, pairs, 6)
    best11_pw = get_best_pw(sm_metric, pairs, 11)
    best23_pw = get_best_pw(sm_metric, pairs, 23)
    best33_pw = get_best_pw(sm_metric, pairs, 33)

    # PW8 S6, PW11 S11, PW42 S23, PW68 S33,
    # m += np.count_nonzero(best6_pw == np.where(pw_label == 8))
    # m += np.count_nonzero(best11_pw == np.where(pw_label == 11))
    # m += np.count_nonzero(best23_pw == np.where(pw_label == 42))
    # m += np.count_nonzero(best33_pw == np.where(pw_label == 68))
    return np.min(abs(best6_pw - np.where(pw_label == 8))) + \
         np.min(abs(best11_pw - np.where(pw_label == 11))) + \
         np.min(abs(best23_pw - np.where(pw_label == 42))) + \
         np.min(abs(best33_pw - np.where(pw_label == 68)))

def overlay(dir, sm):
    # bg = norm_sm(sm, 255).astype(np.uint8)
    bg = sm.astype(np.uint8)
    color_mask = np.zeros((dir.shape[0],dir.shape[1],3))

    sidx = sm.shape[0]-1
    pwidx = sm.shape[1]-1
    count = 0
    path = ['START']
    pairs = []
    while sidx >= 0 and pwidx >= 0:
        count += 1
        color_mask[sidx, pwidx] = [0, 0, 255]
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

def error(best_pw, pw_plate, s_plate):
    # s_idx = int(np.argwhere(s_label == s_plate))
    pw_idx = int(np.argwhere(pw_label == pw_plate))
    pred_sidx = best_pw[pw_idx]
    pred_s = int(np.argwhere(s_label == pred_sidx))

    return abs(pred_s - s_plate)

def get_best_pw(sm_metric, pairs, s_plate):
    # Indices start at 0, plates start at 1
    sidx = s_plate-1

    pidx = np.where(pairs[:, 0] == sidx)
    matches = pairs[pidx, 1].flatten()
    # return pw_label[matches] if len(matches >= 1) else -1
    return pw_label[matches] if len(matches >= 1) else np.array([np.inf])
    # if len(matches) > 1:
    #     metrics = sm_metric[sidx,matches]
    #     best_idx = np.argmax(metrics)
    #     return int(pw_label[matches[best_idx]])
    # elif len(matches) == 1:
    #     # Convert from PW Indices to PW Labels
    #     return int(pw_label[matches])
    # else:
    #     return -1

if __name__ == '__main__':
    # lowest_error = np.inf
    # best_pw = -1
    # best_s = -1
    # for pw_penalty in np.arange(0.4, 0.5, 0.001):
    #     for s_penalty in np.arange(0.4, 0.5, 0.001):
    #         ed, dir = dynamic_prog(norm, pw_penalty=pw_penalty, s_penalty=s_penalty)
    #         pairs = get_pairs(dir)
    #         metric = pair_metric(sm_metric, pairs)
    #         if metric < lowest_error:
    #             print("New error", metric, pw_penalty, s_penalty)
    #             lowest_error = metric
    #             best_pw = pw_penalty
    #             best_s = s_penalty
    # ed, dir = dynamic_prog(norm, pw_penalty=best_pw, s_penalty=best_s)
    # im_overlay, pairs = overlay(dir, sm_metric)
    # best6_pw = get_best_pw(sm_metric,pairs,6)
    # best11_pw = get_best_pw(sm_metric,pairs,11)
    # best23_pw = get_best_pw(sm_metric,pairs,23)
    # best33_pw = get_best_pw(sm_metric,pairs,33)
    # print("[PW8=%s], [PW11=%s], [PW42=%s [PW68=%s]" % (best6_pw, best11_pw, best23_pw, best33_pw))
    #
    # imshow_matches(im_overlay, 'Dynamic Programming')

    # import pylab as plt
    # best_pw = 200
    # best_s = 220
    # ed, dir = dynamic_prog(norm, pw_penalty=best_pw, s_penalty=best_s)
    # pairs = get_pairs(dir)
    # metric = pair_metric(sm_metric, pairs)
    # im_overlay, pairs = overlay(dir, sm_metric)
    # best6_pw = get_best_pw(sm_metric,pairs,6)
    # best11_pw = get_best_pw(sm_metric,pairs,11)
    # best23_pw = get_best_pw(sm_metric,pairs,23)
    # best33_pw = get_best_pw(sm_metric,pairs,33)
    # print("[PW8=%s], [PW11=%s], [PW42=%s [PW68=%s]" % (best6_pw, best11_pw, best23_pw, best33_pw))
    #
    # imshow_matches(im_overlay, 'Dynamic Programming')
    # plt.show()

    # mat = sm_matches
    #
    # pw_penalty = 50
    # s_penalty = 50
    # ed, dir = dynamic_prog(mat, pw_penalty=pw_penalty, s_penalty=s_penalty)
    # im_overlay, pairs = overlay(dir, mat)
    # norm = norm_sm(mat)
    #
    # import pylab as plt
    # fig, axes = plt.subplots(nrows=2, ncols=2)
    # plt.subplots_adjust(left=0.25, bottom=0.25)
    # plt.set_cmap(plt.get_cmap('hot'))
    # # axes.set_title('Dynamic')
    #
    # axes[0,0].set_title('Similarity Matrix')
    # axes[0,0].imshow(mat)
    #
    # axes[0,1].set_title('SM Norm')
    # axes[0,1].imshow(norm_prob_sm(sm_matches))
    #
    # axes[1,0].set_title('ED')
    # axes[1,1].set_title('Overlay')
    #
    # # Sliders
    # axcolor = 'lightgoldenrodyellow'
    # axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    # axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    # # s_pwp = plt.Slider(axfreq, 'PW Penalty', 0, 1, .0001, valfmt='%.8f')
    # # s_sp = plt.Slider(axamp, 'S Penalty', 0, 1, .0001, valfmt='%.8f')
    # s_pwp = plt.Slider(axfreq, 'PW Penalty', 0, 400, 10, valfmt='%.8f')
    # s_sp = plt.Slider(axamp, 'S Penalty', 0, 400, 10, valfmt='%.8f')
    #
    # def update(val):
    #     pw_penalty = s_pwp.val
    #     s_penalty = s_sp.val
    #
    #     ed, dir = dynamic_prog(mat, pw_penalty=pw_penalty, s_penalty=s_penalty)
    #     im_overlay, pairs = overlay(dir, mat)
    #
    #     best6_pw = get_best_pw(sm_metric,pairs,6)
    #     best11_pw = get_best_pw(sm_metric,pairs,11)
    #     best23_pw = get_best_pw(sm_metric,pairs,23)
    #     best33_pw = get_best_pw(sm_metric,pairs,33)
    #     print("[PW8=%s], [PW11=%s], [PW42=%s [PW68=%s]" % (best6_pw, best11_pw, best23_pw, best33_pw))
    #
    #     axes[1,0].imshow(ed)
    #     axes[1,1].imshow(im_overlay)
    #     fig.canvas.draw_idle()
    #
    # s_pwp.on_changed(update)
    # s_sp.on_changed(update)
    # plt.show()

    #%% Runtime Experiments
    mat = sm_matches
    pw_penalty = 50
    s_penalty = 50
    ed, dir = dynamic_prog(mat, pw_penalty=pw_penalty, s_penalty=s_penalty)
    im_overlay, pairs = overlay(dir, mat)

    # Figure prep
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

    # Figure
    plt.figure()
    ax = plt.gca()
    ax.set_title('Dynamic Programming Back-Tracing')
    plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right')
    plt.imshow(im_overlay)
    plt.xticks(pw_ticks_idxs, pw_ticks_vals)
    plt.yticks(np.arange(0, len(s_label)), np.arange(1, len(s_label) + 1))

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(8)

    plt.xlabel('PW Level')
    plt.ylabel('S Level')

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

    # imshow_matches(dynamic_prog(norm, pw_penalty=1, s_penalty=1)[1], '')
    # imshow_matches(overlay(dynamic_prog(sm_matches, 0.9, 0.1)[0], sm_matches), '')

    # aoi = ed[32:35, 38:41]
    # best_s = pw_label[np.argmin(ed,axis=1)]
    # print("PW68 best match", best_pw[np.where(pw_label==68)])
    # print("S33 best match", best_s[np.where(s_label==33)])

