# Author: Jose G Perez
# Version 1.0
# Last Modified: January 31, 2018
from util_im import imshow_matches
from util_sm import load_sm, norm_sm
from util_sift import precompute_sift, load_sift
import pylab as plt
import numpy as np
precompute_sift('S_BB_V4', 'PW_BB_V4')
s_im, s_label, s_kp, s_des = load_sift('S_BB_V4_SIFT.npz')
pw_im, pw_label, pw_kp, pw_des = load_sift('PW_BB_V4_SIFT.npz')

# sm_v1_match, sm_v1_metric = load_sm('sm_v1', s_kp, pw_kp)
sm_matches, sm_metric = load_sm('sm_v4', s_kp, pw_kp)

# imshow_matches(norm_sm(sm_v1_match), 'Experiment 1: Matches Count')
# imshow_matches(norm_sm(sm_v1_metric), 'Experiment 1: Metric')
#
# imshow_matches(norm_sm(sm_v2_match), 'Experiment 2: Matches Count')
# imshow_matches(norm_sm(sm_v2_metric), 'Experiment 2: Metric')

#%%
pw_ticks_idxs = [0]
pw_ticks_vals = [pw_label[0]]
for x in range(len(pw_label)):
    try:
        diff = pw_label[x+1] - pw_label[x]
        if diff > 1:
            pw_ticks_idxs.append(x)
            pw_ticks_vals.append(pw_label[x])
            # print("IDX: ", x, "DIFF:", diff)
    except:
        continue

pw_ticks_idxs.append(len(pw_label)-1)
pw_ticks_vals.append(pw_label[-1])

#%%
plt.figure()
ax = plt.gca()
ax.set_title('Similarity Matrix (Matches)')
plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right')
plt.imshow(sm_matches)
plt.xticks(pw_ticks_idxs, pw_ticks_vals)
plt.yticks(np.arange(0, len(s_label)), np.arange(1, len(s_label)+1))

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(8)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(8)
    # tick.label.set_rotation('vertical')

plt.xlabel('PW Level')
plt.ylabel('S Level')
# heatmap = plt.pcolor(sm_matches)
colorbar = plt.colorbar()
colorbar.set_label('# of Matches')

#%%
plt.figure()
ax = plt.gca()
ax.set_title('Similarity Matrix (Metric)')
plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right')
plt.imshow(sm_metric)
plt.xticks(pw_ticks_idxs, pw_ticks_vals)
plt.yticks(np.arange(0, len(s_label)), np.arange(1, len(s_label)+1))

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(8)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(8)
    # tick.label.set_rotation('vertical')

plt.xlabel('PW Level')
plt.ylabel('S Level')
# heatmap = plt.pcolor(sm_matches)
colorbar = plt.colorbar()
colorbar.set_label('Metric Value')