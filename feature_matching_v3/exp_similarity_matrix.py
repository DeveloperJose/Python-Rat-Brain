# Author: Jose G Perez
# Version 1.0
# Last Modified: January 31, 2018
from util_im import imshow_matches
from util_sm import load_sm, norm_sm
from util_sift import precompute_sift, load_sift
precompute_sift('S_BB_V1', 'PW_BB_V1')
s_im, s_label, s_kp, s_des = load_sift('S_BB_V1_SIFT.npz')
pw_im, pw_label, pw_kp, pw_des = load_sift('PW_BB_V1_SIFT.npz')

sm_v1_match, sm_v1_metric = load_sm('sm_v1', s_kp, pw_kp)
sm_v2_match, sm_v2_metric = load_sm('sm_v2', s_kp, pw_kp)

imshow_matches(norm_sm(sm_v1_match), 'Experiment 1: Matches Count')
imshow_matches(norm_sm(sm_v1_metric), 'Experiment 1: Metric')

imshow_matches(norm_sm(sm_v2_match), 'Experiment 2: Matches Count')
imshow_matches(norm_sm(sm_v2_metric), 'Experiment 2: Metric')