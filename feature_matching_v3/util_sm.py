# Author: Jose G Perez
# Version 1.0
# Last Modified: January 31, 2018
import numpy as np
import os

def load_sm(folder, s_kp, pw_kp):
    sm_match = np.zeros((s_kp.shape[0],pw_kp.shape[0]))
    sm_metric = np.zeros((s_kp.shape[0],pw_kp.shape[0]))
    for sidx in range(sm_match.shape[0]):
        path = os.path.join(folder, str(sidx) + '-M.npz')
        m = np.load(path)['m']
        count = []
        metric = []
        for pidx in range(sm_match.shape[1]):
            pw_matches = m[pidx]
            count.append(len(pw_matches))
            metric.append(len(pw_matches) / (len(s_kp[sidx]) + len(pw_kp[pidx]) - len(pw_matches)))
        sm_match[sidx] = np.asarray(count)
        sm_metric[sidx] = np.asarray(metric)
    return sm_match, sm_metric

def norm2_sm(sm, max_value=255):
    im_result = np.zeros_like(sm)
    for idx in range(sm.shape[0]):
        norm = sm[idx] / np.max(sm[idx])
        im_result[idx] = (norm*max_value).reshape(1, sm.shape[1])
    return im_result

def norm_prob_sm(sm):
    norm = np.zeros_like(sm)
    for idx in range(sm.shape[0]):
        norm[idx] = sm[idx] / np.sum(sm[idx])
    return norm

def norm_sm(sm, max_value=255, min_value=0):
    im_result = np.zeros_like(sm)
    for idx in range(sm.shape[0]):
        x = sm[idx]
        norm = ((x - np.min(x))*(max_value-min_value)) / (np.max(x)-np.min(x)) + min_value
        im_result[idx] = (norm).reshape(1, sm.shape[1])
    return im_result
