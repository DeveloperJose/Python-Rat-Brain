import numpy as np
from timeit import default_timer as timer
from multiprocessing.pool import Pool

from util_ransac import perform_ransac
from util_matching import match
from util_sift import precompute_sift, load_sift

RADIUS = 25
RADIUS_SQUARED = RADIUS ** 2
SCALE_THRESHOLD = 3
DISTANCE_THRESHOLD = 200
RESPONSE_THRESHOLD = 0.01
precompute_sift('S_BB_V4', 'PW_BB_V4')
s_im, s_label, s_kp, s_des = load_sift('S_BB_V4_SIFT.npz')
pw_im, pw_label, pw_kp, pw_des = load_sift('PW_BB_V4_SIFT.npz')

def perform_match(s_idx):
    global s_kp, s_des, pw_kp, pw_des
    matches = []
    print('s_idx', s_idx)
    for pw_idx in range(pw_kp.shape[0]):
        matches.append(match(s_kp[s_idx], s_des[s_idx], pw_kp[pw_idx], pw_des[pw_idx]))

    np.savez_compressed(str(s_idx) + '-M', m=matches)

if __name__ == '__main__':
    time_start = timer()

    pool = Pool()
    s_idx = range(s_kp.shape[0])

    print('Begin pool work')
    pool.map(perform_match, s_idx)
    # pool.map(perform_ransac, s_idx)
    pool.close()
    pool.join()

    duration = timer() - time_start
    duration_m = duration / 60
    print("Program took %.3fs %.3fm" % (duration, duration_m))