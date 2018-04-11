import cProfile
import numpy as np
from timeit import default_timer as timer
import multiprocessing
from multiprocessing.pool import Pool
from util_sift import precompute_sift, load_sift
import itertools

RADIUS = 25
RADIUS_SQUARED = RADIUS ** 2
SCALE_THRESHOLD = 3
DISTANCE_THRESHOLD = 200
RESPONSE_THRESHOLD = 0.01
precompute_sift('S_BB_V1', 'PW_BB_V1')
s_im, s_label, s_kp, s_des = load_sift('S_BB_V1_SIFT.npz')
pw_im, pw_label, pw_kp, pw_des = load_sift('PW_BB_V1_SIFT.npz')

(best_distance, best_idx2) = (float('inf'), -1)
def match_v3_helper2(kpa, dpa, kp2, des2, idx2):
    global best_distance
    kpb = kp2[idx2]
    # (kpa_x, kpa_y, kpa_size, kpa_response, dpa) = (kpa[0], kpa[1], kpa[2], kpa[4], des1[idx1])
    # (kpb_x, kpb_y, kpb_size, kpb_response, dpb) = (kpb[0], kpb[1], kpb[2], kpb[4], des2[idx2])
    if kpb[4] < RESPONSE_THRESHOLD \
            or (kpb[0] - kpa[0]) ** 2 + (kpb[1] - kpa[1]) ** 2 >= RADIUS_SQUARED \
            or abs(kpa[2] - kpb[2]) > SCALE_THRESHOLD:
        return

    # 3: Descriptor L2 Norm
    d_des = np.linalg.norm(des2[idx2] - dpa)
    if(d_des < best_distance):
        best_distance = d_des
        best_idx2 = idx2

def match_v3_helper(kp1, des1, kp2, des2, idx1):
    global best_distance, best_idx2
    kpa = kp1[idx1]
    if kpa[4] < RESPONSE_THRESHOLD:
        return

    [match_v3_helper2(kpa, des1[idx1], kp2, des2, idx2) for idx2 in range(len(kp2))]

    if best_idx2 == -1 or best_distance > DISTANCE_THRESHOLD:
        return

    return np.array([idx1, best_idx2, best_distance], dtype=np.int32)

def match_v3(kp1, des1, kp2, des2):
    matches = [match_v3_helper(kp1,des1,kp2,des2,idx1) for idx1 in range(len(kp1))]

def match_v2_helper(kp1, des1, kp2, des2, idx1):
    kpa = kp1[idx1]
    (kpa_x, kpa_y, kpa_size, kpa_response, dpa) = (kpa[0], kpa[1], kpa[2], kpa[4], des1[idx1])
    (best_distance, best_idx2) = (float('inf'), -1)

    if kpa_response < RESPONSE_THRESHOLD:
        return

    for idx2 in range(len(kp2)):
        kpb = kp2[idx2]
        (kpb_x, kpb_y, kpb_size, kpb_response, dpb) = (kpb[0], kpb[1], kpb[2], kpb[4], des2[idx2])
        if kpb_response < RESPONSE_THRESHOLD \
            or (kpb_x - kpa_x) ** 2 + (kpb_y - kpa_y) ** 2 >= RADIUS_SQUARED \
            or abs(kpa_size - kpb_size) > SCALE_THRESHOLD:
            continue

        # 3: Descriptor L2 Norm
        d_des = np.linalg.norm(dpb - dpa)
        if d_des < best_distance:
            best_distance = d_des
            best_idx2 = idx2

    if best_idx2 == -1 or best_distance > DISTANCE_THRESHOLD:
        return

    return np.array([idx1, best_idx2, best_distance], dtype=np.int32)

def match_v2(kp1, des1, kp2, des2):
    matches = (match_v2_helper(kp1,des1,kp2,des2,idx1) for idx1 in range(len(kp1)))

def match(kp1, des1, kp2, des2):
    matches = []
    for idx1 in range(len(kp1)):
        kpa = kp1[idx1]
        (kpa_x, kpa_y, kpa_size, kpa_response, dpa) = (kpa[0], kpa[1], kpa[2], kpa[4], des1[idx1])
        (best_distance, best_idx2) = (float('inf'), -1)

        if kpa_response < RESPONSE_THRESHOLD:
            return

        for idx2 in range(len(kp2)):
            kpb = kp2[idx2]
            (kpb_x, kpb_y, kpb_size, kpb_response, dpb) = (kpb[0], kpb[1], kpb[2], kpb[4], des2[idx2])
            if kpb_response < RESPONSE_THRESHOLD \
                    or (kpb_x - kpa_x) ** 2 + (kpb_y - kpa_y) ** 2 >= RADIUS_SQUARED \
                    or abs(kpa_size - kpb_size) > SCALE_THRESHOLD:
                continue

            # 3: Descriptor L2 Norm
            d_des = np.linalg.norm(dpb - dpa)
            if d_des < best_distance:
                best_distance = d_des
                best_idx2 = idx2

        if best_idx2 == -1 or best_distance > DISTANCE_THRESHOLD:
            return

        matches.append(np.array([idx1, best_idx2, best_distance], dtype=np.int32))

    return np.asarray(matches)

def perform_match(s_idx):
    global s_kp, s_des, pw_kp, pw_des
    matches = []
    print('s_idx', s_idx)
    for pw_idx in range(5):
        matches.append(match(s_kp[s_idx], s_des[s_idx], pw_kp[pw_idx], pw_des[pw_idx]))

def perform_match_v2(s_idx):
    global s_kp, s_des, pw_kp, pw_des
    matches = []
    print('s_idx', s_idx)
    matches = [match(s_kp[s_idx], s_des[s_idx], pw_kp[pw_idx], pw_des[pw_idx]) for pw_idx in range(5)]

def test():
    s_idx = 0
    return [match_v3(s_kp[s_idx], s_des[s_idx], pw_kp[pw_idx], pw_des[pw_idx]) for pw_idx in range(pw_kp.shape[0])]

def perform_match_v3(s_idx):
    global s_kp, s_des, pw_kp, pw_des
    print('s_idx', s_idx)
    # matches = (match(s_kp[s_idx], s_des[s_idx], pw_kp[pw_idx], pw_des[pw_idx]) for pw_idx in range(5))
    # matches = (match_v2(s_kp[s_idx], s_des[s_idx], pw_kp[pw_idx], pw_des[pw_idx]) for pw_idx in range(5))
    matches = (match_v3(s_kp[s_idx], s_des[s_idx], pw_kp[pw_idx], pw_des[pw_idx]) for pw_idx in range(pw_kp.shape[0]))
    # np.savez_compressed(str(s_idx) + '-M', m=np.asarray(matches2))

if __name__ == '__main__':
    print('Begin pool work')
    pool = Pool()
    # s_idx = range(2)
    s_idx = range(s_kp.shape[0])
    time_start = timer()
    # pool.map(perform_match, s_idx)
    # pool.map(perform_match_v2, s_idx)
    pool.map(perform_match_v3, s_idx)
    time_end = timer()
    pool.close()
    pool.join()
    duration = time_end - time_start
    print("Program took %.3fs" % duration)