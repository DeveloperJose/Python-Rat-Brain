# Author: Jose G Perez
# Version 1.0
# Last Modified: January 31, 2018
import numpy as np
import cv2
import cProfile
RADIUS = 25
RADIUS_SQUARED = RADIUS ** 2
SCALE_THRESHOLD = 3
DISTANCE_THRESHOLD = 200
RESPONSE_THRESHOLD = 0.01

def perform_match(s_idx):
    global s_kp, s_des, pw_kp, pw_des
    matches = []
    print('s_idx', s_idx)
    for pw_idx in range(pw_kp.shape[0]):
        matches.append(match(s_kp[s_idx], s_des[s_idx], pw_kp[pw_idx], pw_des[pw_idx]))

    np.savez_compressed(str(s_idx) + '-M', m=matches)

def match(kp1, des1, kp2, des2):
    matches = []
    for idx1 in range(len(kp1)):
        kpa = kp1[idx1]
        kpa_response = kpa[4]
        if kpa_response < RESPONSE_THRESHOLD:
            continue

        kpa_x = kpa[0]
        kpa_y = kpa[1]
        kpa_size = kpa[2]
        #kpa_angle = kpa[3]
        dpa = des1[idx1]

        best_distance = float('inf')
        best_idx2 = -1

        for idx2 in range(len(kp2)):
            kpb = kp2[idx2]
            # 0: Response Strength
            kpb_response = kpb[4]
            if kpb_response < RESPONSE_THRESHOLD:
                continue

            # 1: Distance/Radius Check
            kpb_x = kpb[0]
            kpb_y = kpb[1]
            d_pt_squared = (kpb_x - kpa_x) ** 2 + (kpb_y - kpa_y) ** 2
            if d_pt_squared >= RADIUS_SQUARED:
                continue

            # 2: Scale Difference
            kpb_size = kpb[2]
            scale_diff = abs(kpa_size - kpb_size)
            if scale_diff > SCALE_THRESHOLD:
                continue

            # 3: Descriptor L2 Norm
            dpb = des2[idx2]
            d_des = np.linalg.norm(dpb - dpa)
            if d_des < best_distance:
                best_distance = d_des
                best_idx2 = idx2

            # 4: ?? Angle ??
            #kpb_angle = kpb[3]

        if best_idx2 == -1 or best_distance > DISTANCE_THRESHOLD:
            continue

        match = np.array([idx1, best_idx2, best_distance], dtype=np.int32)
        matches.append(match)

    return np.asarray(matches)

def match_details(kp1, des1, kp2, des2):
    matches = []
    matches_cv = []
    stat_s_diff = set()
    stat_d_dist = set()
    stat_resp_kp1 = set()
    stat_resp_kp2 = set()
    stat_rad = set()
    for idx1 in range(len(kp1)):
        kpa = kp1[idx1]
        kpa_response = kpa[4]
        stat_resp_kp1.add(kpa_response)
        if kpa_response < RESPONSE_THRESHOLD:
            continue

        kpa_x = kpa[0]
        kpa_y = kpa[1]
        kpa_size = kpa[2]
        #kpa_angle = kpa[3]
        dpa = des1[idx1]

        best_distance = float('inf')
        best_idx2 = -1

        for idx2 in range(len(kp2)):
            kpb = kp2[idx2]
            # 0: Response Strength
            kpb_response = kpb[4]
            stat_resp_kp2.add(kpb_response)
            if kpb_response < RESPONSE_THRESHOLD:
                continue

            # 1: Distance/Radius Check
            kpb_x = kpb[0]
            kpb_y = kpb[1]
            d_pt_squared = (kpb_x - kpa_x) ** 2 + (kpb_y - kpa_y) ** 2
            stat_rad.add(np.linalg.norm(np.array([kpb_x,kpb_y]) - np.array([kpa_x,kpa_y])))
            if d_pt_squared >= RADIUS_SQUARED:
                continue

            # 2: Scale Difference
            kpb_size = kpb[2]
            scale_diff = abs(kpa_size - kpb_size)
            stat_s_diff.add(scale_diff)
            if scale_diff > SCALE_THRESHOLD:
                continue

            # 3: Descriptor L2 Norm
            dpb = des2[idx2]
            d_des = np.linalg.norm(dpb - dpa)
            stat_d_dist.add(d_des)
            if d_des < best_distance:
                best_distance = d_des
                best_idx2 = idx2

            # 4: ?? Angle ??
            # kpb_angle = kpb[3]

        if best_idx2 == -1 or best_distance > DISTANCE_THRESHOLD:
            continue

        match = np.array([idx1, best_idx2, best_distance], dtype=np.int32)
        matches.append(match)

        match = cv2.DMatch()
        match.queryIdx = idx1
        match.imgIdx = idx1
        match.trainIdx = best_idx2
        match.distance = best_distance
        matches_cv.append(match)

    return np.asarray(matches), np.asarray(matches_cv), \
           np.asarray(stat_s_diff), np.asarray(stat_d_dist), \
           np.asarray(stat_resp_kp1), np.asarray(stat_resp_kp2), \
           np.asarray(stat_rad)