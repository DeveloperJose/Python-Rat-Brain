# Author: Jose G Perez <josegperez@mail.com>
import cv2
import os
import numpy as np
import pylab as plt
from multiprocessing.pool import Pool
from timeit import default_timer as timer

FIGURE_IDX = 0
def imshow(im, title=''):
    global FIGURE_IDX
    figure = plt.figure(FIGURE_IDX)
    plt.axis('off')
    plt.tick_params(axis='both',
                    left='off', top='off', right='off', bottom='off',
                    labelleft='off', labeltop='off', labelright='off', labelbottom='off')

    plt.title(title)
    plt.imshow(im)
    FIGURE_IDX += 1
    return figure

def kp_to_array(kp):
    array = np.zeros((len(kp), 7), dtype=np.float32)
    for idx in range(array.shape[0]):
        k = kp[idx]
        array[idx] = np.array([k.pt[0], k.pt[1], k.size,k.angle,k.response,k.octave,k.class_id])
    return array

def array_to_kp(array):
    kp = []
    for idx in range(array.shape[0]):
        k = array[idx]
        kp.append(cv2.KeyPoint(k[0],k[1],k[2],k[3],k[4],k[5],k[6]))
    return kp

def precompute_sift():
    if not os.path.isfile('S_BB_V1_SIFT.npz'):
        print('Precomputing SIFT for S_BB_V1...')
        s_data = np.load('S_BB_V1.npz')
        s_im = s_data['images']
        s_labels = s_data['labels']
        s_kp = []
        s_des = []

        for i in range(0, s_im.shape[0]):
            kp, des = SIFT.detectAndCompute(s_im[i], None)
            kp = kp_to_array(kp)
            s_kp.append(kp)
            s_des.append(des)

        s_kp = np.asarray(s_kp)
        s_des = np.asarray(s_des)

        np.savez_compressed('S_BB_V1_SIFT', images=s_im, labels=s_labels, kp=s_kp, des=s_des)

    if not os.path.isfile('PW_BB_V1_SIFT.npz'):
        print('Precomputing SIFT for PW_BB_V1...')
        pw_data = np.load('PW_BB_V1.npz')
        pw_im = pw_data['images']
        pw_labels = pw_data['labels']
        pw_kp = []
        pw_des = []

        for i in range(0, pw_im.shape[0]):
            kp, des = SIFT.detectAndCompute(pw_im[i], None)
            kp = kp_to_array(kp)
            pw_kp.append(kp)
            pw_des.append(des)

        pw_kp = np.asarray(pw_kp)
        pw_des = np.asarray(pw_des)

        np.savez_compressed('PW_BB_V1_SIFT', images=pw_im, labels=pw_labels, kp=pw_kp, des=pw_des)

def load(path):
    data = np.load(path)
    return data['images'], data['labels'], data['kp'], data['des']

def match_to_cv(matches):
    cv = []
    for i in range(matches.shape[0]):
        m = matches[i]
        temp = cv2.DMatch()
        temp.queryIdx = int(m[0])
        temp.imgIdx = int(m[0])
        temp.trainIdx = int(m[1])
        temp.distance = int(m[2])
        cv.append(temp)
    return cv

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

np.random.seed(1)
np.set_printoptions(threshold=np.nan, linewidth=1000)
BF = cv2.BFMatcher(normType=cv2.NORM_L2)
SIFT = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.05, edgeThreshold=100, sigma=2)
RADIUS = 25
RADIUS_SQUARED = RADIUS ** 2
SCALE_THRESHOLD = 3
DISTANCE_THRESHOLD = 200
RESPONSE_THRESHOLD = 0.01

DISTANCE_RATIO = 0.95
RANSAC_REPROJ_TRESHHOLD = 10 # The higher the threshold, the lower the inliers
RANSAC_MAX_ITERS = 2000
RANSAC_CONFIDENCE = 0.99

# Precompute SIFT
precompute_sift()

# Load the data
s_im, s_label, s_kp, s_des = load('S_BB_V1_SIFT.npz')
pw_im, pw_label, pw_kp, pw_des = load('PW_BB_V1_SIFT.npz')

# =======******* Individual Testing [S33 PW68]
# s_idx = np.where(s_label == 33)[0][0]
# pw_idx = np.where(pw_label == 68)[0][0]
# (im1, kp1, des1) = (s_im[s_idx], s_kp[s_idx], s_des[s_idx])
# (im2, kp2, des2) = (pw_im[pw_idx], pw_kp[pw_idx], pw_des[pw_idx])
# matches = match(kp1,des1,kp2,des2)
# # Convert to OpenCV objects for viewing
# matches = match_to_cv(matches)
# kp1 = array_to_kp(kp1)
# kp2 = array_to_kp(kp2)
# im_matches = cv2.drawMatches(im1, kp1, im2, kp2, matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# plt.gray()
# imshow(im_matches, 'New Matching')

# =======******* Row Testing [S33 and all PW]
# PW68 is IDX:39
# matches = []
# s_idx = np.where(s_label == 33)[0][0]
# (im1, kp1, des1) = (s_im[s_idx], s_kp[s_idx], s_des[s_idx])
# for pw_idx in range(pw_im.shape[0]):
#     print(pw_idx, '/', pw_im.shape[0])
#     (im2, kp2, des2) = (pw_im[pw_idx], pw_kp[pw_idx], pw_des[pw_idx])
#     m = match(kp1, des1, kp2, des2)
#     matches.append(m)
#
# count = []
# for match in matches:
#     count.append(len(match))
#
# count_norm = np.array(count) / np.max(count)
# count_im = (count_norm * 255).reshape(1, 89)
# plt.gray()
# plt.imshow(count_im)
total = 0
def perform_pass(s_idx):
    global s_kp, s_des, pw_kp, pw_des
    matches = []
    print('s_idx', s_idx)
    for pw_idx in range(pw_kp.shape[0]):
        matches.append(match(s_kp[s_idx], s_des[s_idx], pw_kp[pw_idx], pw_des[pw_idx]))

    np.savez_compressed(str(s_idx) + '-M', m=matches)
    #return matches

def perform_ransac(s_idx):
    global s_kp, s_des, pw_kp, pw_des
    matches = []
    print('s_idx', s_idx)
    for pw_idx in range(pw_kp.shape[0]):
        mat = BF.knnMatch(s_des[s_idx], pw_des[pw_idx], k=2)
        mat = [m[0] for m in mat if len(m) == 2 and m[0].distance < m[1].distance * DISTANCE_RATIO]
        src_pts = np.float32([(s_kp[s_idx][m.queryIdx][0],s_kp[s_idx][m.queryIdx][1]) for m in mat])
        dst_pts = np.float32([(pw_kp[pw_idx][m.trainIdx][0],pw_kp[pw_idx][m.trainIdx][1]) for m in mat])
        H, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ_TRESHHOLD,maxIters=RANSAC_MAX_ITERS, confidence=RANSAC_CONFIDENCE)
        #matchesMask = mask.ravel().tolist()
        matches.append(np.array([mask.sum()]))

    np.savez_compressed(str(s_idx) + '-R', m=matches)

# if __name__ == '__main__':
#     time_start = timer()
#
#     pool = Pool()
#     s_idx = range(s_kp.shape[0])
#     # s_idx = range(32, 34)
#
#     print('Begin pool work')
#     # pool.map(perform_pass, s_idx)
#     pool.map(perform_ransac, s_idx)
#     pool.close()
#     pool.join()
#
#     duration = timer() - time_start
#     print("Program took %.3fs" % duration)

# Drawwww
def get_drawings(path):
    im_result = np.zeros((73, 89))
    im_metric = np.zeros((73, 89))
    for sidx in range(73):
        temp_path = os.path.join(path, str(sidx) + '-M.npz')
        m = np.load(temp_path)['m']
        count = []
        metric = []
        for pidx in range(89):
            pw_matches = m[pidx]
            count.append(len(pw_matches))
            metric.append(len(pw_matches) / (len(s_kp[sidx]) + len(pw_kp[pidx]) - len(pw_matches)))
        count_norm = np.array(count) / np.max(count)
        metric_norm = np.array(metric) / np.max(metric)
        im_result[sidx] = (count_norm * 255).reshape(1, 89)
        im_metric[sidx] = (metric_norm * 255).reshape(1, 89)

    return im_result, im_metric

def imshow_matches(im, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('PW Level')
    ax.set_ylabel('S Level')
    ax.set_xticks(np.arange(0,89,5))
    ax.set_yticks(np.arange(0,72,5))
    ax.set_title(title)
    plt.set_cmap(plt.get_cmap('hot'))
    plt.imshow(im)

im_result, im_metric = get_drawings('sm_v1')
imshow_matches(im_result, 'Experiment 1: Matches Count')
imshow_matches(im_metric, 'Experiment 1: Metric')

im_result, im_metric = get_drawings('sm_v2')
imshow_matches(im_result, 'Experiment 2: Matches Count')
imshow_matches(im_metric, 'Experiment 2: Metric')

# RANSAC
# im_inliers = np.zeros((73, 89))
# for sidx in range(73):
#     path = os.path.join('sm_ransac', str(sidx) + '-R.npz')
#     m = np.load(path)['m']
#     inliers = []
#     for pidx in range(89):
#         pw_matches = m[pidx]
#         inliers.append(pw_matches[0])
#     inlier_norm = np.array(inliers) / np.max(inliers)
#     im_inliers[sidx] = (inlier_norm * 255).reshape(1, 89)
#
# fig = plt.figure(0)
# ax = fig.add_subplot(111)
# ax.set_xlabel('PW Level')
# ax.set_ylabel('S Level')
# ax.set_xticks(np.arange(0,89,5))
# ax.set_yticks(np.arange(0,72,5))
# ax.set_title('Inliers')
# plt.set_cmap(plt.get_cmap('hot'))
# plt.imshow(im_inliers)