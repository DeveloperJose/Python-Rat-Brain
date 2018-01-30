# Author: Jose G Perez <josegperez@mail.com>
import cv2
import os
import numpy as np
import pylab as plt
from multiprocessing.pool import ThreadPool
from timeit import default_timer as timer

FIGURE_IDX = 0
def imshow(im, title=''):
    global FIGURE_IDX
    plt.figure(FIGURE_IDX)
    plt.axis('off')
    plt.tick_params(axis='both',
                    left='off', top='off', right='off', bottom='off',
                    labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    plt.title(title)
    plt.imshow(im)

    FIGURE_IDX += 1

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
            d_pt_squared = (kpb_x - kpa_x) ** 2 + (kpb_x - kpa_x) ** 2
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
SIFT = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.05, edgeThreshold=100, sigma=2)
RADIUS = 50
RADIUS_SQUARED = RADIUS ** 2
SCALE_THRESHOLD = 3
DISTANCE_THRESHOLD = 200
RESPONSE_THRESHOLD = 0.01

# Precompute SIFT
precompute_sift()

# Load the data
s_im, s_label, s_kp, s_des = load('S_BB_V1_SIFT.npz')
pw_im, pw_label, pw_kp, pw_des = load('PW_BB_V1_SIFT.npz')

# =======******* Individual Testing [S33 PW68]
s_idx = np.where(s_label == 33)[0][0]
pw_idx = np.where(pw_label == 68)[0][0]
(im1, kp1, des1) = (s_im[s_idx], s_kp[s_idx], s_des[s_idx])
(im2, kp2, des2) = (pw_im[pw_idx], pw_kp[pw_idx], pw_des[pw_idx])
matches = match(kp1,des1,kp2,des2)
# Convert to OpenCV objects for viewing
matches = match_to_cv(matches)
kp1 = array_to_kp(kp1)
kp2 = array_to_kp(kp2)
im_matches = cv2.drawMatches(im1, kp1, im2, kp2, matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
plt.gray()
imshow(im_matches, 'New Matching')