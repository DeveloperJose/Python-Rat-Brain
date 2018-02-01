import numpy as np
import cv2

DISTANCE_RATIO = 0.95
RANSAC_REPROJ_TRESHHOLD = 10 # The higher the threshold, the lower the inliers
RANSAC_MAX_ITERS = 2000
RANSAC_CONFIDENCE = 0.99
BF = cv2.BFMatcher(normType=cv2.NORM_L2)

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
        matches.append(np.array([mask.sum()]))

    np.savez_compressed(str(s_idx) + '-R', m=matches)

def imshow_ransac():
    import os
    import pylab as plt
    im_inliers = np.zeros((73, 89))
    for sidx in range(73):
        path = os.path.join('sm_ransac', str(sidx) + '-R.npz')
        m = np.load(path)['m']
        inliers = []
        for pidx in range(89):
            pw_matches = m[pidx]
            inliers.append(pw_matches[0])
        inlier_norm = np.array(inliers) / np.max(inliers)
        im_inliers[sidx] = (inlier_norm * 255).reshape(1, 89)

    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    ax.set_xlabel('PW Level')
    ax.set_ylabel('S Level')
    ax.set_xticks(np.arange(0,89,5))
    ax.set_yticks(np.arange(0,72,5))
    ax.set_title('Inliers')
    plt.set_cmap(plt.get_cmap('hot'))
    plt.imshow(im_inliers)