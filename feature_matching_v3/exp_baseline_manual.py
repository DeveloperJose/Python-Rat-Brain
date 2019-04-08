import numpy as np
import os
import pylab as plt
import cv2
from skimage.transform import warp, PiecewiseAffineTransform
from PIL import Image
from timeit import default_timer as timer

USE_SIFT = False
USE_OPTICALFLOW = False

def load(filename):
    WIDTH = 800
    HEIGHT = 400
    im = Image.open(filename)
    im = im.resize((WIDTH, HEIGHT), Image.LANCZOS)
    P = np.array(im, dtype=np.uint8)
    return P[:, 0:P.shape[1] // 2, :]

def to_gray(P):
    return P.mean(axis=2)

def get_controls(P):
    LEFT_PAD = 0
    RIGHT_PAD = 0
    COL_INTERVALS = 5
    CTRL_THRESHOLD = 200
    w = P.shape[0]
    h = P.shape[1]
    ctrl_pts = [[0, 0], [w, 0], [0, h], [w, h]]
    # Top to bottom
    for col in range(LEFT_PAD, P.shape[1] - RIGHT_PAD, COL_INTERVALS):
        for row in range(0, P.shape[0], 1):
            if P[row, col] <= CTRL_THRESHOLD:
                ctrl_pts.append([row, col])
                break

    # Bottom to top
    for col in range(LEFT_PAD, P.shape[1] - RIGHT_PAD, COL_INTERVALS):
        for row in range(P.shape[0] - 1, 0, -1):
            if P[row, col] <= CTRL_THRESHOLD:
                ctrl_pts.append([row, col])
                break

    return ctrl_pts

#%% Load images
print("Loading images")
dir1 = 'C:/Users/xeroj/Dropbox/Training data sets - Khan-Fuentes/Paxinos and Watson, 2014 (7th Edition) Image set/'
dir2 = 'C:/Users/xeroj/Downloads/Processed'

P1 = load(os.path.join(dir1, 'RBSC7-068.jpg'))
P2 = load(os.path.join(dir1, 'RBSC7-070.jpg'))

PM1 = load(os.path.join(dir2, '18-016 LHA s4t2.tif'))
PM2 = load(os.path.join(dir2, '18-016 LHA s4t3.tif'))
PM3 = load(os.path.join(dir2, '18-016 LHA s4t4.tif'))

#%% Plate selection
S1 = P1
S2 = P2

#%% SIFT Stuff
if USE_SIFT:
    #%% SIFT Features
    print("Computing SIFT features")
    # SIFT = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.02, edgeThreshold=100, sigma=2)
    SIFT = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = SIFT.detectAndCompute(S1, None)
    kp2, des2 = SIFT.detectAndCompute(S2, None)

    #%% Matching (SIFT + Homography)
    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1,des2, k=2)
    # matches = np.asarray([m for m in matches if m[0].distance < 0.9*m[1].distance])
    # if len(matches[:,0]) >= 4:
    #     src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    #     dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    #     H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 50.0)
    #     im_match = cv2.drawMatches(P1, kp1, P2, kp2, matches[:,0], None, flags=2)
    #     plt.figure()
    #     plt.imshow(im_match)
    #
    # for m in matches[:,0]:
    #     pt1 = kp1[m.queryIdx].pt
    #     pt2 = kp2[m.trainIdx].pt
    #
    #     (x1, y1) = (pt1[1], pt1[0])
    #     (x2, y2) = (pt2[1], pt2[0])
    #
    #     x1 = (x1 + x2) / 2
    #     y1 = (y1 + y2) / 2
    #     # x2 = (x1 + x2) / 2
    #     # y2 = (y1 + y2) / 2
    #
    #     src_pts.append([x1, y1])
    #     dst_pts.append([x2, y2])
    #
    # src_pts = np.array(src_pts, dtype=np.float32)
    # dst_pts = np.array(dst_pts, dtype=np.float32)

    #%% Matching (Mine)
    import util_matching, util_cv, util_sift
    src_pts = [[0, 0], [S1.shape[0], 0], [0, S1.shape[1]], [S1.shape[0], S1.shape[1]]]
    dst_pts = [[0, 0], [S2.shape[0], 0], [0, S2.shape[1]], [S2.shape[0], S2.shape[1]]]
    print("Performing my matching algorithm")
    matches2 = util_matching.match(util_sift.kp_to_array(kp1), des1, util_sift.kp_to_array(kp2), des2)
    matches2 = util_cv.match_to_cv(matches2)
    im_match2 = cv2.drawMatches(S1, kp1, S1, kp2, matches2, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    plt.figure(6)
    plt.imshow(im_match2)

    print("Adding matching points for warping")
    for m in matches2:
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[m.trainIdx].pt

        (x1, y1) = (pt1[1], pt1[0])
        (x2, y2) = (pt2[1], pt2[0])

        # x1 = (x1 + x2) / 2
        # y1 = (y1 + y2) / 2
        x2 = (x1 + x2) / 2
        y2 = (y1 + y2) / 2

        src_pts.append([x1, y1])
        dst_pts.append([x2, y2])

    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)

    #%% Warping
    print("Warping")
    tform = PiecewiseAffineTransform()
    e1 = tform.estimate(dst_pts, src_pts)
    im_warp1 = warp(S1, tform)
    im_warp1 = to_gray((im_warp1 * 255).astype(np.uint8))

    tform = PiecewiseAffineTransform()
    e2 = tform.estimate(src_pts, dst_pts)
    im_warp2 = warp(S2, tform)
    im_warp2 = to_gray((im_warp2 * 255).astype(np.uint8))

    print("Cross-Dissolve")
    im_gen = (im_warp1 * 0.5) + (im_warp2 * 0.5)
    plt.figure(7)
    plt.suptitle("SIFT Generated")
    plt.imshow(im_gen, cmap='gray')

#%% OF
if USE_OPTICALFLOW:
    print("Optical Flow")
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=500,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.03))

    P1_C = cv2.imread('P1.png')
    P2_C = cv2.imread('P2.png')

    P1_CG = cv2.cvtColor(P1_C, cv2.COLOR_BGR2GRAY)
    P2_CG = cv2.cvtColor(P2_C, cv2.COLOR_BGR2GRAY)

    p0 = cv2.goodFeaturesToTrack(P1_CG, mask=None, **feature_params)
    mask = np.zeros_like(P1)
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(P1_CG, P2_CG, p0, None, **lk_params)
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(P2, (a, b), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)
    cv2.imshow('frame', img)

    src_pts = good_old
    dst_pts = good_new

    #%% Warping
    print("Warping")
    tform = PiecewiseAffineTransform()
    e1 = tform.estimate(dst_pts, src_pts)
    im_warp1 = warp(P1, tform)
    im_warp1 = to_gray((im_warp1 * 255).astype(np.uint8))

    tform = PiecewiseAffineTransform()
    e2 = tform.estimate(src_pts, dst_pts)
    im_warp2 = warp(P2, tform)
    im_warp2 = to_gray((im_warp2 * 255).astype(np.uint8))

    print("Cross-Dissolve")
    im_gen = (im_warp1 * 0.5) + (im_warp2 * 0.5)
    plt.figure(7)
    plt.suptitle("SIFT Generated")
    plt.imshow(im_gen, cmap='gray')


#%% Stack

S1 = np.array(Image.open('face.jpg'))
S2 = S1
plt.figure(8)
plt.suptitle("Intermediate PM2")
plt.imshow(PM2)

S1_pts = [[0, 0], [S1.shape[0], 0], [0, S1.shape[1]], [S1.shape[0], S1.shape[1]]]
S2_pts = [[0, 0], [S2.shape[0], 0], [0, S2.shape[1]], [S2.shape[0], S2.shape[1]]]

HS1 = ((S1 * 0.8) + (S2 * 0.2)).astype(np.uint8)
HS2 = ((S1 * 0.2) + (S2 * 0.8)).astype(np.uint8)

plt.figure(1)
plt.imshow(np.hstack((HS1, HS2)))
while True:
    plt.figure(1)
    plt.suptitle('Select point on left plate or press enter to generate warp with current points')
    p1 = plt.ginput(n=1, timeout=0)

    if len(p1) == 0:
        plt.suptitle("Generating warp with current points...")

        tform = PiecewiseAffineTransform()
        e1 = tform.estimate(np.array(S2_pts), np.array(S1_pts))
        im_warp1 = warp(S1, tform)
        im_warp1 = to_gray((im_warp1 * 255).astype(np.uint8))

        tform = PiecewiseAffineTransform()
        e2 = tform.estimate(np.array(S1_pts), np.array(S2_pts))
        im_warp2 = warp(S2, tform)
        im_warp2 = to_gray((im_warp2 * 255).astype(np.uint8))

        # plt.figure(2)
        # plt.suptitle("Warps")
        # plt.imshow(np.hstack((im_warp1, im_warp2)), cmap='gray')

        plt.figure(3)
        plt.suptitle("Cross-dissolve")
        im_gen = (im_warp1 * 0.5) + (im_warp2 * 0.5)
        plt.imshow(im_gen, cmap='gray')

        continue

    c = np.random.uniform(0, 1, 3)
    (x1, y1) = (p1[0][0], p1[0][1])
    l1 = plt.plot(x1, y1, marker='x', markersize=15, color=c)
    plt.suptitle('Select point on right plate')

    p2 = plt.ginput(n=1, timeout=0)
    if len(p2) == 0:
        l1.pop(0).remove()
        plt.suptitle("Breaking out of infinite loop")
        break

    # Translate
    (x2, y2) = (p2[0][0], p2[0][1])

    # If you click on the right side
    if x2 > S1.shape[0]:
        x2 = x2 - S1.shape[0]
        print("Right side click")
    else:
        print("Left side click")

    x2 = (x2 + x1) / 2
    y2 = (y2 + y1) / 2
    plt.plot(x2, y2, marker='x', markersize=15, color=c)

    S1_pts.append([x1, y1])
    S2_pts.append([x2, y2])
    print("Total points so far: ", len(S1_pts))
    # plt.figure(2)
    # plt.imshow(S1)
    # plt.plot(x1, y1, marker='x', markersize=5, color='red')
    #
    # plt.figure(3)
    # plt.imshow(S2)
    # plt.plot(x2, y2, marker='x', markersize=5, color='red')


#%% Z-Plane using built-in warping


#%%
# print("Blending")
# plt.figure()
# im_b1 = im_warp1
# im_b2 = im_warp2
# for w in np.linspace(0, 1, 20):
#     b1 = im_b1 * w
#     b2 = im_b2 * (1-w)
#     gen = b1 + b2
#     plt.suptitle("Im1 Weight " + str(w))
#     plt.imshow(gen, cmap='gray')
#     plt.waitforbuttonpress()