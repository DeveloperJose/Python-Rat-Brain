import numpy as np
import os
from PIL import Image
import pylab as plt
import cv2

np.set_printoptions(threshold=np.nan)
plt.gray()

#%% Generation function
def to_gray(P):
    return P.mean(axis=2)

def generate_color(P1, P2, rstrides, cstrides, threshold):
    P3 = np.zeros_like(P1)
    rows = P1.shape[0] // rstrides
    cols = P1.shape[1] // cstrides

    for ridx in range(rows):
        rstart = ridx * rstrides
        if ridx == rows - 1:
            rend = P1.shape[0]
        else:
            rend = rstart + rstrides

        for cidx in range(cols):
            cstart = cidx * cstrides
            if cidx == cols - 1:
                cend = P1.shape[1]
            else:
                cend = cstart + cstrides

            # Extract plate quadrants
            q1 = P1[rstart:rend, cstart:cend, :]
            q1 = q1.reshape(q1.shape[0] * q1.shape[1], 3)

            q2 = P2[rstart:rend, cstart:cend, :]
            q2 = q2.reshape(q2.shape[0] * q2.shape[1], 3)

            # Calculate quadrant coverage
            px_total = q1.shape[0]
            z1 = np.sum(q1 < threshold)
            z2 = np.sum(q2 < threshold)
            coverage1 = z1 / px_total
            coverage2 = z2 / px_total
            coverage3 = (coverage1 + coverage2) / 2

            # Create blank quadrant
            q3 = np.zeros_like(q1)
            q3[:, :] = 255

            # Figure out how much we need to cover it
            q3_pixels = int(px_total * coverage3)

            # Randomly select pixels to paint
            idxs = np.random.choice(q3.shape[0], q3_pixels, True)

            # Paint by averaging the two image quadrants
            q3[idxs, :] = (q1[idxs, :] + q2[idxs, :]) / 2
            # q3[idxs, :] = 0

            # Reshape back into original form, add to output image
            q3 = q3.reshape(rend-rstart, cend-cstart, 3)
            P3[rstart:rend, cstart:cend, :] = q3

    return P3


def generate_gray(P1, P2, rstrides, cstrides, threshold, w1=0.5):
    P3 = np.zeros_like(P1)
    rows = P1.shape[0] // rstrides
    cols = P1.shape[1] // cstrides

    for ridx in range(rows):
        rstart = ridx * rstrides
        if ridx == rows - 1:
            rend = P1.shape[0]
        else:
            rend = rstart + rstrides

        for cidx in range(cols):
            cstart = cidx * cstrides
            if cidx == cols - 1:
                cend = P1.shape[1]
            else:
                cend = cstart + cstrides

            # Extract plate quadrants
            q1 = P1[rstart:rend, cstart:cend]
            q1 = q1.reshape(q1.shape[0] * q1.shape[1])

            q2 = P2[rstart:rend, cstart:cend]
            q2 = q2.reshape(q2.shape[0] * q2.shape[1])

            # Calculate quadrant coverage
            px_total = q1.shape[0]
            z1 = np.sum(q1 < threshold)
            z2 = np.sum(q2 < threshold)
            coverage1 = z1 / px_total
            coverage2 = z2 / px_total
            coverage3 = (coverage1 + coverage2) / 2

            # Create blank quadrant
            q3 = np.zeros_like(q1)
            q3[:] = 255

            # Figure out how much we need to cover it
            q3_pixels = int(px_total * coverage3)

            # Randomly select pixels to paint
            idxs = np.random.choice(q3.shape[0], q3_pixels, True)

            # Paint by averaging the two image quadrants
            w2 = 1 - w1
            # q3[idxs] = (w1 * q1[idxs] + w2 * q2[idxs]) / 2
            q3[idxs] = 0

            # Reshape back into original form, add to output image
            q3 = q3.reshape(rend-rstart, cend-cstart)
            P3[rstart:rend, cstart:cend] = q3

    return P3
#%% Load Image Files
print("Loading images")
WIDTH = 800
HEIGHT = 400
dir = 'C:/Users/xeroj/Dropbox/Training data sets - Khan-Fuentes/Paxinos and Watson, 2014 (7th Edition) Image set/'

filename1 = os.path.join(dir, 'RBSC7-068.jpg')
filename2 = os.path.join(dir, 'RBSC7-070.jpg')

im1 = Image.open(filename1)
im1 = im1.resize((WIDTH, HEIGHT), Image.LANCZOS)
P1 = np.array(im1, dtype=np.uint8)
P1 = P1[:, 0:P1.shape[1]//2, :]

im2 = Image.open(filename2)
im2 = im2.resize((WIDTH, HEIGHT), Image.LANCZOS)
P2 = np.array(im2, dtype=np.uint8)
P2 = P2[:, 0:P2.shape[1]//2, :]

P1_g = to_gray(P1)
P2_g = to_gray(P2)

#%% Control points
print("Computing control points")
LEFT_PAD = 0
RIGHT_PAD = 0
COL_INTERVALS = 5
CTRL_THRESHOLD = 200
def get_controls(P):
    w = P.shape[0]
    h = P.shape[1]
    ctrl_pts = [[0, 0], [w, 0], [0, h], [w, h]]
    # Top to bottom
    for col in range(LEFT_PAD, P.shape[1]-RIGHT_PAD, COL_INTERVALS):
        for row in range(0, P.shape[0], 1):
            if P[row,col] <= CTRL_THRESHOLD:
                ctrl_pts.append([row, col])
                break

    # Bottom to top
    for col in range(LEFT_PAD, P.shape[1]-RIGHT_PAD, COL_INTERVALS):
        for row in range(P.shape[0]-1, 0, -1):
            if P[row,col] <= CTRL_THRESHOLD:
                ctrl_pts.append([row, col])
                break


    return ctrl_pts

src_pts = get_controls(P1_g)
dst_pts = get_controls(P2_g)

size = min(len(src_pts), len(dst_pts))
src_pts = src_pts[:size]
dst_pts = dst_pts[:size]

#%% Visualize Control Points
print("Visualizing control points")
fig, ax = plt.subplots(nrows=1,ncols=2)
title = 'Control Points'
fig.suptitle(title, fontsize=22)

ax[0].set_title('PW68')
ax[0].imshow(P1)

ax[1].set_title('PW72')
ax[1].imshow(P2)

for pt in src_pts:
    ax[0].plot(pt[1], pt[0], marker='x', color='red', markersize=5)

for pt in dst_pts:
    ax[1].plot(pt[1], pt[0], marker='x', color='red', markersize=5)

#%% SIFT computing
print("Computing SIFT features")
# SIFT = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.02, edgeThreshold=100, sigma=2)
SIFT = cv2.xfeatures2d.SIFT_create()
kp1, des1 = SIFT.detectAndCompute(P1, None)
kp2, des2 = SIFT.detectAndCompute(P2, None)

fig, ax = plt.subplots(nrows=1,ncols=2)
title = 'SIFT KeyPoints'
fig.suptitle(title, fontsize=22)

ax[0].set_title('Plate 1')
ax[0].imshow(cv2.drawKeypoints(P1, kp1, None))

ax[1].set_title('Plate 2')
ax[1].imshow(cv2.drawKeypoints(P2, kp2, None))

#%% SIFT + Homography
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1,des2, k=2)
# matches = np.asarray([m for m in matches if m[0].distance < 0.8*m[1].distance])
# if len(matches[:,0]) >= 4:
#     src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
#     dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
#     H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 50.0)
#     im_match = cv2.drawMatches(P1, kp1, P2, kp2, matches[:,0], None, flags=2)
#     plt.figure(1)
#     plt.imshow(im_match)
#     # dst = cv2.warpPerspective(P1,H,(P1.shape[1] + P2.shape[1], P2.shape[0]))

#%% SIFT for warping
# for m in matches[:,0]:
#     src_pts.append(kp1[m.queryIdx].pt)
#     dst_pts.append(kp2[m.trainIdx].pt)
#
# src_pts = np.array(src_pts, dtype=np.float32)
# dst_pts = np.array(dst_pts, dtype=np.float32)

#%% My Method
print("Performing my matching algorithm")
import util_matching, util_cv, util_sift
matches2 = util_matching.match(util_sift.kp_to_array(kp1), des1, util_sift.kp_to_array(kp2), des2)
matches2 = util_cv.match_to_cv(matches2)
im_match2 = cv2.drawMatches(P1, kp1, P1, kp2, matches2, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(im_match2)

#%% Mine for warping
print("Adding matching points for warping")
for m in matches2:
    pt1 = kp1[m.queryIdx].pt
    pt2 = kp2[m.trainIdx].pt
    src_pts.append([pt1[1], pt1[0]])
    dst_pts.append([pt2[1], pt2[0]])

src_pts = np.array(src_pts, dtype=np.float32)
dst_pts = np.array(dst_pts, dtype=np.float32)

#%% Warping
print("Warping")
from skimage.transform import warp, PiecewiseAffineTransform
tform = PiecewiseAffineTransform()
tform.estimate(src_pts, dst_pts)
im_warp1 = warp(P1_g, tform)

tform = PiecewiseAffineTransform()
tform.estimate(dst_pts, src_pts)
im_warp2 = warp(P2_g, tform)
#
# pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
# pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
# M = cv2.getPerspectiveTransform(src_pts, dst_pts)
# wrp = cv2.warpAffine(P1.astype(np.int32), M, dsize=(P1.shape[0], P1.shape[1]), flags=cv2.INTER_LINEAR)

#%% Warp Visualization
# fig, ax = plt.subplots(nrows=3,ncols=2)
# title = 'Warping'
# fig.suptitle(title, fontsize=22)
#
# ax[0,0].set_title('PW68')
# ax[0,0].imshow(cv2.drawKeypoints(P1, kp1, None))
#
# ax[0,1].set_title('PW72')
# ax[0,1].imshow(cv2.drawKeypoints(P2, kp2, None))
#
# ax[1,0].set_title('Match (SIFT + RANSAC)')
# ax[1,0].imshow(None)
#
# ax[1,1].set_title("Match (Mine)")
# ax[1,1].imshow(im_match2)
#
# ax[2,0].set_title('WARP1')
# ax[2,0].imshow(im_warp1)
#
# ax[2,1].set_title('WARP2')
# ax[2,1].imshow(im_warp2)

#%% Calculation of Baseline & Figure
print("Generating P3 baseline")
rstrides = 2
cstrides = 2
threshold = 200
im_src = im_warp1
im_dst = im_warp2
P3 = generate_gray(im_src, im_dst, rstrides, cstrides, threshold, 0.5)
plt.figure()
plt.imshow(P3)

#%% P3 Visualization
print("Visualizing P3 results")
fig, ax = plt.subplots(nrows=1,ncols=3)
title = 'Intermediate Plate Generation Baseline'
fig.suptitle(title, fontsize=22)

ax[0].set_title('Source')
ax[0].imshow(im_src)

ax[1].set_title('Destination')
ax[1].imshow(im_dst)

ax[2].set_title(str(rstrides) + ' by ' + str(cstrides))
ax[2].imshow(P3)

#%%
print("Blending")
plt.figure()
im_b1 = im_warp1
im_b2 = im_warp2
# im_b1 = to_gray((im_warp1 * 255).astype(np.uint8))
# im_b2 = to_gray((im_warp2 * 255).astype(np.uint8))
for w in np.linspace(0, 1, 10):
    b1 = im_b1 * w
    b2 = im_b2 * (1-w)
    gen = b1 + b2
    plt.suptitle("Im1 Weight " + str(w))
    plt.imshow(gen)
    plt.waitforbuttonpress()

#%% Optical Flow
print("Optical Flow")
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.03))

P1_C = cv2.imread('P1.png')
P2_C = cv2.imread('P2.png')

P1_CG = cv2.cvtColor(P1_C, cv2.COLOR_BGR2GRAY)
P2_CG = cv2.cvtColor(P2_C, cv2.COLOR_BGR2GRAY)

p0 = cv2.goodFeaturesToTrack(P1_CG, mask = None, **feature_params)
mask = np.zeros_like(P1)
# Create some random colors
color = np.random.randint(0,255,(100,3))

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



#%% Generate many intermediate baselines and save as files
# t = 150
# for r in range(1, 20, 1):
#     for c in range(1, 20, 1):
#         P = generate(P1, P2, r, c, t)
#         filename = 'baseline/' + str(r) + ' by ' + str(c) + '.png'
#         Image.fromarray(P).save(filename)
