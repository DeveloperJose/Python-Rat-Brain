import numpy as np
import os
import pylab as plt
import cv2
from skimage.transform import warp, PiecewiseAffineTransform
from PIL import Image
from timeit import default_timer as timer

def load(filename, split=True):
    WIDTH = 400
    HEIGHT = 400
    im = Image.open(filename)
    im = im.resize((WIDTH, HEIGHT), Image.LANCZOS)
    P = np.array(im, dtype=np.uint8)
    if split:
        return P[:, 0:P.shape[1] // 2, :]
    else:
        return P

def to_gray(P):
    return P.mean(axis=2)


#%% Load images
print("Loading images")
dir1 = r'C:\Users\xeroj\Dropbox\Training data sets - Khan-Fuentes\Paxinos and Watson, 2014 (7th Edition) Image set'
dir2 = r'C:\Users\xeroj\Desktop\Local_Schoolwork\_Research\CytoData - Processed'
dir3 = r'C:\Users\xeroj\Downloads\cne24381-sup-0011-suppinfo11\SI Folder 4 BM4 Nissls 300 dpi'

P1 = load(os.path.join(dir1, 'RBSC7-068.jpg'))
P2 = load(os.path.join(dir1, 'RBSC7-070.jpg'))

SW1 = load(os.path.join(dir3, 'Level 31 photo.tif'), False)
SW2 = load(os.path.join(dir3, 'Level 32 photo.tif'), False)
SW3 = load(os.path.join(dir3, 'Level 33 photo.tif'), False)

PM1 = load(os.path.join(dir2, '18-016 LHA s4t2.tif'))
PM2 = load(os.path.join(dir2, '18-016 LHA s4t3.tif'))
PM3 = load(os.path.join(dir2, '18-016 LHA s4t4.tif'))

#%% Plate selection
S1 = SW1
S2 = SW3


#%% Stack
plt.figure()
plt.gray()
plt.suptitle("Intermediate")
plt.imshow(SW2)

S1_pts = [[0, 0], [S1.shape[0], 0], [0, S1.shape[1]], [S1.shape[0], S1.shape[1]]]
S2_pts = [[0, 0], [S2.shape[0], 0], [0, S2.shape[1]], [S2.shape[0], S2.shape[1]]]

HS1 = ((S1 * 0.8) + (S2 * 0.2)).astype(np.uint8)
HS2 = ((S1 * 0.2) + (S2 * 0.8)).astype(np.uint8)

# plt.figure()
# plt.suptitle("Originals")
# plt.imshow(np.hstack((S1, S2)))

plt.figure()
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
        # im_warp1 = to_gray((im_warp1 * 255).astype(np.uint8))
        im_warp1 = ((im_warp1 * 255).astype(np.uint8))

        tform = PiecewiseAffineTransform()
        e2 = tform.estimate(np.array(S1_pts), np.array(S2_pts))
        im_warp2 = warp(S2, tform)
        # im_warp2 = to_gray((im_warp2 * 255).astype(np.uint8))
        im_warp2 = ((im_warp2 * 255).astype(np.uint8))

        # plt.figure()
        # plt.suptitle("Warps")
        # plt.imshow(np.hstack((im_warp1, im_warp2)), cmap='gray')

        # plt.figure()
        # plt.suptitle("Original Plate (Left), Warped Plate (Right)")
        # plt.imshow(np.hstack((to_gray(S1), im_warp1)), cmap='gray')
        #
        # plt.figure()
        # plt.suptitle("Original Plate (Left), Warped Plate (Right)")
        # plt.imshow(np.hstack((to_gray(S2), im_warp2)), cmap='gray')

        plt.figure()
        plt.suptitle("Cross-dissolve")
        im_gen = (im_warp1 * 0.5) + (im_warp2 * 0.5)
        plt.imshow(im_gen, cmap='gray')

        continue

    c = np.random.uniform(0, 1, 3)
    (x1, y1) = (p1[0][0], p1[0][1])
    l1 = plt.plot(x1, y1, marker='x', markersize=10, markeredgewidth=5, color=c)
    plt.suptitle('Select point on right plate')

    p2 = plt.ginput(n=1, timeout=0)
    if len(p2) == 0:
        l1.pop(0).remove()
        plt.suptitle("Breaking out of infinite loop")
        break

    # Translate
    (x2, y2) = (p2[0][0], p2[0][1])
    xviz = x2
    yviz = y2

    # If you click on the right side
    if x2 > S1.shape[0]:
        x2 = x2 - S1.shape[0]
        print("Right side click")
    else:
        print("Left side click")

    # x2 = (x2 + x1) / 2
    # y2 = (y2 + y1) / 2

    x2 = (x2 + x1) / 2
    y2 = (y2 + y1) / 2

    plt.plot(xviz, yviz, marker='x', markersize=10, markeredgewidth=5, color=c)

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