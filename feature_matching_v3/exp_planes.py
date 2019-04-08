import numpy as np
import os
import pylab as plt
import cv2
from skimage.transform import warp, PiecewiseAffineTransform
from PIL import Image
from timeit import default_timer as timer

def load(filename):
    WIDTH = 800
    HEIGHT = 400
    im = Image.open(filename)
    im = im.resize((WIDTH, HEIGHT), Image.LANCZOS)
    P = np.array(im, dtype=np.uint8)
    return P[:, 0:P.shape[1] // 2, :]

def to_gray(P):
    return P.mean(axis=2)


#%% Load images
print("Loading images")
dir1 = 'C:/Users/xeroj/Dropbox/Training data sets - Khan-Fuentes/Paxinos and Watson, 2014 (7th Edition) Image set/'
dir2 = 'C:/Users/xeroj/Downloads/Processed'

P1 = load(os.path.join(dir1, 'RBSC7-068.jpg'))
P2 = load(os.path.join(dir1, 'RBSC7-070.jpg'))
FACE = np.array(Image.open('face.jpg'))

PM1 = load(os.path.join(dir2, '18-016 LHA s4t2.tif'))
PM2 = load(os.path.join(dir2, '18-016 LHA s4t3.tif'))
PM3 = load(os.path.join(dir2, '18-016 LHA s4t4.tif'))

# data = np.load('P1P2.npz')
# S1_pts = data['src']
# S2_pts = data['dst']

data = np.load('face.npz')
S1_pts = data['src']
S2_pts = data['dst']
S1 = FACE
S2 = FACE

#%% Scipy Warping
tform = PiecewiseAffineTransform()
e1 = tform.estimate(np.array(S2_pts), np.array(S1_pts))
im_warp1 = warp(S1, tform)
im_warp1 = to_gray((im_warp1 * 255).astype(np.uint8))

plt.figure()
plt.suptitle('warp')
plt.imshow(im_warp1)
#%% Inverse Warping with KNN Interpolation
src = S1
dst = np.zeros_like(src)
for x in range(dst.shape[0]):
    for y in range(dst.shape[1]):
        cp_idx = np.argmin(np.sum((S1_pts - [x,y]) ** 2, axis=1))
        diff = [x,y] - S2_pts[cp_idx]
        disp = S1_pts[cp_idx] - S2_pts[cp_idx]
        dist = np.linalg.norm(diff)
        w = np.exp(-dist/100)
        (dx,dy) = (disp[0]*w, disp[1]*w)
        u = min(int(round(x+dy)),dst.shape[0]-1)
        v = min(int(round(y+dx)),dst.shape[1]-1)
        dst[x,y] = src[u,v]

plt.figure()
plt.imshow(dst)