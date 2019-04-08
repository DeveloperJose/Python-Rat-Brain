import pylab as plt
import numpy as np
from PIL import Image
import time

im = np.array(Image.open('face.jpg'), dtype=np.uint8)
(rows, cols, colors) = im.shape

#%% Control point loading
data = np.load('face.npz')
src_pts = data['src']
dst_pts = data['dst']

src_pts[0] = [0,0]
src_pts[1] = [im.shape[1], 0]
src_pts[2] = [0,im.shape[0]]
src_pts[3] = [im.shape[1], im.shape[0]]

dst_pts[0] = [0,0]
dst_pts[1] = [im.shape[1], 0]
dst_pts[2] = [0,im.shape[0]]
dst_pts[3] = [im.shape[1], im.shape[0]]


plt.figure(1)
plt.imshow(im)
for i in range(len(src_pts)):
    (x1,y1) = src_pts[i]
    (x2,y2) = dst_pts[i]
    plt.plot([x1,x2], [y1,y2], color='purple')
    plt.plot(x1, y1, marker='x', markersize=3, color='blue')
    plt.plot(x2, y2, marker='o', markersize=3, color='red')

start = time.time()

#%% Generate pixels coordinates in the destination image
dest_im = np.zeros(im.shape, dtype=np.uint8)
max_row = im.shape[0] - 1
max_col = im.shape[1] - 1
dest_rows = dest_im.shape[0]
dest_cols = dest_im.shape[1]

# Painting outline of source image black, so out of bounds pixels can be painted black
im[0] = 0
im[max_row] = 0
im[:, 0] = 0
im[:, max_col] = 0

# Generate pixel coordinates in the destination image
ind = np.arange(dest_rows * dest_cols)
row_vect = ind // dest_cols
col_vect = ind % dest_cols
coords = np.vstack((row_vect, col_vect))

# Computing pixel weights, pixels close to p[1] will have higher weights
dist = np.sqrt(np.square(p[1][1] - row_vect) + np.square(p[1][0] - col_vect))
weight = np.exp(-dist / 100)  # Constant needs to be tweaked depending on image size

# Computing pixel weights, pixels close to p[1] will have higher weights
source_coords = np.zeros(coords.shape, dtype=np.int)
disp_r = (weight * (p[0][1] - p[1][1])).astype(int)
disp_c = (weight * (p[0][0] - p[1][0])).astype(int)
source_coords[0] = coords[0] + disp_r
source_coords[1] = coords[1] + disp_c

# Fixing out-of-bounds coordinates
source_coords[source_coords < 0] = 0
source_coords[0, source_coords[0] > max_row] = max_row
source_coords[1, source_coords[1] > max_col] = max_col

dest_im = source_im[source_coords[0], source_coords[1], :].reshape(dest_rows, dest_cols, 3)

plt.figure(2)
plt.imshow(dest_im)
plt.show()

elapsed_time = time.time() - start
print('Elapsed time: {0:.2f} '.format(elapsed_time))

