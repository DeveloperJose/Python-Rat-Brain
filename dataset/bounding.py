# Author: Jose G Perez <josegperez@mail.com
from PIL import Image
import pylab as plt
import numpy as np
import os

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

def gallery(array, ncols=3):
    # Grayscale
    if len(array.shape) == 3:
        nindex, height, width = array.shape
        nrows = nindex//ncols
        result = (array.reshape(nrows, ncols, height, width)
                  .swapaxes(1,2)
                  .reshape(height*nrows, width*ncols))
        return result
    # Color
    else:
        nindex, height, width, intensity = array.shape
        nrows = nindex//ncols
        # want result.shape = (height*nrows, width*ncols, intensity)
        result = (array.reshape(nrows, ncols, height, width, intensity)
                  .swapaxes(1,2)
                  .reshape(height*nrows, width*ncols, intensity))
        return result

def process_plate(filename, split=False):
    im = Image.open(filename).convert("L")

    # Split in half if required
    if split:
        box = (0, 0, im.width / 2, im.height)
        im = im.crop(box)

    im = im.resize((WIDTH, HEIGHT), Image.LANCZOS)
    im = np.array(im, dtype=np.uint8)

    # Convert values very close to white to white for cropping
    im[im >= WHITE_THRESHOLD] = 255

    # Bounding box cropping
    # https://stackoverflow.com/questions/9396312/use-python-pil-or-similar-to-shrink-whitespace
    idx = np.where(im - 255)
    box = list(map(min, idx))[::-1] + list(map(max, idx))[::-1]
    region = Image.fromarray(im).crop(box)
    region = region.resize((WIDTH, HEIGHT), Image.LANCZOS)
    im_cropped = np.array(region, dtype=np.uint8)

    return im, im_cropped


WIDTH = 240
HEIGHT = 300
WHITE_THRESHOLD = 235

s_im = np.empty((73, HEIGHT, WIDTH), dtype=np.uint8)
s_original = np.empty((73, HEIGHT, WIDTH), dtype=np.uint8)
s_label = np.empty(73, dtype=np.uint8)
for plate in range(1, 73+1): # S Plates [01, ..., 73]
    filename = 'Level-' + str(plate).zfill(2) + '.jpg'
    filename = os.path.join('atlas_s', filename)

    if not os.path.exists(filename):
        print("Couldn't find ", filename, ", skipping")
        continue

    im, im_cropped = process_plate(filename)

    # Account for arrays starting at 0
    index = plate-1

    s_original[index] = im
    s_im[index] = im_cropped
    s_label[index] = plate

np.savez_compressed('atlas_s_cropped', images=s_im, labels=s_label, originals=s_original)

# 89 of 161 use Nissl
pw_im = np.empty((89, HEIGHT, WIDTH), dtype=np.uint8)
pw_original = np.empty((89, HEIGHT, WIDTH), dtype=np.uint8)
pw_label = np.empty(89, dtype=np.uint8)
index = 0
for plate in range(0, 161):
    filename = 'RBSC7-' + str(plate+1).zfill(3) + '.jpg'
    filename = os.path.join('atlas_pw', filename)

    if not os.path.exists(filename):
         print("Couldn't find ", filename, ", skipping")
         continue

    im, im_cropped = process_plate(filename, True)

    pw_original[index] = im
    pw_im[index] = im_cropped
    pw_label[index] = plate

    index += 1

np.savez_compressed('atlas_pw_cropped', images=pw_im, labels=pw_label, originals=pw_original)

plt.gray()

imshow(gallery(s_im[:20], 5), 'S 1-19 Cropped')
imshow(gallery(s_original[:20], 5), 'S 1-19 Original')

imshow(gallery(pw_im[:20], 5), 'PW 1-19 Cropped')
imshow(gallery(pw_original[:20], 5), 'PW 1-19 Original')