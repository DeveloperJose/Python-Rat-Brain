# Author: Jose G Perez
# Version: 1.2
# Last Modified: Jan 31st, 2018
from timeit import default_timer as timer
from PIL import Image
import numpy as np
import os

WIDTH = 240
HEIGHT = 300
WHITE_THRESHOLD = 235


def process_plate(filename, split=False):
    im = Image.open(filename).convert("L")

    # Split in half if required
    if split:
        box = (0, 0, im.width // 2, im.height)
        crop = im.crop(box)
        crop.load()
        im = crop

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


def process_atlas(folder, prefix, ext, zfill, plate_min, plate_max, split):
    atlas_im = []
    atlas_label = []
    atlas_original = []
    print('[', end='', flush=True)
    for plate in range(plate_min, plate_max+1):
        filename = prefix + str(plate).zfill(zfill) + ext
        filename = os.path.join(folder, filename)

        print(plate, ',', end='', flush=True)
        if not os.path.exists(filename):
            print("Couldn't find ", filename, ", skipping")
            continue

        im, im_cropped = process_plate(filename, split)

        atlas_im.append(im_cropped)
        atlas_label.append(plate)
        atlas_original.append(im)

    print(']\n', end='', flush=True)
    return np.asarray(atlas_im), np.asarray(atlas_label), np.asarray(atlas_original)


if __name__ == '__main__':
    print("===== Starting timer")
    time_start = timer()

    print("Processing S...")
    s_im, s_label, s_original = process_atlas('atlas_s', 'Level-', '.jpg', 2, 1, 73, False)

    print("Processing PW...")
    pw_im, pw_label, pw_original = process_atlas('atlas_pw', 'RBSC7-', '.jpg', 3, 1, 161, True)

    print("Saving...")
    np.savez_compressed('S_BB_V', images=s_im, labels=s_label, originals=s_original)
    np.savez_compressed('PW_BB_V', images=pw_im, labels=pw_label, originals=pw_original)

    duration = timer() - time_start
    print("Program took %.3fs" % duration)