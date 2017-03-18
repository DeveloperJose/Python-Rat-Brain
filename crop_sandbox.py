# -*- coding: utf-8 -*-
import pylab as plt
from PIL import Image
from PIL import ImageOps

filename = 'dataset\Complete-Atlas-Level-01'
im_pil = Image.open(filename)
w, h = im_pil.size

# Crop
im_crop = ImageOps.crop(im_pil, 150)

plt.imshow(im_crop)