# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pylab as plt
from PIL import Image, ImageOps

pil_im = Image.open('dataset\Complete-Atlas-Level-01')
img_crop = ImageOps.crop(pil_im, 180)
w, h = img_crop.size
#atlas = img_crop.crop(((w/2), 0, w - (w/4), h - (h/4)))
#both = img_crop.crop(((w/4), 0, (w/2), h - (h/4)))
#hist = img_crop.crop(((w/4), 0, (w/2), h - (h/4)))

img = np.array(img_crop)
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)

img=cv2.drawKeypoints(gray,kp, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.imshow(img)


sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray,None)