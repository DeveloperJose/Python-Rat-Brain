# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
from util import get_hist, get_map, get_hist_gray, get_map_gray
from lib_segment import segment_label, segment_ncut, segment_ncut2

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    img1 = img1[:,:,:3]
    img2 = img2[:,:,:3]

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    
    #out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')    
    out = np.hstack([img1, img2])
    
    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:
        c = np.random.randint(10,256,3)

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, c, 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, c, 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), c, 1)

    # Also return the image if you'd like a copy
    return out

flatten = lambda l: [item for sublist in l for item in sublist]

# For color
#pil_im = Image.open('dataset\Complete-Atlas-Level-01')
#img_crop = ImageOps.crop(pil_im, 180)
#w, h = img_crop.size
#atlas_c = np.array(img_crop.crop(((w/2), 0, w - (w/4), h - (h/4))))[:,:,:3]
#hist_c = np.array(img_crop.crop(((w/4), 0, (w/2), h - (h/4))))[:,:,:3]
#both = np.array(img_crop.crop(((w/4), 0, (w/2), h - (h/4))))[:,:,:3]

plt.gray()

#pil_im = Image.open('dataset\Map-Only-Atlas-Level-01').convert("L")
#img_crop = ImageOps.crop(pil_im, 180)
#w, h = img_crop.size
#atlas = np.array(img_crop.crop(((w/2), 0, w - (w/4), h - (h/4))))
#hist = np.array(img_crop.crop(((w/4), 0, (w/2), h - (h/4))))
#both = np.array(img_crop.crop(((w/4), 0, (w/2), h - (h/4))))

algorithm = "SIFT"

#img1 = get_hist(1) # queryImage
#img1_g = get_hist_gray(1)

#from scipy.ndimage.filters import gaussian_filter
#img1_g = gaussian_filter(img1_g, 1)

#img2 = get_map(1)  # trainImage
#img2_g = get_map_gray(1)
#img2_g = gaussian_filter(img2_g, 1)

nissl = Image.open("./nissl/32.png")

img1 = np.array(nissl)
img1_g = np.array(nissl.convert("L"))

img2 = np.array(nissl)[:, 100:200]
img2_g = np.array(nissl.convert("L"))[:, 100:200]
cut_color = np.array(nissl)[:, 100:200, :]


if algorithm == "ORB":
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1_g,None)
    kp2, des2 = orb.detectAndCompute(img2_g,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    
    output = drawMatches(img1, kp1, img2, kp2, matches[:15])
else:
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1_g,None)
    kp2, des2 = sift.detectAndCompute(img2_g,None)
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.5*n.distance:
           # Removed the brackets around m 
           good.append(m)

            
    output = drawMatches(img1, kp1, img2, kp2, good)
    
# Show the image
plt.figure(0, figsize=(15, 15))
plt.imshow(output)

plt.figure(1)
plt.imshow(cut_color)