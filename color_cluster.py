# -*- coding: utf-8 -*-
"""
Course: CS 4365/5354 [Computer Vision]
Author: Jose Perez [ID: 80473954]
Assignment: Lab 4
Instructor: Olac Fuentes
"""
from timeit import default_timer as timer
from PIL import Image, ImageEnhance
import numpy as np
import pylab as plt
from math import ceil
from sklearn.cluster import KMeans
from scipy.ndimage.filters import maximum_filter

# =============== Variables
filename = "dataset2/Hist-Level-05"
k_min = 3
k_max = 5
k_step = 1

# =============== Constants
DECIMALS = '{:.3f}' # Set the decimals for the timer results
IMAGE_COLUMNS = 3.0 # (Floating pt) Number of images to show per row in results 

# ============================== Load image
im = Image.open(filename + ".jpg")

sharpness = ImageEnhance.Sharpness(im)
im = sharpness.enhance(0.5)

brightness = ImageEnhance.Brightness(im)
im = brightness.enhance(1.2)

color = ImageEnhance.Color(im)
im = color.enhance(1)

contrast = ImageEnhance.Contrast(im)
im = contrast.enhance(1.5)

im = maximum_filter(np.array(im), size=(3,1,1))

im_original = np.array(im, dtype=np.uint8)

# Flatten for k-means
(w, h, c) = im_original.shape
im_flat = np.reshape(im_original, (w * h, c))

# Prepare figure
#figure = plt.figure(0, figsize=(4, 7))
#figure.suptitle(filename + " | k-means" )
nrows = ceil(k_max / k_step / IMAGE_COLUMNS)

start_time = timer()
for k in range(k_min, k_max+k_step, k_step):
    temp = im_flat.copy()
    # Run K-Means
    start_time = timer()
    k_means = KMeans(n_clusters = k).fit(temp)
    #print "K-Means took ", time_format.format(timer() - start_time) , "s for k =", k
    
    start_time = timer()
    for cluster in range(k):
        temp[k_means.labels_ == cluster] = k_means.cluster_centers_[cluster]
    
    new_im = np.reshape(temp, (w, h, c)).astype(np.uint8)
    plt.figure(k + 1, figsize=(8, 8))
    plt.imshow(new_im)
    
    Image.fromarray(new_im).save(filename + "-k" + str(k) + "-new.jpg")
    
    
    #print "Replacing took ", time_format.format(timer() - start_time), "s"

    #subplot = figure.add_subplot(nrows+1, IMAGE_COLUMNS, k / k_step)
    #subplot.set_xticks(())
    #subplot.set_yticks(())
    #subplot.set_xlabel("k="+str(k))
    #subplot.imshow(np.reshape(temp, (w, h, c)).astype(np.uint8))
    
plt.show()

print "Took ", DECIMALS.format(timer() - start_time) , "s"