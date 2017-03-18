# -*- coding: utf-8 -*-
"""
Course: CS 4365/5354 [Computer Vision]
Author: Jose Perez [ID: 80473954]
Assignment: Lab 5
Instructor: Olac Fuentes
"""
from timeit import default_timer as timer
from PIL import Image
from lib_segment import segment_label, segment_ncut, segment_ncut2

import numpy as np
import pylab as plt
import os

# Number of images
IMAGES = 2

# Image folder
DIR = "dataset3"

# Prefix for histological cut images
PREFIX = "Hist-Level-"

# Prefix for atlas images
ATLAS_PREFIX = "Complete-Atlas-Level-"

# Image extension
EXT = ".jpg"

# {NCUT, NCUT2 (Experimental), LABEL}
SEGMENTATION = "LABEL"

# Number of clusters for each level starting at level 1
############## 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 
K = np.array([-1, 3, 4, 4, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])

# Set the decimals for the timer results
DECIMALS = '{:.3f}'

NCOLS = 2
NROWS = (IMAGES / NCOLS) * 2
LAST_FIGURE = 1

figure = plt.figure(figsize=(5, 25))
figure.suptitle("Segmentation using " + SEGMENTATION, fontsize=20)

program_start = timer()
for index in range(1, IMAGES):
    start_time = timer()
    
    # ZFill(2) adds a leading zero in front
    filename = PREFIX + str(index).zfill(2) + EXT
    path = os.path.join(DIR, filename)
    #atlas_path = os.path.join(DIR, ATLAS_PREFIX + str(index).zfill(2) + EXT)
    print "Processing: ", path

    # Load images
    im = np.array(Image.open(path))
    im_g = np.array(Image.open(path).convert("L"))
    #im_atlas = np.array(Image.open(atlas_path))        
    
    # Segment
    if SEGMENTATION == "LABEL":
        segment, label, num = segment_label(im_g)
    elif SEGMENTATION == "NCUT2":
        segment, labels1, labels2, g = segment_ncut2(im)
    else:
        temp = im
        segment = segment_ncut(temp, 4)
    
    # Plot the original image
    subplot = figure.add_subplot(NROWS, NCOLS, LAST_FIGURE)
    subplot.set_title(filename)
    subplot.set_xticks(())
    subplot.set_yticks(())
    subplot.imshow(im)
    
    # Plot the segmented image
    subplot = figure.add_subplot(NROWS, NCOLS, LAST_FIGURE+1)
    subplot.set_xticks(())
    subplot.set_yticks(())    
    subplot.imshow(segment)
    
    if num:
        print "Regions: " + str(num)
    
    # Plot the atlas image
    #subplot = figure.add_subplot(1, 3, 3)
    #subplot.set_xticks(())
    #subplot.set_yticks(())    
    #subplot.imshow(im_atlas)
    
    LAST_FIGURE += 2
    
    print "Processing took", DECIMALS.format(timer() - start_time), "s"
    
print "Program ran for ",DECIMALS.format(timer() - program_start), "s"