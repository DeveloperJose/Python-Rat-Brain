# -*- coding: utf-8 -*-
"""
Course: CS 4365/5354 [Computer Vision]
Author: Jose Perez [ID: 80473954]
Assignment: Lab 5
Instructor: Olac Fuentes
"""
import numpy as np

def segment_label(im):
    import skimage.filters as filters
    from skimage.segmentation import clear_border
    from skimage.measure import label
    import skimage.morphology as morphology
    # Apply threshold
    t = filters.threshold_adaptive(im, block_size=5, offset=8, method="gaussian", param=10)
    
    bw = morphology.opening(t, morphology.diamond(4))

    # remove artifacts connected to image border
    cleared = bw.copy()
    clear_border(cleared)

    # label image regions
    label_image, num = label(cleared, return_num=True)
    borders = np.logical_xor(bw, cleared)
    label_image[borders] = -1

    # Clean image
    sizes = np.bincount(label_image.ravel())
    mask_sizes = sizes > 20
    mask_sizes[0] = 0
    cleaned = mask_sizes[label_image]

    return cleaned, label_image, num

def segment_ncut2(im):
    from skimage import data, segmentation, color, filters, io
    from skimage.future import graph
    
    #gimg = color.rgb2gray(im)
    
    labels1 = segmentation.slic(im, compactness=30, n_segments=10)
    #out1 = color.label2rgb(labels1, im, kind='avg')
    
    g = graph.rag_mean_color(im, labels1, mode='similarity')
    #edges = filters.sobel(gimg)
    #edges_rgb = color.gray2rgb(edges)

    #g = graph.rag_boundary(labels1, edges)
    labels2 = graph.cut_normalized(labels1, g)
    out2 = color.label2rgb(labels2, im, kind='avg')
    
    return out2, labels1, labels2, g
    
# Size of images for eigenvector calculation
WID = 50
def segment_ncut(im, k):
    from lib_ncut import ncut_graph_matrix, cluster
    from scipy.misc import imresize
    global WID
    
    (w, h) = im.shape[:2]
    
    # Resize image to (wid,wid) for fast eigenvector calculation
    rim = imresize(im,(WID,WID),interp='bilinear')
    rim = np.array(rim,'f')
    
    # Create normalized cut matrix
    A = ncut_graph_matrix(rim,sigma_d=1,sigma_g=.0004)
    
    # Cluster
    code, V = cluster(A,k,ndim=3)
    
    # Reshape to original image size
    codeim = imresize(code.reshape(WID,WID),(w,h),interp='nearest')

    return codeim
    
#-----===== For testing segment_label
#-----===============================
#im = np.array(Image.open("Hist-Level-01.jpg"))
#im_g = np.array(Image.open("Hist-Level-01.jpg").convert("L"))    
#label_image = segment_label(im)
#image_label_overlay = label2rgb(label_image, image=im)    
#
## Plot Settings
#fig = plt.figure(5)
#fig.suptitle("Filename", fontsize=20)
#
## Subplot Settings
#nrows = 1
#ncols = 3
#font_size = 12
#
## First Image
#subplot = fig.add_subplot(nrows,ncols,1)
#subplot.set_title("Original", fontsize=font_size)
#subplot.imshow(im)
#subplot.set_xticks(())
#subplot.set_yticks(())
#
## Second Image
#subplot = fig.add_subplot(nrows,ncols,2)
#subplot.set_title("Processed", fontsize=font_size)
#subplot.imshow(t)
#subplot.set_xticks(())
#subplot.set_yticks(())
#
## Third Image
#subplot = fig.add_subplot(nrows,ncols,3)
#subplot.set_title("W/Labels (" + str(num) + ")", fontsize=font_size)
#subplot.imshow(image_label_overlay)
#subplot.set_xticks(())
#subplot.set_yticks(())