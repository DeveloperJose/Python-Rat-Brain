# -*- coding: utf-8 -*-
import numpy as np
import cv2
import config

# ****************************** SIFT Parameters
SIFT = cv2.xfeatures2d.SIFT_create(nfeatures=config.SIFT_FEATURES, nOctaveLayers=config.SIFT_OCTAVE_LAYERS, contrastThreshold=config.SIFT_CONTRAST_THRESHOLD, edgeThreshold=config.SIFT_EDGE_THRESHOLD, sigma=config.SIFT_SIGMA)

def extract_sift(im):
    # Convert floating point images
    if im.dtype != np.uint8:
            im = (im * 255).astype(np.uint8)

    # Convert image to grayscale
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    return SIFT.detectAndCompute(im, None)


# https://isotope11.com/blog/storing-surf-sift-orb-keypoints-using-opencv-in-python
def pickle_sift(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        if i >= len(descriptors):
            break
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id, descriptors[i])
        i += 1
        temp_array.append(temp)
    return temp_array

def unpickle_sift(array):
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)
