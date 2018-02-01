# Author: Jose G Perez
# Version 1.0
# Last Modified: January 31, 2018
import numpy as np
import cv2
import os
SIFT = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.05, edgeThreshold=100, sigma=2)

def kp_to_array(kp):
    array = np.zeros((len(kp), 7), dtype=np.float32)
    for idx in range(array.shape[0]):
        k = kp[idx]
        array[idx] = np.array([k.pt[0], k.pt[1], k.size,k.angle,k.response,k.octave,k.class_id])
    return array

def array_to_kp(array):
    kp = []
    for idx in range(array.shape[0]):
        k = array[idx]
        kp.append(cv2.KeyPoint(k[0],k[1],k[2],k[3],k[4],k[5],k[6]))
    return kp

def __precompute_atlas(name):
    if not os.path.isfile(name + '_SIFT.npz'):
        print('Precomputing SIFT for ', name)
        atlas_data = np.load(name + ".npz")
        atlas_im = atlas_data['images']
        atlas_labels = atlas_data['labels']
        atlas_kp = []
        atlas_des = []

        for i in range(0, atlas_im.shape[0]):
            kp, des = SIFT.detectAndCompute(atlas_im[i], None)
            kp = kp_to_array(kp)
            atlas_kp.append(kp)
            atlas_des.append(des)

        atlas_kp = np.asarray(atlas_kp)
        atlas_des = np.asarray(atlas_des)

        np.savez_compressed(name + '_SIFT', images=atlas_im, labels=atlas_labels, kp=atlas_kp, des=atlas_des)

def precompute_sift(S_NAME, PW_NAME):
    __precompute_atlas(S_NAME)
    __precompute_atlas(PW_NAME)

def load_sift(path):
    data = np.load(path)
    return data['images'], data['labels'], data['kp'], data['des']