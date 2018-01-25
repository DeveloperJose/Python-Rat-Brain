# Author: Jose G Perez <josegperez@mail.com>
import cv2
import numpy as np
import config
import feature
from multiprocessing.pool import ThreadPool
from timeit import default_timer as timer
# Load datasets
s_data = np.load('atlas_sw.npz')
s_im = s_data['images']
s_label = s_data['labels']

pw_data = np.load('atlas_pw.npz')
pw_im = pw_data['images']
pw_label = pw_data['labels']

# Prepare SM
similarity_matrix = np.zeros((len(s_im), len(pw_im)))
print("Similarity Matrix Shape", similarity_matrix.shape)

# Precompute SIFT descriptors for atlas
kp = []
des = []
for j in range(len(pw_im)):
	im2 = pw_im[j]
	kp2, des2 = feature.extract_sift(im2)
	kp.append(kp2)
	des.append(des2)

# Begin matching
start_time = timer()
for i in range(similarity_matrix.shape[0]):
	print("Matching S", s_label[i])
	im1 = s_im[i]
	kp1, des1 = feature.extract_sift(im1)
	
	for j in range(similarity_matrix.shape[1]):
		im2 = pw_im[j]
		kp2 = kp[j]
		des2 = des[j]
		
		matches = feature.match_fast(kp1, des1, kp2, des2)
		similarity_matrix[i,j] = len(matches)
			
duration = timer() - start_time
print("Creating matrix took %.3fs" % duration)

# Output result
np.set_printoptions(threshold=np.nan, linewidth=1000)
print("Similarity: SIFT")
print(similarity_matrix)

# Save result
np.savez_compressed('sm', sm=similarity_matrix)