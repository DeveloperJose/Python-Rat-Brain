import pylab as plt
import numpy as np
s_data = np.load('atlas_sw.npz')
s_im = s_data['images']
s_label = s_data['labels']

pw_data = np.load('atlas_pw.npz')
pw_im = pw_data['images']
pw_label = pw_data['labels']
plt.figure(0)
plt.imshow(pw_im[0])
input(0)