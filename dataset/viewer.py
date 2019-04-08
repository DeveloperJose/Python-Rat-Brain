import numpy as np
import pylab as plt

s_data = np.load('S_BB_V.npz')
s_im = s_data['images']
s_label = s_data['labels']
s_orig = s_data['originals']

pw_data = np.load('PW_BB_V.npz')
pw_im = pw_data['images']
pw_label = pw_data['labels']
pw_orig = pw_data['originals']

