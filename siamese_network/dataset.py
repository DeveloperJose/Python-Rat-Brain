import numpy as np

def normalize(dataset):
    mean = np.mean(dataset)
    std = np.std(dataset)
    return (dataset - mean) / std

def get_training():
    # X Shape: (Plates, Height, Width)
    sw_data = np.load('atlas_sw.npz')
    x_train = normalize(sw_data['images'])
    y_train = sw_data['labels']
    return x_train, y_train

def get_testing():
    pw_data = np.load('atlas_pw.npz')
    x_test = pw_data['images']
    y_test = pw_data['labels']
    return x_test, y_test


