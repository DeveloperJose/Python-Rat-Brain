import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, InputLayer, GaussianNoise
from keras.wrappers.scikit_learn import KerasRegressor

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


