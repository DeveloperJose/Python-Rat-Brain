# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator

import sys
import config
import logbook
logger = logbook.Logger(__name__)
logbook.StreamHandler(sys.stdout, level=logbook.DEBUG, format_string=config.LOGGER_FORMAT_STRING).push_application()

class RANSACHomographyModel(BaseEstimator):
    def __init__(self):
        pass

    def predict(self, X, y):


    def fit(self, X, y):
        # X = Training Data = src_pts
        # y = Target Values = dst_pts
        print("X:" + str(len(X)))
        print("Y:" + str(len(y)))
        #self.H = H_from_points(X, y)
        self.H_ = None
        return self

    def score(self, X, y):
        return 0

    def _H_from_points(src_pts, dst_pts):
        return None

def is_data_valid(X, y):
    return True

def is_model_valid(model, X, y):
    if model.H_ is None:
        return True

    # Transform corners


    return True

def get_ransac_homography(src_pts, dst_pts):
    # Set-up RANSAC parameters
    RANSAC = RANSACRegressor(base_estimator=RANSACHomographyModel(), min_samples=4, residual_threshold=None, is_data_valid=is_data_valid, is_model_valid=is_model_valid, max_trials=100, stop_n_inliers=np.inf, stop_score=np.inf, stop_probability=0.99, residual_metric=None, loss='absolute_loss', random_state=None)

    # Fit the points using RANSAC
    RANSAC.fit(src_pts, dst_pts)

    # Inliers
    inlier_mask = RANSAC.inlier_mask_

    return RANSAC.estimator_.H, inlier_mask

if __name__ == '__main__':
    import feature
    im_region = feature.im_read('C:/Users/xeroj/Desktop/Local_Programming/Vision-Rat-Brain/scripts_testing/region.jpg')
    feature.match_region_nissl(im_region, nissl_level=34)