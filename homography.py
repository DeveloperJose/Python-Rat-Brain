# -*- coding: utf-8 -*-
import numpy as np
import cv2

import sys
import config
import logbook
logger = logbook.Logger(__name__)
logbook.StreamHandler(sys.stdout, level=logbook.DEBUG, format_string=config.LOGGER_FORMAT_STRING).push_application()

class RansacHomographyModel(object):
    """ Class for testing homography fit with ransac.py from
        http://www.scipy.org/Cookbook/RANSAC"""

    def __init__(self, src_pts, dst_pts, affine):
        self.src_pts = src_pts
        self.dst_pts = dst_pts
        self.affine = affine

    def fit(self, data_indices):
        """ Fit homography to four selected correspondences. """
        logger.debug("Data Indices {0}", data_indices.shape)
        fp = self.src_pts[data_indices, :]
        tp = self.dst_pts[data_indices, :]

        logger.debug("Fit - Fp {0}, Tp {1}", fp.shape, tp.shape)

        if self.affine:
            return Haffine_from_points(make_homog(fp.T), make_homog(tp.T))
        else:
            return H_from_points(make_homog(fp.T), make_homog(tp.T))

    def is_valid_model(self, model, data_indices):

        return True

    def get_error(self, data_indices, model):
        """ Apply homography to all correspondences,
            return error for each transformed point. """

        fp = make_homog(self.src_pts[data_indices, :].T)
        tp = make_homog(self.dst_pts[data_indices, :].T)

        # transform fp
        fp_transformed = np.dot(model,fp)

        # normalize hom. coordinates
        fp_transformed = normalize(fp_transformed)

        # Point counts are not the same?
        #if tp.shape != fp_transformed.shape:
        #    logger.debug("Tp {0}, Fp_T {1}, Data {2}", tp.shape, fp_transformed.shape, data.shape)
        #    return None

        # return error per point
        diff = tp-fp_transformed
        err_sum = np.sum(diff, axis=0)**2
        return np.sqrt(err_sum)

def H_from_ransac(fp,tp,model,maxiter=1000,match_theshold=10):
    """ Robust estimation of homography H from point
        correspondences using RANSAC (ransac.py from
        http://www.scipy.org/Cookbook/RANSAC).

        input: fp,tp (3*n arrays) points in hom. coordinates. """

    import ransac

    # group corresponding points
    #data = np.vstack((fp,tp))
    data = fp

    # compute H and return
    H, ransac_data = ransac.ransac(data,model,4,maxiter,match_theshold,10,return_all=True)
    return H,ransac_data['inliers']


def H_from_points(fp,tp):
    """ Find homography H, such that fp is mapped to tp
        using the linear DLT method. Points are conditioned
        automatically. """

    if fp.shape != tp.shape:
        logger.debug("Cannot calculate homography due to size")
        return None

    # condition points (important for numerical reasons)
    # --from points--
    m = np.mean(fp[:2], axis=1)
    maxstd = np.max(np.std(fp[:2], axis=1)) + 1e-9
    C1 = np.diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp = np.dot(C1,fp)

    # --to points--
    m = np.mean(tp[:2], axis=1)
    maxstd = np.max(np.std(tp[:2], axis=1)) + 1e-9
    C2 = np.diag([1/maxstd, 1/maxstd, 1])
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp = np.dot(C2,tp)

    # create matrix for linear method, 2 rows for each correspondence pair
    nbr_correspondences = fp.shape[1]
    A = np.zeros((2*nbr_correspondences,9))
    for i in range(nbr_correspondences):
        A[2*i] = [-fp[0][i],-fp[1][i],-1,0,0,0,
                    tp[0][i]*fp[0][i],tp[0][i]*fp[1][i],tp[0][i]]
        A[2*i+1] = [0,0,0,-fp[0][i],-fp[1][i],-1,
                    tp[1][i]*fp[0][i],tp[1][i]*fp[1][i],tp[1][i]]

    U,S,V = np.linalg.svd(A)
    H = V[8].reshape((3,3))

    # decondition
    H = np.dot(np.linalg.inv(C2),np.dot(H,C1))

    # normalize and return
    return H / H[2,2]


def Haffine_from_points(fp,tp):
    """ Find H, affine transformation, such that
        tp is affine transf of fp. """

    if fp.shape != tp.shape:
        return None

    # condition points
    # --from points--
    m = np.mean(fp[:2], axis=1)
    maxstd = np.max(np.std(fp[:2], axis=1)) + 1e-9
    C1 = np.diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp_cond = np.dot(C1,fp)

    # --to points--
    m = np.mean(tp[:2], axis=1)
    C2 = C1.copy() #must use same scaling for both point sets
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp_cond = np.dot(C2,tp)

    # conditioned points have mean zero, so translation is zero
    A = np.concatenate((fp_cond[:2],tp_cond[:2]), axis=0)
    U,S,V = np.linalg.svd(A.T)

    # create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]

    tmp2 = np.concatenate((np.dot(C,np.linalg.pinv(B)),np.zeros((2,1))), axis=1)
    H = np.vstack((tmp2,[0,0,1]))

    # decondition
    H = np.dot(np.linalg.inv(C2),np.dot(H,C1))

    return H / H[2,2]


def normalize(points):
    """ Normalize a collection of points in
        homogeneous coordinates so that last row = 1. """

    for row in points:
        row /= points[-1]
    return points


def make_homog(points):
    """ Convert a set of points (dim*n array) to
        homogeneous coordinates. """

    return np.vstack((points,np.ones((1,points.shape[1]))))


if __name__ == '__main__':
    import feature
    im_region = feature.im_read('C:/Users/xeroj/Desktop/Local_Programming/Vision-Rat-Brain/scripts_testing/region.jpg')
    feature.match_region_nissl(im_region, nissl_level=34)