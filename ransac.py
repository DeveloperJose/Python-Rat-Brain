import numpy as np
import scipy # use numpy if scipy unavailable
import scipy.linalg # use numpy if scipy unavailable
import cv2
import homography

import config
import sys
import logbook
logger = logbook.Logger(__name__)
logbook.StreamHandler(sys.stdout, level=logbook.DEBUG, format_string=config.LOGGER_FORMAT_STRING).push_application()

def ransac(src_pts, dst_pts, corners, threshold=500,max_iters=1000):
    # Keep track of our best results
    best_homography = None
    best_inliers_count = 0
    best_inliers_mask = None

    count_non_convex = 0
    count_bad_homography = 0

    # Loop for the specified iterations
    # You could also do it using time
    iterations = 0
    while iterations < max_iters:
        # Get all the indices as a mask of booleans
        inliers = np.full(src_pts.shape[0], False, np.bool)

        # Randomly get 4 indices for points
        # No duplicates are allowed
        rand_indices = np.random.choice(src_pts.shape[0], 4, replace=False)

        # They are inliers, naturally
        inliers[rand_indices] = True

        # Get the points
        rand_src = src_pts[inliers]
        rand_dst = dst_pts[inliers]

        # Calculate the homography from the 4 points
        H, mask = cv2.findHomography(rand_src, rand_dst, method=0)

        # Check if the homography is valid
        if H is None or len(H.shape) != 2:
            iterations+=1
            count_bad_homography+=1
            continue

        # Transform the corners of the original image
        try:
            transformedCorners = cv2.perspectiveTransform(corners, H)
        except:
            iterations+=1
            count_non_convex+=1

        # Check the convexity of the transformed points
        isConvex = cv2.isContourConvex(transformedCorners)
        if not isConvex:
            iterations+=1
            count_non_convex+=1
            continue

        # Part 2: Get the inliers and outliers
        # Get the remaining points
        rem_src = src_pts[~inliers]
        rem_dst = dst_pts[~inliers]

        # Get the error for each point
        rem_src_hom = cv2.convertPointsToHomogeneous(rem_src)
        rem_dst_hom = cv2.convertPointsToHomogeneous(rem_dst)

        calc = rem_dst_hom - (H * rem_src_hom)
        error = np.linalg.norm(calc, axis=(1, 2))
        error_mask = error < threshold

        # Get the inliers from the points whose error is lower than the threshold
        inliers[~inliers] = error_mask

        #import pdb
        #pdb.set_trace()

        # Compare inlier counts
        if np.sum(inliers) > best_inliers_count:
            best_homography = H
            best_inliers_mask = inliers
            best_inliers_count = np.sum(inliers)

        # Update iteration count
        iterations+=1

    # End of RANSAC loop
    logger.debug("[Ransac] Finished iterations. Total {0}, Bad H {1}, Non Convex {2}", iterations, count_bad_homography, count_non_convex)

    logger.debug("Inliers: {0}", best_inliers_count)

    return best_homography, best_inliers_mask.tolist()

if __name__ == '__main__':
    import feature
    im_region = feature.im_read('C:/Users/xeroj/Desktop/Local_Programming/Vision-Rat-Brain/scripts_testing/region.jpg')

    feature.match_region_nissl(im_region, nissl_level=34)