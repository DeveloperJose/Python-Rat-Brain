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

def ransac(src_pts, dst_pts, corners, threshold=10,max_iters=1000,max_inliers=np.inf):
    # Keep track of our best results
    best_homography = None
    best_err = np.inf
    best_inliers = None

    count_non_convex = 0
    count_bad_homography = 0

    # Loop for the specified iterations
    # You could also do it using time
    iterations = 0
    while iterations < max_iters:
        # Get all the indices as a mask of booleans
        # This is where we will store the inliers
        inliers = np.full(src_pts.shape[0], False, np.bool)

        # Randomly select 4 points
        rand_indices = np.random.choice(np.arange(src_pts.shape[0]), 4)
        rand_src = src_pts[rand_indices]
        rand_dst = dst_pts[rand_indices]

        # They are inliers naturally
        inliers[rand_indices] = True

        import pdb
        pdb.set_trace()

        # Create the homography from the 4 points
        H, mask = cv2.findHomography(rand_src, rand_dst, method=0)

        # Check if the homography is valid
        if H is None or len(H.shape) != 2:
            iterations+=1
            count_bad_homography+=1
            continue

        # Transform the corners
        try:
            transformedCorners = cv2.perspectiveTransform(corners, H)
        except:
            iterations+=1
            count_non_convex+=1
            continue

        # Check the convexity
        isConvex = cv2.isContourConvex(transformedCorners)
        if not isConvex:
            iterations+=1
            continue

        # Part 2: Get the inliers and outliers
        # Get the remaining points
        remaining_src = src_pts[~rand_indices]
        remaining_dst = dst_pts[~rand_indices]

        import pdb
        pdb.set_trace()

        # Get the error for each point
        error = np.linalg.norm(maybe_dst - cv2.convertPointsToHomogeneous(np.dot(H, maybe_src)))

    # End of RANSAC loop
    logger.debug("[Ransac] Finished iterations. Total {0}, Bad H {1}, Non Convex {2}", iterations, count_bad_homography, count_non_convex)

    return best_homography, inliers

if __name__ == '__main__':
    import feature
    im_region = feature.im_read('C:/Users/xeroj/Desktop/Local_Programming/Vision-Rat-Brain/scripts_testing/region.jpg')

    feature.match_region_nissl(im_region, nissl_level=34)