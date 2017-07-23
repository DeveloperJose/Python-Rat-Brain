import numpy as np
import cv2

import config
import logbook
logger = logbook.Logger(__name__)

def calc_reproj_error(H, src_pts, dst_pts):
    # How many points are there in total?
    total_pts = src_pts.shape[0]

    # Convert the source points to homogeneous coordinates
    # This is so we can multiply a (2x2) point with a (3x3) matrix
    src_pts_hom = cv2.convertPointsToHomogeneous(src_pts)

    # Calculate the reprojection error
    dot_prod = np.sum(H * src_pts_hom, axis=2)
    dot_prod_hom = cv2.convertPointsFromHomogeneous(dot_prod).reshape(total_pts, 2)
    difference = dst_pts - dot_prod_hom
    error = np.linalg.norm(difference, axis=1)

    return error

def calc_new_homography(H, src_pts, dst_pts, threshold):
    # First calculate the reprojection error
    error = calc_reproj_error(H, src_pts, dst_pts)

    # Get the inliers from the points whose error is lower than the threshold
    error_mask = error < threshold
    inliers = error_mask

    # Sum all the errors to find the total error
    total_error = error[error_mask].sum()

    # Update the homography using the inliers as well
    update_inliers = inliers.copy()

    # Update the homography with the new set of inliers
    new_src = src_pts[update_inliers]
    new_dst = dst_pts[update_inliers]

    H, mask = cv2.findHomography(new_src, new_dst, method=0)

    return H, inliers, error, total_error

def ransac(src_pts, dst_pts, corners, src_kp, dst_kp, im1, im2, threshold=20,max_iters=1500):
    if not config.NEW_RANSAC:
        return cv2_ransac(src_pts, dst_pts)

    # Keep track of our best results
    best_comparison_metric = 0
    best_homography = None
    best_inliers_count = 0
    best_inliers_mask = None
    best_inliers_original = None
    best_total_error = sys.maxsize

    # Debug information
    debug_min_error = sys.maxsize
    debug_max_error = 0
    debug_avg_error = 0

    # Looping variables
    count_non_convex = 0
    count_bad_homography = 0
    iterations = 0

    # Constants
    total_pts = src_pts.shape[0]

    # Loop for the specified iterations
    # You could also do it using time
    while iterations < max_iters:
        # Get all the indices as a mask of booleans
        inliers = np.full(total_pts, False, np.bool)

        # Randomly get 4 indices for points
        # No duplicates are allowed (replace=False)
        rand_indices = np.random.choice(total_pts, 4, replace=False)

        # They are inliers, naturally
        inliers[rand_indices] = True

        # Original 4 points for debug display
        # Make sure it's a copy or we get in trouble with references
        original_pts = inliers.copy()

        # Get the 4 points from the source and destination
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
            transformed_corners = cv2.perspectiveTransform(corners, H)
        except:
            iterations+=1
            count_non_convex+=1

        # Check the convexity of the transformed corners
        isConvex = cv2.isContourConvex(transformed_corners)
        if not isConvex:
            iterations+=1
            count_non_convex+=1
            continue

        # Part 2: Get the inliers and outliers
        for i in range(5):
            H_2, inliers_2, error_2, total_error_2 = calc_new_homography(H, src_pts, dst_pts, threshold)

            if H_2 is not None:
                # Linear combination
                #vec1 = H_2[0][0:2]
                #vec2 = H_2[1][0:2]
                #vec1_mag = np.linalg.norm(vec1)
                #vec2_mag = np.linalg.norm(vec2)
                #angle = np.rad2deg(util.angle_between(vec1, vec2))
                #inlier_count = np.sum(inliers_2)

                #m0 = 0.25 * (inlier_count/1000)
                #m1 = 0.4 * (inlier_count/total_pts)
                #m2 = 0.15 * min(vec1_mag/vec2_mag, vec2_mag/vec1_mag)
                #m3 = 0.2 * np.abs(np.sin(angle))
                #linear = m0 + m1 + m2 + m3

                # Compare inlier counts to update our best model
                #if np.sum(inliers) > best_inliers_count:

                #kp_inliers = src_kp[inliers_2]
                #metric = 0
                #for kp in kp_inliers:
                #    metric += (kp.size)
                metric = np.sum(inliers_2)
                if metric > best_comparison_metric:
                    H = H_2
                    inliers = inliers_2
                    error = error_2
                    total_error = total_error_2

                    best_comparison_metric = metric
                    best_homography = H
                    best_inliers_mask = inliers
                    best_inliers_count = np.sum(inliers)
                    best_inliers_original = original_pts
                    best_total_error = total_error

                    # Debug information to see the range of error
                    debug_min_error = min(error.min(), debug_min_error)
                    debug_max_error = max(error.max(), debug_max_error)
                    debug_avg_error = (np.average(error))

        # Update iteration count
        iterations+=1

    # End of RANSAC loop
    logger.debug("[Ransac] Finished iterations. Total {0}, Bad H {1}, Non Convex {2}, Decent {3}", iterations, count_bad_homography, count_non_convex, iterations-count_bad_homography-count_non_convex)

    return {
            "homography": best_homography,
            'metric': best_comparison_metric,
            'inlier_mask': best_inliers_mask,
            'inlier_count': best_inliers_count,
            'original_inlier_mask': best_inliers_original,
            'total_error': best_total_error,
            'min_error': debug_min_error,
            'max_error': debug_max_error,
            'avg_error': debug_avg_error,
            'src_pts': src_pts,
            'dst_pts': dst_pts
            }

def cv2_ransac(src_pts, dst_pts):
    import cv2
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, config.RANSAC_REPROJ_TRESHHOLD, None, config.RANSAC_MAX_ITERS, config.RANSAC_CONFIDENCE)
    return  {
            "homography": H,
            "inlier_mask": mask,
            "inlier_count": mask.ravel().sum(),
            "original_inlier_mask": None,
            "total_error": -1,
            "min_error": -1,
            "max_error": -1
            }

import sys
logbook.StreamHandler(sys.stdout, level=logbook.DEBUG, format_string=config.LOGGER_FORMAT_STRING).push_application()
if __name__ == '__main__':
    import feature
    im_region = feature.im_read('C:/Users/xeroj/Desktop/Local_Programming/Vision-Rat-Brain/scripts_testing/region.jpg')

    feature.match_region_nissl(im_region, nissl_level=34)