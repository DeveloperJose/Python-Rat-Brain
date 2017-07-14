# -*- coding: utf-8 -*-
import cv2
import numpy as np
import logbook
logger = logbook.Logger(__name__)

import config
import atlas
import util
import sift
import ransac
# ****************************** Matcher Parameters
FLANN = cv2.FlannBasedMatcher(config.FLANN_INDEX_PARAMS, config.FLANN_SEARCH_PARAMS)
BF = cv2.BFMatcher(normType=cv2.NORM_HAMMING)

class ImageInfo(object):
    def __init__(self, im, title, filename):
        self.im = im
        self.title = title
        self.filename = filename

class Match(object):
    def __init__(self, nissl_level, matches, ransac_results, extra_data):
        self.nissl_level = nissl_level
        self.matches = matches
        self.ransac_results = ransac_results

        # Ransac Result Parsing
        self.H = ransac_results["homography"]
        self.inlier_count = ransac_results["inlier_count"]
        self.homography_det = np.linalg.det(self.H)
        self.cond_num = np.linalg.cond(self.H)

        # Variable Parsing
        self.matches_count = len(matches)
        self.inlier_ratio = self.inlier_count / self.matches_count * 100

        # Extra Data Parsing
        self.im_results = extra_data['images']
        self.is_convex = extra_data['is_convex']

        # Linear Combination Vectors
        self.vec1 = self.H[0][0:2]
        self.vec2 = self.H[1][0:2]

        self.vec1_mag = np.linalg.norm(self.vec1)
        self.vec2_mag = np.linalg.norm(self.vec2)
        self.angle = np.rad2deg(util.angle_between(self.vec1, self.vec2))

        self.vec_arr = np.array([[self.H[0][0], self.H[0][1]], [self.H[1][0], self.H[1][1]]])
        self.vec_arr_cond = np.linalg.cond(self.vec_arr)

        # Linear combination
        self.a0 = (self.inlier_count/1000)
        self.a1 = (self.inlier_count/self.matches_count)
        self.a2 = min(self.vec1_mag/self.vec2_mag, self.vec2_mag/self.vec1_mag)
        self.a3 = np.abs(np.sin(self.angle))
        self.linear = ransac_results['metric']

    def comparison_key(self):
        return self.linear

    def get_results(self):
        return {
                'Plate #': self.nissl_level,
                'Matches': self.matches_count,
                'Inliers': self.inlier_count,
                'LinearComb': self.linear,
                'Inliers/1000': self.a0,
                'Inlier Ratio': self.a1,
                'Min(x/y,y/x)': self.a2,
                'abs(sin(angle))': self.a3,
                'Cond #': self.cond_num,
                'H Det': self.homography_det
                }

def __match(im1, kp1, des1, im2, kp2, des2, atlas_level):
    if des1 is None or des2 is None:
        logger.debug("Des1 or Des2 are empty. Cannot match")
        return None

    if len(des2) <= 1:
        logger.debug("Des2 has 1 or no features")
        return None

    if config.MATCH_WITH_FLANN:
        matches = FLANN.knnMatch(des1, des2, k=2)
    else:
        matches = BF.knnMatch(des1, des2, k=2)

    # Apply Ratio Test
    matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * config.DISTANCE_RATIO]

    if len(matches) < config.MIN_MATCH_COUNT:
        logger.debug("Matches lower than threshold. {0} < {1}", len(matches), config.MIN_MATCH_COUNT)
        return None

    # For homography calculation
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

    im1_h, im1_w = im1.shape[:2]
    corners = np.float32([[0, 0], [0, im1_h-1], [im1_w-1, im1_h-1], [im1_w-1, 0]]).reshape(-1, 1, 2)

    # Calculate the homography using RANSAC
    ransac_results = ransac.ransac(src_pts, dst_pts, corners, config.RANSAC_REPROJ_TRESHHOLD, config.RANSAC_MAX_ITERS)

    extra_data = __is_good_match(ransac_results['homography'], corners)
    if not extra_data:
        return None

    dict.update(extra_data,
            {
            'images': __get_extra_images(im1, kp1, im2, kp2, matches, ransac_results, corners),
            'corners': corners
            })

    return Match(atlas_level, matches, ransac_results, extra_data)

def __is_good_match(H, corners):
    # Check homography validity
    if H is None or len(H.shape) != 2:
        logger.debug("Couldn't get homography")
        return False

    # Attempt to transform corners based on homography
    try:
        transformed_corners = cv2.perspectiveTransform(corners, H)
    except:
        logger.debug("Couldn't transform corners")
        return False

    # Calculate the homography determinant
    det = np.linalg.det(H)

    # If it's too low for comfort, don't consider the match
    if abs(det) < config.HOMOGRAPHY_DETERMINANT_THRESHOLD or abs(det) > 20:
        logger.debug("Failed homography test")
        return False

    # Only accept corners that remain convex after being transformed
    is_convex = cv2.isContourConvex(transformed_corners)
    if not is_convex and not config.ALLOW_NON_CONVEX_CORNERS:
        logger.debug("Not convex")
        return False

    # Get the moments
    original_moments = cv2.moments(corners)
    transformed_moments = cv2.moments(transformed_corners)

    # Get the 7 Hu invariant moments
    original_hu_moments = cv2.HuMoments(original_moments).flatten()
    transformed_hu_moments = cv2.HuMoments(transformed_moments).flatten()

    # Find the Euclidean distance between the moments
    hu_distance = np.linalg.norm(original_hu_moments - transformed_hu_moments)

    # Ignore moments that are too large
    if hu_distance > config.HU_DISTANCE_THRESHOLD:
        logger.debug("Failed hu moment test")
        return False

    return {
            'transformed_corners': transformed_corners,
            'h_det': det,
            'is_convex': is_convex,
            'original_moments': original_moments,
            'transformed_moments': transformed_moments,
            'original_hu_moments': original_hu_moments,
            'transformed_hu_moments': transformed_hu_moments,
            'hu_distance': hu_distance
            }

def __get_extra_images(im1, kp1, im2, kp2, matches, ransac_results, corners):
    H = ransac_results['homography']

    # ******************** Image 1: All lines
    im_all_lines = cv2.drawMatches(im1, kp1, im2, kp2, matches, None, None, None, None, cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    # ******************** Image 2: Only inliers + corner rectangle
    # Get the inliers
    inlier_mask = ransac_results["inlier_mask"]

    # Draw a polygon on the second image joining the transformed corners
    transformedCorners = cv2.perspectiveTransform(corners, H)
    im_corner_rectangle = cv2.polylines(im2, [np.int32(transformedCorners)], True, config.MATCH_RECT_COLOR, 2, cv2.LINE_AA)

    # Draw the lines between the 2 images connecting matches
    im_inliers = cv2.drawMatches(im1, kp1, im_corner_rectangle, kp2, matches, None, None, None, inlier_mask.tolist(), cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # ******************** Image 3: The original 4 points
    # Get the 4 points
    original_inlier_mask = ransac_results["original_inlier_mask"]

    im_original = cv2.drawMatches(im1, kp1, im2, kp2, matches, None, None, None, original_inlier_mask.tolist(), cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    src_pts = ransac_results['src_pts']
    original_pts = src_pts[original_inlier_mask]

    transformed_pts = None
    try:

        transformed_pts = cv2.perspectiveTransform(np.array([original_pts]), H).reshape(4, 2)
    except:
        logger.debug("Couldn't transform original 4 pts")
        im_original_2 = im2

    if transformed_pts is not None:
        import pylab as plt
        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.imshow(im2)
        axes.scatter(*zip(*transformed_pts), s=5, c='b')
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        from PIL import Image
        im_original_2 = np.array(Image.open(buf))

        buf.close()
        plt.close()

    # ******************** Image 4: Transformed overlay
    # Warp the first image onto the second image
    im_warp = cv2.warpPerspective(im1, H, (im2.shape[1],im2.shape[0]))

    # Find the pixels which are visible in the warped image
    overlay_mask = (im_warp != 0)

    # Create a copy of the second image
    im_overlay = im2.copy()

    # Overlay the warped image on top of the 2nd image
    # If the mask is grayscale the overlay will only be applied to the red channel
    if len(overlay_mask.shape) == 3: # Overlay colors
        im_overlay[overlay_mask] = im_warp[overlay_mask]
    else:
        im_overlay[overlay_mask, 0] = im_warp[overlay_mask]

    # End of image results
    extra_images = (
            ImageInfo(im_all_lines, "All Lines", "all-lines"),
            ImageInfo(im_inliers, "Inliers Only", "inliers"),
            ImageInfo(im_original, "Original 4 Points", "orig-4"),
            ImageInfo(im_overlay, "Overlay", "overlay-warp"),
            ImageInfo(im_original_2, "4 points 2", "orig"))

    return extra_images

def match_region(im_region, atlas_level, kp1=None, des1=None):
    logger.debug("{0} Matching {1}", "="*75, atlas_level)

    # Extract sift descriptors from region (if needed)
    if kp1 is None:
        kp1, des1 = sift.extract_sift(im_region)

    # Load descriptors and image for plate
    kp2, des2 = atlas.load_sift(atlas_level)
    im_nissl = atlas.load_image(atlas_level)

    return __match(im_region, kp1, des1, im_nissl, kp2, des2, atlas_level)