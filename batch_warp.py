# -*- coding: utf-8 -*-
import numpy as np

import csv
import feature
import config

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()

im = feature.im_read('nissl_regions/Level-34-Region.jpg')
point_range = range(5, 15)
min_disp_range = range(0, 8)
max_disp_range = range(0, 8)
nissl_range = range(1, config.NISSL_COUNT + 1)

print("***** Beginning batch processing")

with open('level-34-3.csv', 'w') as csvfile:
    fieldnames = ['warp_points', 'warp_min_disp', 'warp_max_disp', 'plate', 'matches', 'inliers']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for points in point_range:
        print ("***** Progress: ", (points / point_range[0] / len(point_range)) * 100, "%")
        for min_disp in min_disp_range:
            for max_disp in max_disp_range:
                # Warp image
                print("[-- Warping: ", min_disp, max_disp, "]")
                im_warp = feature.warp(im, points, min_disp, max_disp)
                print("[-- Warped. Doing Nissl comparisons now]")
                # Matching
                best_inliers = -1
                best_match = None
                best_level = -1
                for nissl_level in nissl_range:
                    match = feature.match(im_warp, nissl_level)
                    #print("[", int(nissl_level / len(nissl_range) * 100), "%]", end='', flush=True)
                    printProgressBar(nissl_level, len(nissl_range))

                    if match is None:
                        continue

                    if match.inlier_count > best_inliers:
                        best_inliers = match.inlier_count
                        best_match = match
                        best_level = nissl_level
                print ("\n** [Completed] Inliers: ", best_inliers, "\n\n")
                writer.writerow({'warp_points': points, 'warp_min_disp': min_disp, 'warp_max_disp': max_disp, 'plate': best_level, 'matches': len(best_match.matches), 'inliers': best_inliers })
                csvfile.flush()