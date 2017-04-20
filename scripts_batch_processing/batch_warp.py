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

filename = 'atlas_swanson_regions/Level-34-Region.jpg'
output_filename = 'results/Level-34-range2'
points = 30
disp_range = range(30, 50)
nissl_range = range(1, config.NISSL_COUNT + 1)

print("***** Beginning batch processing")
im = feature.im_read(filename)
csv_filename = output_filename + "-" + str(points) + "pts.csv"

with open(csv_filename, 'w') as csvfile:
    fieldnames = ['warp_points', 'warp_disp', 'plate', 'matches', 'inliers']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', lineterminator='\n')

    writer.writeheader()

    for disp in disp_range:
        printProgressBar(int(disp / disp[0]), len(disp_range), prefix='Disp: ')
        # Warp image
        im_warp = feature.warp(im, points, None, None, disp, None)
        print("[-- Warped. Doing Nissl comparisons now]")
        # Matching
        best_inliers = -1
        best_match = None
        best_level = -1
        for nissl_level in nissl_range:
            match = feature.match(im_warp, nissl_level)
            printProgressBar(nissl_level, len(nissl_range), prefix='Nissl Matching: ')

            if match is None:
                continue

            if match.inlier_count > best_inliers:
                best_inliers = match.inlier_count
                best_match = match
                best_level = nissl_level
        print ("** [Completed] Inliers: ", best_inliers, "\n\n")
        writer.writerow({'warp_points': points, 'warp_disp': disp, 'plate': best_level, 'matches': len(best_match.matches), 'inliers': best_inliers })
        csvfile.flush()