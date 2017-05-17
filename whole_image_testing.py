# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pylab as plt
import csv

import scipy.misc as misc

import feature
import timing
import config

import threading
from multiprocessing.pool import ThreadPool
import logbook
import sys
import os
import pickle
logger = logbook.Logger(__name__)
logbook.StreamHandler(sys.stdout, level=logbook.INFO).push_application()

class Testing(object):
    def progressBar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
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

    def set_main_nissl(self, atlas):
        if atlas == "SW":
            logger.info("Setting main atlas to Swanson")
            config.NISSL_DIR = 'atlas_swanson'
            config.NISSL_PREFIX = 'Level-'
            config.NISSL_DIGITS = 2
            config.NISSL_COUNT = 73
        else:
            logger.info("Setting main atlas to Paxinos/Watson")
            config.NISSL_DIR = 'atlas_pw'
            config.NISSL_PREFIX = 'RBSC7-'
            config.NISSL_DIGITS = 3
            config.NISSL_COUNT = 161

    def get_sift(self, atlas, level):
        if atlas == "SW":
            filename = 'Level-' + str(level).zfill(2) + ".sift"
            path = os.path.join('atlas_swanson', filename)
        else:
            filename = 'RBSC7-' + str(level).zfill(3) + ".sift"
            path = os.path.join('atlas_pw', filename)

        raw_sift = pickle.load(open(path, "rb"))
        kp, des = feature.unpickle_sift(raw_sift)
        return kp, des

    def get_im(self, atlas, level):
        if atlas == "SW":
            filename = 'Level-' + str(level).zfill(2) + ".jpg"
            path = os.path.join('atlas_swanson', filename)
        else:
            filename = 'RBSC7-' + str(level).zfill(3) + ".jpg"
            path = os.path.join('atlas_pw', filename)

        return feature.im_read(path)


    def process_with_pw(self, sw_level):
        self.progressBar(self.current, 74, prefix='Comparing to Swanson')
        match = feature.match_sift_nissl(self.im, self.kp, self.des, sw_level)

        self.current += 1

        if match is None:
            return

        with self.lock:
            self.writer.writerow((self.pw_level, sw_level, match.inlier_count, match.homography_det, match.svd_ratio))
            self.csvfile.flush()
            self.current += 1

    def process_with_sw(self, pw_level):
        self.progressBar(self.current, 162, prefix='Comparing to Paxinos Watson')
        match = feature.match_sift_nissl(self.im, self.kp, self.des, pw_level)

        self.current += 1

        if match is None:
            return

        with self.lock:
            self.writer.writerow((self.sw_level, pw_level, match.inlier_count, match.homography_det, match.svd_ratio))
            self.csvfile.flush()


    def main(self):
        process_with_pw = True
        process_with_sw = False

        # ***** Precalculation and saving of SIFT
        logger.info("Beginning SIFT preparations")
        timing.stopwatch()

        self.set_main_nissl('SW')
        for i in range(1, 73 + 1):
            feature.nissl_load_sift(i)

        self.set_main_nissl('PW')
        for i in range(1, 161 + 1):
            feature.nissl_load_sift(i)

        timing.stopwatch("Finished preparations. Time: ")

        # ********** For every Paxinos/Watson plate find the best matching Swanson plate
        if process_with_pw:
            logger.info("For every Paxinos/Watson plate finding the closest Swanson match")

            # Set the main atlas as Swanson
            self.set_main_nissl('SW')
            timing.stopwatch()

            with open('paxinos_watson_to_swanson.csv', 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
                writer.writerow(('PW Level', 'SW Level', 'Inliers', 'Homography Det', 'SVD'))

                self.writer = writer
                self.csvfile = csvfile
                self.lock = threading.Lock()

                for pw_level in range(1, 162):
                    logger.info("*************** Paxinos/Watson Plate {0}/{1}", pw_level, 161)
                    self.im = self.get_im("PW", pw_level)
                    self.kp, self.des = self.get_sift("PW", pw_level)
                    self.pw_level = pw_level
                    self.current = 0

                    # Use all the processors for matching
                    pool = ThreadPool(processes = cv2.getNumberOfCPUs())
                    pool.map(self.process_with_pw, range(1, 74))

            timing.stopwatch("Matching time: ")


        # ********** For every Swanson plate find the best matching Paxinos/Watson plate
        if process_with_sw:
            logger.info("For every Swanson plate finding the closest Paxinos/Watson match")

            # Set the main atlas as Paxinos/Watson
            self.set_main_nissl('PW')
            timing.stopwatch()

            with open('swanson_to_paxinos_watson.csv', 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
                writer.writerow(('SW Level', 'PW Level', 'Inliers', 'Homography Det', 'SVD'))

                self.writer = writer
                self.csvfile = csvfile
                self.lock = threading.Lock()


                for sw_level in range(1, 73):
                    logger.info("*************** Swanson {0}/{1}", sw_level, 73)
                    self.im = self.get_im("SW", sw_level)
                    self.kp, self.des = self.get_sift("SW", sw_level)
                    self.sw_level = sw_level
                    self.current = 0

                    # Use all the processors for matching
                    pool = ThreadPool(processes = cv2.getNumberOfCPUs())
                    pool.map(self.process_with_sw, range(1, 162))

            timing.stopwatch("Matching time: ")

if __name__ == '__main__':
    Testing().main()