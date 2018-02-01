# -*- coding: utf-8 -*-
from PyQt5.QtCore import QThread, Signal
from multiprocessing.pool import ThreadPool
import numpy as np
import cv2

import util_sift
import config
import matching
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class MatchingThread(QThread):
    def __init__(self, parent):
        QThread.__init__(self)
        self.parent = parent

    ##################################################################################
    #   Thread Signals
    ##################################################################################
    startProgress = Signal(int)
    updateProgress = Signal(int)
    endProgress = Signal(type(np.array([])))

    ##################################################################################
    #   Functions
    ##################################################################################
    def set_im(self, im):
        self.im = im

    def process_level(self, nissl_level):
        match = matching.match_region(self.im, nissl_level, self.kp1, self.des1)
        self.updateProgress.emit(self.total)
        self.total += 1

        if match is None:
            return

        inlier_percent = match.inlier_count / len(match.matches) * 100
        if (inlier_percent >= 90):
            logger.info("Level {0} has an inlier percentage of {1}", nissl_level, inlier_percent)

        if self.results is None:
            self.results = np.array([match])
        else:
            self.results = np.hstack((self.results, np.array([match])))

    def run(self):
        self.results = np.array([])

        # Let subscriber know the total
        self.startProgress.emit(config.NISSL_COUNT)

        # For updating the progress in the UI
        self.total = 1

        # Compute SIFT for region before starting
        self.kp1, self.des1 = util_sift.extract_sift(self.im)

        # Set multithreading if capable
        if config.MULTITHREAD:
            pool = ThreadPool(processes = cv2.getNumberOfCPUs())
        else:
            pool = ThreadPool(processes = 1)

        # Total number of images to compare against
        nissl = range(1, config.NISSL_COUNT + 1)

        # Begin mapping process
        pool.map(self.process_level, nissl)
        pool.close()
        pool.join()

        # Tell UI the results
        self.endProgress.emit(self.results)