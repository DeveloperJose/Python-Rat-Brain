# -*- coding: utf-8 -*-
from PyQt5.QtCore import QThread, Signal
import numpy as np
import cv2
from multiprocessing.pool import ThreadPool

import config
import feature

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
        match = feature.match_sift_nissl(self.im, self.kp1, self.des1, nissl_level)
        self.updateProgress.emit(self.total)
        self.total += 1

        if match is None:
            return

        if self.results is None:
            self.results = np.array([match])
        else:
            self.results = np.hstack((self.results, np.array([match])))

    def run(self):
        print("thread run");
        self.results = np.array([])

        # Let subscriber know the total
        self.startProgress.emit(config.NISSL_COUNT)
        self.total = 1

        self.kp1, self.des1 = feature.extract_sift(self.im)

        pool = ThreadPool(processes = cv2.getNumberOfCPUs())
        nissl = range(1, config.NISSL_COUNT + 1)
        pool.map(self.process_level, nissl)
        pool.close()
        pool.join()

        self.endProgress.emit(self.results)