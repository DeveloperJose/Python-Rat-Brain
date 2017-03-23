# -*- coding: utf-8 -*-
from PyQt5.QtCore import QThread, Signal
import numpy as np

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

    def run(self):
        results = np.array([])

        # Let subscriber know the total
        self.startProgress.emit(config.NISSL_COUNT)

        for nissl_level in range(1, config.NISSL_COUNT + 1):
            match = feature.match(self.im, nissl_level)
            self.updateProgress.emit(nissl_level)

            if match is None:
                continue

            if results is None:
                results = np.array([match])
            else:
                results = np.hstack((results, np.array([match])))

        self.endProgress.emit(results)