# -*- coding: utf-8 -*-
import sys
import os
import numpy as np

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import config
import feature
from graph import Graph
from dialog import ResultsDialog
from thread import MatchingThread

class Prototype(QWidget):
    def __init__(self, parent = None):
        super(Prototype, self).__init__(parent)

        # ================================================================================
        #   Window Setup
        # ================================================================================
        self.setWindowTitle("Prototype Demo")
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)

        # ================================================================================
        #   Variable Setup
        # ================================================================================
        self.nissl_filename = os.path.join(config.NISSL_DIR, config.NISSL_DEFAULT_FILE)

        # ================================================================================
        #   UI Layout Preparation
        # ================================================================================
        layout_main = QVBoxLayout() # ***** Vertical Layout (Main)

        # ***** Top Layout (Input Layout + Region Layout)
        layout_main.addLayout(self.get_layout_top())

        # Set the main layout
        self.setLayout(layout_main)

        # ================================================================================
        #   Routines after UI Preparation
        # ================================================================================
        self.refresh_image() # Set default image

        self.thread_match = MatchingThread(self)
        self.thread_match.startProgress.connect(self.on_thread_match_start)
        self.thread_match.updateProgress.connect(self.on_thread_match_update)
        self.thread_match.endProgress.connect(self.on_thread_match_end)


    ##################################################################################
    #   UI Layout/Widget Routines
    ##################################################################################
    def get_layout_top(self):
        layout = QHBoxLayout()
        layout.addStretch(1)

        layout.addWidget(self.get_widget_input())
        layout.addWidget(self.get_widget_region())

        return layout

    def get_widget_input(self):
        canvas_box = QGroupBox("Input Image")
        layout = QVBoxLayout()

        # *** Button (Open File)
        self.btn_open = QPushButton("Open Nissl file")
        self.btn_open.clicked.connect(self.on_click_btn_open)
        layout.addWidget(self.btn_open)

        # *** Canvas (Input Image)
        self.canvas_input = Graph(self, width=1, height=8, dpi=500)
        self.canvas_input.corners_callback = self.on_canvas_input_corners_update
        layout.addWidget(self.canvas_input)

        canvas_box.setLayout(layout)
        return canvas_box

    def get_widget_region(self):
        canvas_box = QGroupBox("Selected Region")
        layout = QVBoxLayout()

        # ================================================================================
        # ==================== Horizontal Layout (Matches Button + Add Entire Image Button)
        match_layout = QHBoxLayout()

        # *** Button (Find Best Match)
        self.btn_find_match = QPushButton("Find best matches")
        self.btn_find_match.clicked.connect(self.on_click_btn_find_match)
        self.btn_find_match.setEnabled(False)
        match_layout.addWidget(self.btn_find_match)

        self.btn_add_image = QPushButton("Add entire image")
        self.btn_add_image.clicked.connect(self.on_click_btn_add_image)
        match_layout.addWidget(self.btn_add_image)

        layout.addLayout(match_layout)
        # ==================== End Horizontal Layout (Match Layout)
        # ================================================================================

        progress_layout = QHBoxLayout()
        # *** ProgressBar (Matching Completion)
        self.progressbar_match = QProgressBar()
        self.progressbar_match.setValue(0)
        progress_layout.addWidget(self.progressbar_match)

        layout.addLayout(progress_layout)

        # *** Label (Matching Status)
        self.label_match_status = QLabel("Select a region you want to find a match for")
        layout.addWidget(self.label_match_status)

        # *** Canvas (Region Crops)
        self.canvas_region = Graph(self, width=5, height=5, dpi=100)
        self.canvas_region.is_interactive = False
        layout.addWidget(self.canvas_region)

        # *** Label (Ratio Test Distance)
        self.label_slider_ratio_test = QLabel("")
        layout.addWidget(self.label_slider_ratio_test)

        # *** Slider (Ratio Test Distance)
        self.slider_ratio_test = QSlider(Qt.Horizontal)
        self.slider_ratio_test.valueChanged.connect(self.on_slider_change_ratio_test)
        self.slider_ratio_test.setMinimum(0)
        self.slider_ratio_test.setMaximum(100)
        self.slider_ratio_test.setValue(int(config.DISTANCE_RATIO * 100))
        self.slider_ratio_test.setTickPosition(QSlider.NoTicks)
        self.slider_ratio_test.setTickInterval(1)
        self.slider_ratio_test.setEnabled(False)
        layout.addWidget(self.slider_ratio_test)

        # *** Label (Angle)
        self.label_angle = QLabel("Angle: 0")
        layout.addWidget(self.label_angle)

        # *** Slider (Angle)
        self.slider_angle = QSlider(Qt.Horizontal)
        self.slider_angle.valueChanged.connect(self.on_slider_change_angle)
        self.slider_angle.setMinimum(0)
        self.slider_angle.setMaximum(360)
        self.slider_angle.setTickPosition(QSlider.NoTicks)
        self.slider_angle.setTickInterval(1)
        self.slider_angle.setEnabled(False)
        layout.addWidget(self.slider_angle)

        # *** Label (Warp Points)
        self.label_warp_points = QLabel("")
        layout.addWidget(self.label_warp_points)

        # *** Slider (Warp Points)
        self.slider_warp_points = QSlider(Qt.Horizontal)
        self.slider_warp_points.valueChanged.connect(self.on_slider_change_warp_points)
        self.slider_warp_points.setMinimum(3)
        self.slider_warp_points.setMaximum(50)
        self.slider_warp_points.setTickPosition(QSlider.NoTicks)
        self.slider_warp_points.setTickInterval(1)
        self.slider_warp_points.setValue(5)
        self.slider_warp_points.setEnabled(False)
        layout.addWidget(self.slider_warp_points)

        # *** Label (Warp Disp Min)
        self.label_warp_disp_min = QLabel("Min Displacement: ")
        layout.addWidget(self.label_warp_disp_min)

        # *** Label (Warp Disp Max)
        self.label_warp_disp_max = QLabel("Max Displacement: ")
        layout.addWidget(self.label_warp_disp_max)

        from qrangeslider import QRangeSlider
        self.slider_warp_disp = QRangeSlider()
        self.slider_warp_disp.setMin(0)
        self.slider_warp_disp.setMax(50)
        self.slider_warp_disp.setRange(1, 5)
        self.slider_warp_disp.setEnabled(False)
        #self.slider_warp_disp.startValueChanged.connect(self.on_slider_warp_disp_start)
        #self.slider_warp_disp.setBackgroundStyle('background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #222, stop:1 #333);')
        #self.slider_warp_disp.handle.setStyleSheet('background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #282, stop:1 #393);')
        layout.addWidget(self.slider_warp_disp)

        self.btn_warp = QPushButton("Warp")
        self.btn_warp.clicked.connect(self.on_click_btn_warp)
        self.btn_warp.setEnabled(False)
        layout.addWidget(self.btn_warp)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.on_click_btn_reset)
        self.btn_reset.setEnabled(False)
        layout.addWidget(self.btn_reset)

        canvas_box.setLayout(layout)
        return canvas_box

    ##################################################################################
    #   Slider Events
    ##################################################################################
    def on_slider_change_ratio_test(self):
        new_ratio = float(self.slider_ratio_test.value() / 100.0)
        config.DISTANCE_RATIO = new_ratio
        self.label_slider_ratio_test.setText("Distance Ratio Test: " + str(config.DISTANCE_RATIO))

    def on_slider_change_angle(self):
        angle = self.slider_angle.value()
        self.label_angle.setText("Angle: " + str(angle))

        if self.canvas_region.im is not None:
            import scipy.misc
            im = scipy.misc.imrotate(self.region, angle)
            self.canvas_region.imshow(im)

    def on_slider_change_warp_points(self):
        points = self.slider_warp_points.value()
        self.label_warp_points.setText("Warp Points: " + str(points))

    def on_slider_change_warp_min(self):
        pass

    def on_slider_change_warp_max(self):
        pass

    ##################################################################################
    #   Class Functions
    ##################################################################################
    def refresh_image(self):
        im = feature.im_read(self.nissl_filename)
        self.canvas_input.imshow(im)

    def set_im_region(self, im_region):
        w, h, c = im_region.shape

        # Scale
        #import scipy.misc
        #size = (300, 300)
        #print ("Size before scale", im_region.shape)
        #im_region = scipy.misc.imresize(im_region, size)

        # Saving
        #feature.im_write("part.jpg", im_region)

        self.region = im_region
        self.canvas_region.imshow(self.region)
        self.canvas_input.clear_corners()
        self.btn_find_match.setEnabled(True)
        self.btn_warp.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.slider_angle.setEnabled(True)
        self.slider_ratio_test.setEnabled(True)
        self.slider_warp_points.setEnabled(True)
        self.slider_warp_disp.setEnabled(True)

        if config.SAVE_REGION:
            feature.im_write('region.jpg', im_region)

    ##################################################################################
    #   Canvas Events
    ##################################################################################
    def on_canvas_input_corners_update(self):
        count = len(self.canvas_input.corners)

        if count == 4:
            x, y = [], []

            for corner in self.canvas_input.corners:
                # Numpy slicing uses integers
                x.append(corner[0].astype(np.uint64))
                y.append(corner[1].astype(np.uint64))

            x1, x2, y1, y2 = min(x), max(x), min(y), max(y)
            im_region = self.canvas_input.im[y1:y2, x1:x2].copy()
            self.set_im_region(im_region)

        # Redraw corner scatterplot
        return True

    ##################################################################################
    #   Button Events
    ##################################################################################
    def on_click_btn_find_match(self):
        self.thread_match.set_im(self.canvas_region.im)
        self.thread_match.start()

    def on_click_btn_add_image(self):
        self.set_im_region(self.canvas_input.im.copy())

    def on_click_btn_warp(self):
        min_disp = self.slider_warp_disp.getRange()[0]
        max_disp = self.slider_warp_disp.getRange()[1]
        print("Range: ", min_disp, max_disp)
        im_warp = feature.warp(self.canvas_region.im, 5, min_disp, max_disp)
        self.canvas_region.imshow(im_warp)

    def on_click_btn_reset(self):
        self.set_im_region(self.region)

    def on_click_btn_open(self):
        new_filename, extra = QFileDialog.getOpenFileName(self, 'Open file',
                                            config.NISSL_DIR, "Image files (*.jpg *.png)")

        if new_filename:
            self.nissl_filename = new_filename
            self.refresh_image()

    ##################################################################################
    #   Thread Events
    ##################################################################################
    def on_thread_match_start(self, total):
        self.progressbar_match.setMaximum(total)

        self.canvas_input.is_interactive = False

        self.btn_open.setEnabled(False)
        self.btn_find_match.setEnabled(False)

        self.slider_ratio_test.setEnabled(False)
        self.slider_angle.setEnabled(False)

    def on_thread_match_update(self, index):
        self.progressbar_match.setValue(index)
        self.label_match_status.setText("Matching with plate " + str(index))

    def on_thread_match_end(self, matches):
        self.canvas_input.is_interactive = True

        self.btn_open.setEnabled(True)
        self.btn_find_match.setEnabled(True)

        self.slider_ratio_test.setEnabled(True)
        self.slider_angle.setEnabled(True)

        if len(matches) <= 0:
            self.label_match_status.setText("Didn't find a good match.")

        else:
            self.label_match_status.setText("Found " + str(len(matches)) + " possible matches")
            results_diag = ResultsDialog(self.nissl_filename, matches, self)
            results_diag.show()

##################################################################################
#   Program Main
##################################################################################
def main():
    app = QApplication(sys.argv)
    ui = Prototype()
    ui.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()