# -*- coding: utf-8 -*-
import sys, os, cv2, glob
import pylab as plt
import numpy as np
import pickle
import pdb
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import config, feature

# TODO: Create SIFT features automatically if not found
# TODO: Download Nissl files if not found

class Worker(QThread):

    #This is the signal that will be emitted during the processing.
    #By including int as an argument, it lets the signal know to expect
    #an integer argument when emitting.
    startProgress = Signal(int)
    updateProgress = Signal(int)
    endProgress = Signal(type(np.array([])))

    #You can do any extra things in this init you need, but for this example
    #nothing else needs to be done expect call the super's init
    def __init__(self, parent):
        QThread.__init__(self)        
        self.parent = parent
    
    def set_im(self, im):
        self.im = im
    
    #A QThread is run by calling it's start() function, which calls this run()
    #function in it's own "thread". 
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

class Graph(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=10, height=10, dpi=100):
        figure = plt.Figure(figsize=(width, height), dpi=dpi)
        figure.subplots_adjust(left   = 0.0,
                               right  = 1.0,
                               top    = 1.0,
                               bottom = 0.0,
                               wspace = 0.0,
                               hspace = 0.0)
        super(Graph, self).__init__(figure)
        
        self.parent = parent
        self.figure = figure
        
        # Fill the area with the graph
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.axes = figure.add_subplot(111)
        self.axes.set_xticks(())
        self.axes.set_yticks(())
        
        self.is_interactive = True
        self.corners_callback = None
        self.corners = None
        
        self.scatter_plots = []
        
        figure.canvas.mpl_connect("button_press_event", self.button_press_event)
    
    def clear_corners(self):
        self.corners = None
        
        # Clear scatterplot overlays if they exist
        if len(self.scatter_plots) > 0:
            for plot in self.scatter_plots:
                plot.remove()
                
        self.scatter_plots = []
        self.draw()
        
    def button_press_event(self, event):
        if not self.is_interactive:
            return
        
        # Right-click erases scatterplot overlays
        if event.button == 3:
            self.clear_corners()
        
        # Left-click adds point markers
        elif event.button == 1 and event.xdata and event.ydata:
            p = np.array([event.xdata, event.ydata])
            
            if self.corners is None:
                self.corners = np.array([p])

            else:
                temp = np.vstack((self.corners, p))
                self.corners = temp
                
            draw_plot = True
            if self.corners_callback:
                draw_plot = self.corners_callback()
            
            if draw_plot and self.corners is not None:              
                # Draw new overlay
                plot = self.axes.scatter(*zip(*self.corners), c="r", s=2)
                self.scatter_plots.append(plot)            
                self.draw()
    
    def imshow(self, im):
        # Fixes the issue of plot auto-rescaling
        # When overlaying scatterplot points
        self.axes.set_xlim([0, im.shape[1]])
        self.axes.set_ylim([im.shape[0], 0])
        
        self.im = im
        self.axes.imshow(im)
        self.draw()
        
class ImageDialog(QDialog):
    def __init__(self, im, parent=None):
        super(ImageDialog, self).__init__(parent)
        
        layout = QVBoxLayout()
        
        self.canvas = Graph(self, width=5, height=5, dpi=100)
        self.canvas.is_interactive = False
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        self.setWindowTitle("Image")

        self.canvas.imshow(im)
        
class ResultsDialog(QDialog):
    def __init__(self, filename, matches, parent=None):
        super(ResultsDialog, self).__init__(parent)
        
        self.filename = filename
        self.matches = matches
        
        layout = QVBoxLayout()
        
        self.result_list = QTableWidget()
        self.result_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.result_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.result_list.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.result_list.setMinimumWidth(750)
        self.result_list.setMinimumHeight(450)
        self.result_list.setRowCount(len(matches))
        self.result_list.setColumnCount(5)
        self.result_list.setHorizontalHeaderLabels(['Plate', 'Match Count', 'Inlier Count', 'Largest Distance', '', ''])
        
        row = 0
        matches = sorted(matches, key=lambda x:x.comparison_key(), reverse=True)
        
        for match in matches:
            string_repr = match.to_string_array()
            self.result_list.setItem(row, 0, QTableWidgetItem(string_repr[0]))
            self.result_list.setItem(row, 1, QTableWidgetItem(string_repr[1]))
            self.result_list.setItem(row, 2, QTableWidgetItem(string_repr[2]))
            self.result_list.setItem(row, 3, QTableWidgetItem(string_repr[3]))
            row += 1
        
        self.result_list.doubleClicked.connect(self.double_click_table)
        
        layout.addWidget(self.result_list)
        
        self.setLayout(layout)
        self.setWindowTitle("Results for " + filename)
    
    def double_click_table(self):
        rows = sorted(set(index.row() for index in
                      self.table.selectedIndexes()))
        
        if (len(rows) > 0):
            row = rows[0]
            match = self.matches[row]
            
            im_result = match.result
        
            image_diag = ImageDialog(im=im_result, parent=None)
            image_diag.show()
        
class Prototype(QWidget):
    def __init__(self, parent = None):
        super(Prototype, self).__init__(parent)
		
        # Constants/Vars
        self.setWindowTitle("Prototype Demo")
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        self.filename = os.path.join(config.NISSL_DIR, "Level-34.jpg")
          
        # ***** Main Layout
        main_layout = QVBoxLayout()
        #main_layout.addStretch(1)
        
        # ***** Images
        images_layout = QHBoxLayout()
        images_layout.addStretch(1)
        
        # Canvas for displaying input images
        self.canvas = Graph(self, width=1, height=8, dpi=500)
        self.canvas.corners_callback = self.on_corners_update
        
        # Create a box for the input canvas
        canvasBox = QGroupBox("Input Image")
        canvasBoxLayout = QVBoxLayout()
        
        # Button for opening image files
        self.btn_open = QPushButton("Open Nissl file")
        self.btn_open.clicked.connect(self.open_file)
        
        canvasBoxLayout.addWidget(self.btn_open) 
        canvasBoxLayout.addWidget(self.canvas)
        
        canvasBox.setLayout(canvasBoxLayout)        
        images_layout.addWidget(canvasBox)

        # Canvas for displaying region crops
        self.region_canvas = Graph(self, width=5, height=5, dpi=100)
        self.region_canvas.is_interactive = False
        
        # Create a box for the region canvas
        canvasBox = QGroupBox("Selected Region")
        canvasBoxLayout = QVBoxLayout()
        
        matchLayout = QHBoxLayout()
        self.btn_match = QPushButton("Find best matches")
        self.btn_match.clicked.connect(self.find_match)
        self.btn_match.setEnabled(False)
        matchLayout.addWidget(self.btn_match)
        
        self.progress_match = QProgressBar()
        matchLayout.addWidget(self.progress_match)
        canvasBoxLayout.addLayout(matchLayout)
        
        self.label_match = QLabel("Status: Nothing to report")
        canvasBoxLayout.addWidget(self.label_match)
        
        self.label_slider = QLabel("")
        canvasBoxLayout.addWidget(self.label_slider)
        
        self.slider_match = QSlider(Qt.Horizontal)
        self.slider_match.valueChanged.connect(self.slider_change)
        self.slider_match.setMinimum(0)
        self.slider_match.setMaximum(100)
        self.slider_match.setValue(int(config.DISTANCE_RATIO * 100))
        self.slider_match.setTickPosition(QSlider.NoTicks)
        self.slider_match.setTickInterval(1)
        canvasBoxLayout.addWidget(self.slider_match)
        
        canvasBoxLayout.addWidget(self.region_canvas)
        canvasBox.setLayout(canvasBoxLayout)        
        images_layout.addWidget(canvasBox)

        # Add images to main layout
        main_layout.addLayout(images_layout)
        
        # ***** Main layout
        self.setLayout(main_layout)

        # ***** Routines after UI creation
        self.refresh_image()
        
        self.thread_match = Worker(self)
        self.thread_match.startProgress.connect(self.start_progress)
        self.thread_match.updateProgress.connect(self.update_progress)
        self.thread_match.endProgress.connect(self.end_progress)        
		
    def slider_change(self):
        new_ratio = float(self.slider_match.value() / 100.0)
        config.DISTANCE_RATIO = new_ratio
        self.label_slider.setText("Distance Ratio Test: " + str(config.DISTANCE_RATIO))
        
    def on_corners_update(self):
        count = len(self.canvas.corners)

        if count == 4:
            x, y = [], []
            
            for corner in self.canvas.corners:
                # Numpy slicing uses integers
                x.append(corner[0].astype(np.uint64))
                y.append(corner[1].astype(np.uint64))
                
            x1, x2, y1, y2 = min(x), max(x), min(y), max(y)
            im_region = self.canvas.im[y1:y2, x1:x2].copy()
            w, h, c = im_region.shape
            
            import scipy.misc
            # Scale
            size = (100, 100)
            #print ("Size before scale", im_region.shape)
            #im_region = scipy.misc.imresize(im_region, size)
            
            # Rotation
            angle = 137 # In degrees
            #im_region = scipy.misc.imrotate(im_region, angle)
            
            feature.im_write("part.jpg", im_region)
            
            self.region_canvas.imshow(im_region)
            self.canvas.clear_corners()
            self.btn_match.setEnabled(True)

        # Redraw corner scatterplot            
        return True
        
        
    def find_match(self):        
        self.thread_match.set_im(self.region_canvas.im)
        self.thread_match.start()
        
    def start_progress(self, total):
        self.progress_match.setMaximum(total)
        self.btn_match.setEnabled(False)
        self.slider_match.setEnabled(False)
        
    def update_progress(self, index):
        self.progress_match.setValue(index)
        
    def end_progress(self, matches):
        self.btn_match.setEnabled(True)
        
        if len(matches) <= 0:
            config.DISTANCE_RATIO += 0.1
            self.label_match.setText("Didn't find a match. Trying with a higher distance ratio test.")
            self.slider_match.setValue(self.slider_match.value() + 10)
            self.find_match()
            
        else:
            self.slider_match.setEnabled(True)
            self.label_match.setText("Found " + str(len(matches)) + " possible matches")
            
            results_diag = ResultsDialog(self.filename, matches, self)
            results_diag.show()
            
    def open_file(self):
        new_filename, extra = QFileDialog.getOpenFileName(self, 'Open file', 
                                            config.NISSL_DIR, "Image files (*.jpg *.png)")
        
        if new_filename:
            self.filename = new_filename
            self.refresh_image()
            
    def refresh_image(self):
        im = feature.im_read(self.filename)       
        self.canvas.imshow(im)

def main():
    app = QApplication(sys.argv)
    ui = Prototype()
    ui.show()
    
    sys.exit(app.exec_())
	
if __name__ == '__main__':
    main()