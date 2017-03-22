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
    updateProgress = Signal(int, str)
    endProgress = Signal(type(np.array([])))

    #You can do any extra things in this init you need, but for this example
    #nothing else needs to be done expect call the super's init
    def __init__(self, parent):
        QThread.__init__(self)        
        self.parent = parent
    
    def set_im(self, im):
        self.im = im
        
        # Convert from RGB to BGR
        self.im = cv2.cvtColor(self.im, cv2.COLOR_RGB2BGR)
    
    #A QThread is run by calling it's start() function, which calls this run()
    #function in it's own "thread". 
    def run(self):
        results = np.array([])
        index = 1
        
        kp1, des1 = feature.extract_sift(self.im, True)
        
        # Let subscriber know the total
        total = len(glob.glob1(config.NISSL_DIR, "*.sift"))
        self.startProgress.emit(total)
        
        for filename in os.listdir(config.NISSL_DIR):
            if filename.endswith(".sift"):
                path = os.path.join(config.NISSL_DIR, filename)
                raw_sift = pickle.load(open(path, "rb"))
                kp2, des2 = feature.unpickle_sift(raw_sift)
                        
                match = feature.match(filename, kp1, des1, kp2, des2, k=2)
                
                self.updateProgress.emit(index, filename)
                index += 1
                
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
        
        figure.canvas.mpl_connect("button_press_event", self.button_press_event)
    
    def clear_corners(self):
        self.corners = None
        
        # Clear scatterplot overlay if it exists
        if self.scat:
            self.scat.remove()
            self.scat = None    
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
            
            if draw_plot:
                self.scat = self.axes.scatter(*zip(*self.corners), c="r", s=10)            
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
        
        layout = QVBoxLayout()
        
        self.result_list = QTableWidget()
        self.result_list.setMinimumWidth(750)
        self.result_list.setMinimumHeight(450)
        self.result_list.setRowCount(len(matches))
        self.result_list.setColumnCount(5)
        self.result_list.setHorizontalHeaderLabels(['Filename', 'Match Count', 'Largest Distance', 'Unused', 'Unused'])
        
        row = 0
        
        matches = sorted(matches, key=lambda x:x.comparison_key())
        
        for match in matches:
            string_repr = match.to_string_array()
            self.result_list.setItem(row, 0, QTableWidgetItem(string_repr[0]))
            self.result_list.setItem(row, 1, QTableWidgetItem(string_repr[1]))
            self.result_list.setItem(row, 2, QTableWidgetItem(string_repr[2]))
            self.result_list.setItem(row, 3, QTableWidgetItem(string_repr[3]))
            row += 1
        
        layout.addWidget(self.result_list)
        
        self.setLayout(layout)
        self.setWindowTitle("Results for " + filename)
            
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
        self.slider_match.setValue(int(config.RATIO * 100))
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
        config.RATIO = new_ratio
        self.label_slider.setText("Tolerance Ratio: " + str(config.RATIO))
        
    def on_corners_update(self):
        count = len(self.canvas.corners)

        if count == 2:
            # Get the selected region
            # Note: You can only slice with INTEGERS
            top_left = self.canvas.corners[0].astype(np.uint64)
            bottom_right = self.canvas.corners[1].astype(np.uint64)            
            
            x = top_left[0]
            y = top_left[1]
            w = bottom_right[1] - top_left[1]
            h = bottom_right[0] - top_left[0]
            im_region = self.canvas.im[y:y+h, x:x+w].copy()
            cv2.imwrite("part.jpg", im_region)

            self.region_canvas.imshow(im_region)
            self.canvas.clear_corners()
            self.btn_match.setEnabled(True)
            
            return False
            
        return True
        
        
    def find_match(self):
        self.thread_match.set_im(self.region_canvas.im)
        self.thread_match.start()
        
    def start_progress(self, total):
        self.progress_match.setMaximum(total)
        self.btn_match.setEnabled(False)
        self.slider_match.setEnabled(False)
        
    def update_progress(self, index, filename):
        self.progress_match.setValue(index)
        
    def end_progress(self, matches):
        self.btn_match.setEnabled(True)
        
        if len(matches) <= 0:
            config.RATIO += 0.1
            self.label_match.setText("Didn't find a match. Trying with a higher tolerance ratio.")
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
        im = cv2.imread(self.filename)      
        
        # Convert from Matplotlib BGR to RGB
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        
        self.canvas.imshow(im)

def main():
    app = QApplication(sys.argv)
    ui = Prototype()
    ui.show()
    
    sys.exit(app.exec_())
	
if __name__ == '__main__':
    main()