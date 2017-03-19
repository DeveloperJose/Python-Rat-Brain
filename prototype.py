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

class Worker(QThread):

    #This is the signal that will be emitted during the processing.
    #By including int as an argument, it lets the signal know to expect
    #an integer argument when emitting.
    startProgress = Signal(int)
    updateProgress = Signal(int)
    endProgress = Signal(str, int)

    #You can do any extra things in this init you need, but for this example
    #nothing else needs to be done expect call the super's init
    def __init__(self, parent):
        QThread.__init__(self)        
        self.parent = parent

    def unpickle_sift(self, array):
        keypoints = []
        descriptors = []
        for point in array:
            temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
            temp_descriptor = point[6]
            keypoints.append(temp_feature)
            descriptors.append(temp_descriptor)
        return keypoints, np.array(descriptors)
    
    def set_im(self, im):
        self.im = im
        #self.im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.kp1, self.des1 = self.sift.detectAndCompute(im, None)
    
    #A QThread is run by calling it's start() function, which calls this run()
    #function in it's own "thread". 
    def run(self):        
        best_filename = None
        best_count = -1        

        index = 1
        total = len(glob.glob1(self.parent.nissl_root, "*.sift"))
        
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        
        # Let subscriber know the total
        self.startProgress.emit(total)
        
        for filename in os.listdir(self.parent.nissl_root):
            if filename.endswith(".sift"):
                path = os.path.join(self.parent.nissl_root, filename)
                raw_sift = pickle.load(open(path, "rb"))
                kp2, des2 = self.unpickle_sift(raw_sift)
                
                matches = flann.knnMatch(des2, self.des1, k=2)
                
                # Apply ratio test
                good = []
                for m,n in matches:
                    if m.distance < 0.4*n.distance:
                        good.append([m])
                        
                print ("Matches?", path, len(good))
                if len(good) > best_count:
                    best_filename = path
                    best_count = len(good)
                    
                # Update progress
                self.updateProgress.emit(index)
                index += 1
                        
        self.endProgress.emit(best_filename, best_count)

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
                print ("setting initial p")
                self.corners = np.array([p])
                print (self.corners)
            else:
                print ("attempting magic")
                temp = np.vstack((self.corners, p))
                self.corners = temp
                
                print (temp)
            
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
            
class Prototype(QWidget):
    def __init__(self, parent = None):
        super(Prototype, self).__init__(parent)
		
        # Constants/Vars
        self.setWindowTitle("Prototype Demo")
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        self.nissl_root = "nissl"
        self.nissl_ext = ".jpg"
        self.filename = os.path.join(self.nissl_root, "Level-34.jpg")
          
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
        self.btn_match = QPushButton("Find best match")
        self.btn_match.clicked.connect(self.find_match)
        self.btn_match.setEnabled(False)
        matchLayout.addWidget(self.btn_match)
        
        self.progress_match = QProgressBar()
        matchLayout.addWidget(self.progress_match)
        canvasBoxLayout.addLayout(matchLayout)
        
        self.label_match = QLabel("Region not matched yet")
        canvasBoxLayout.addWidget(self.label_match)
        
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
		
    def on_corners_update(self):
        count = len(self.canvas.corners)
        print("Hey listen", count)
        if count == 2:
            # Get the selected region
            # Note: You can only slice with INTEGERS
            top_left = self.canvas.corners[0].astype(np.uint64)
            bottom_right = self.canvas.corners[1].astype(np.uint64)            
            
            print ("R")
            x = top_left[0]
            y = top_left[1]
            w = bottom_right[1] - top_left[1]
            h = bottom_right[0] - top_left[0]
            im_region = self.canvas.im[y:y+h, x:x+w].copy()

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
        self.label_match.setText("Searching for match....")
        
    def update_progress(self, index):
        self.progress_match.setValue(index)
        
    def end_progress(self, best_filename, best_count):
        self.btn_match.setEnabled(True)
        self.label_match.setText("Best Match: " + best_filename + ", " + str(best_count))
        
    def open_file(self):
        new_filename, extra = QFileDialog.getOpenFileName(self, 'Open file', 
                                            self.nissl_root, "Image files (*.jpg *.png)")
        
        if new_filename:
            self.filename = new_filename
            self.refresh_image()
            
    def refresh_image(self):
        im = cv2.imread(self.filename)
        
        # Convert from Matplotlib BGR to RGB
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        w, h, b = im.shape
        
        self.canvas.imshow(im)

def main():
    app = QApplication(sys.argv)
    ui = Prototype()
    ui.show()
    sys.exit(app.exec_())
	
if __name__ == '__main__':
    main()