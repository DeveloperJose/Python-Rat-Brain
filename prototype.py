# -*- coding: utf-8 -*-
import sys, os, cv2
import pylab as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

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
                print "setting initial p"
                self.corners = np.array([p])
                print self.corners
            else:
                print "attempting magic"
                temp = np.vstack((self.corners, p))
                self.corners = temp
                
                print temp
            
            draw_plot = True
            if self.corners_callback:
                draw_plot = self.corners_callback()
            
            if draw_plot:
                self.scat = self.axes.scatter(*zip(*self.corners), c="r", s=40)            
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
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        self.setWindowTitle("Image Region")

        self.canvas.imshow(im)
            
class Prototype(QWidget):
    def __init__(self, parent = None):
        super(Prototype, self).__init__(parent)
		
        self.nissl_root = "nissl"
        self.nissl_ext = ".jpg"
        self.filename = os.path.join(self.nissl_root, "Level-01.jpg")
          
        layout = QVBoxLayout()
        self.btn_open = QPushButton("Open Nissl file")
        self.btn_open.clicked.connect(self.open_file)
        layout.addWidget(self.btn_open)

        self.canvas = Graph(self)
        self.canvas.corners_callback = self.on_corners_update
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        self.setWindowTitle("Prototype Demo")

        self.refresh_image()
		
    def on_corners_update(self):
        count = len(self.canvas.corners)
        print "corners, count", count
        if count == 2:
            self.canvas.is_interactive = False
            
            top_left = self.canvas.corners[0]
            bottom_right = self.canvas.corners[1]            
            im_region = self.get_im_region(self.im, top_left, bottom_right)

            self.diag = ImageDialog(im=im_region, parent=self)
            self.diag.exec_()
            
        return True
    
    def get_im_region(self, im, top_left, bottom_right):
        region_x = top_left[0]
        region_y = top_left[1]
        region_width = bottom_right[1] - top_left[1]
        region_height = bottom_right[0] - top_left[0]
        
        return im[region_y:region_y+region_width, region_x:region_x+region_height]
        
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
        
        #if self.canvas.corners:
            #self.canvas.axes.scatter(*zip(*self.canvas.corners), c="r", s=40)
        self.im = im
        self.canvas.imshow(im)

def main():
    app = QApplication(sys.argv)
    ui = Prototype()
    ui.show()
    sys.exit(app.exec_())
	
if __name__ == '__main__':
    main()