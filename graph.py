# -*- coding: utf-8 -*-
import pylab as plt
import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5.QtWidgets import QSizePolicy

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

