# -*- coding: utf-8 -*-
from skimage.future import graph
from skimage import data, segmentation, color, filters, io
from skimage.util.colormap import viridis

from PIL import Image
import numpy as np

#img = data.coffee()
filename = "dataset3/Hist-Level-01.jpg"
img = np.array(Image.open(filename))
gimg = color.rgb2gray(img)

labels = segmentation.slic(img, compactness=30, n_segments=400)
edges = filters.sobel(gimg)
edges_rgb = color.gray2rgb(edges)

g = graph.rag_boundary(labels, edges)

out = graph.draw_rag(labels, g, edges_rgb, node_color="#999999",
                     colormap=viridis)

io.imshow(out)
io.show()