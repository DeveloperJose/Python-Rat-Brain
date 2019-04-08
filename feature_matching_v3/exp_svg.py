import numpy as np
import pylab as plt
from PIL import Image
import random
from svgpathtools import svg2paths2, wsvg
paths, attributes, svg_attributes = svg2paths2('Level33.svg')

for i in range(len(attributes)):
    attributes[i]['fill'] = "#%06x" % random.randint(0, 0xFFFFFF)

#%%
wsvg(paths[0:5], attributes=attributes[0:5], svg_attributes=svg_attributes, filename='output.svg')

#%%
for i in range(len(paths)):
    wsvg(paths[i], attributes=[attributes[i]], svg_attributes=svg_attributes, filename='output'+str(i)+'.svg')