# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image, ImageOps

def get_hist_gray(level):
    level_str = str(level).zfill(2)
    hist = Image.open('dataset\Complete-Atlas-Level-' + level_str).convert("L")
    w, h = hist.size
    
    first_crop = ImageOps.fit(hist, (200, 400), bleed=0.1)
    second_crop = first_crop.crop((0, 0, 100, 400))
    
    return np.array(second_crop)
def get_hist(level):
    level_str = str(level).zfill(2)
    hist = Image.open('dataset\Complete-Atlas-Level-' + level_str)
    w, h = hist.size
    
    first_crop = ImageOps.fit(hist, (200, 400), bleed=0.1)
    second_crop = first_crop.crop((0, 0, 100, 400))
    
    return np.array(second_crop)

def get_map(level):
    level_str = str(level).zfill(2)
    
    #hist = Image.open('dataset\Complete-Atlas-Level-' + level_str)
    #w, h = hist.size
    
    #first_crop = ImageOps.fit(hist, (200, 400), bleed=0.1)
    #second_crop = first_crop.crop((100, 0, 200, 400))
    
    #second_crop = ImageOps.mirror(second_crop)
    
    hist = Image.open('dataset\Map-Only-Atlas-Level-' + level_str + ".png")
    w, h = hist.size
    first_crop = ImageOps.fit(hist, (200, 400), bleed=0.1)
    second_crop = first_crop.crop((0, 0, 100, 400))
    return np.array(second_crop)

def get_map_gray(level):
    level_str = str(level).zfill(2)
    
    #hist = Image.open('dataset\Complete-Atlas-Level-' + level_str).convert("L")
    #w, h = hist.size
    
    #first_crop = ImageOps.fit(hist, (200, 400), bleed=0.1)
    #second_crop = first_crop.crop((100, 0, 200, 400))
    
    #second_crop = ImageOps.mirror(second_crop)
    hist = Image.open('dataset\Map-Only-Atlas-Level-' + level_str + ".png")
    w, h = hist.size
    first_crop = ImageOps.fit(hist, (200, 400), bleed=0.1)
    second_crop = first_crop.crop((0, 0, 100, 400))
    return np.array(second_crop)
