#%% Visualization utilities for experiments
# Author: Jose G Perez
# Version 1.0
# Last Modified: March 20, 2020
from skimage import color
import pylab as plt
import numpy as np


def imshow(im, title=''):
    figure = plt.figure()
    plt.axis('off')
    plt.tick_params(axis='both',
                    left='off', top='off', right='off', bottom='off',
                    labelleft='off', labeltop='off', labelright='off', labelbottom='off')

    plt.title(title)
    plt.imshow(im)
    return figure


def imshow_matches__(im, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('PW Level')
    ax.set_ylabel('S Level')
    ax.set_title(title)
    plt.set_cmap(plt.get_cmap('hot'))
    plt.imshow(im)


def imshow_matches(im, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_xlabel('PW Level')
    # ax.set_ylabel('S Level')
    # ax.set_xticks(s_label)
    # ax.set_yticks(pw_label)
    ax.set_title(title)
    plt.set_cmap(plt.get_cmap('hot'))
    plt.imshow(im)


def overlay(im_bg, im_fg):
    img_color = np.stack((im_bg,)*3,axis=2)
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(im_fg)
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1]
    im_overlay = color.hsv2rgb(img_hsv)

    return im_overlay


def imshow_detailed(np_arr, title, axis_xlabel, axis_ylabel, xlabel_arr, ylabel_arr, value_to_str_func=lambda i, j, val: val):
    fig = plt.figure()
    ax = fig.gca()

    plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(8)

    ax.set_title(title)
    ax.set_xlabel(axis_xlabel)
    ax.set_ylabel(axis_ylabel)

    # Show all ticks
    ax.set_xticks(np.arange(len(xlabel_arr)))
    ax.set_yticks(np.arange(len(ylabel_arr)))

    # Proper labels
    ax.set_xticklabels(xlabel_arr)
    ax.set_yticklabels(ylabel_arr)

    # Show values
    if value_to_str_func is not None:
        for i in range(np_arr.shape[0]):
            for j in range(np_arr.shape[1]):
                text = ax.text(j, i, value_to_str_func(i, j, np_arr[i, j]), ha="center", va="center", color="w")

    ax.imshow(np_arr)