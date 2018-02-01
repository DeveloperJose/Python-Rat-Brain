# Author: Jose G Perez
# Version 1.0
# Last Modified: January 31, 2018
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
