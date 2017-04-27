# -*- coding: utf-8 -*-
import csv
import numpy as np
import pylab as plt

#FILENAME = 'results/csv/paxinos_watson_to_swanson.csv'
FILENAME = 'results/csv/swanson_to_paxinos_watson.csv'

# How many levels before another chart is created?
SPLIT_AFTER_LEVELS = 3

# The name of the first field in the CSV file
#MAIN_FIELD = 'PW Level'
MAIN_FIELD = 'SW Level'

# The name of the second field in the CSV file
#SEC_FIELD = 'SW Level'
SEC_FIELD = 'PW Level'

# The title for the legend in the charts
#LEGEND = 'PW Plate #'
LEGEND = 'SW Plate #'

# The title for the vertical axis in the charts
#AXIS = 'Swanson Plate #'
AXIS = 'PW Plate #'

# Size of figures
SIZE = (10, 4)

def new_figure():
    # figsize = (w, h)
    figure = plt.figure(figsize=SIZE)
    axes = figure.add_subplot(111)
    axes.set_xticks(())
    axes.set_ylabel(AXIS)

    return figure, axes

def autolabel(rects, axes):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()

        if height > 8:
            y = height - 8
            text_color = 'black'
        else:
            y = height
            text_color = 'black'

        x = rect.get_x() + rect.get_width()/2.0
        axes.text(x, y,'%d' % int(height), ha='center', va='bottom', color=text_color)


with open(FILENAME, 'r') as inputfile:
    reader = csv.DictReader(inputfile, delimiter=',', lineterminator='\n')

    past_level = None
    past_matches = []

    figure, axes = new_figure()

    bar_index = 0
    level = 0

    for row in reader:
        reader_level = row[MAIN_FIELD]

        # Check if it's the first level
        if past_level is None:
            past_level = reader_level
            past_matches = [row[SEC_FIELD]]
            continue

        # The levels changed
        if reader_level != past_level:
            # Create bar graph
            x = np.arange(bar_index, len(past_matches) + bar_index)
            y = np.fromiter(past_matches, dtype=np.uint32)
            bar_label = LEGEND + past_level
            rects = axes.bar(x, y, label=bar_label)

            # Add labels
            autolabel(rects, axes)

            # Add legend
            lgd = axes.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, ncol=1)

            bar_index += len(past_matches)
            level += 1

            if level >= SPLIT_AFTER_LEVELS:
                level = 0
                figure.savefig('figures/' + past_level + '.jpg', bbox_extra_artists=(lgd,), bbox_inches='tight')
                figure, axes = new_figure()

            # Reset
            past_level = reader_level
            past_matches = [row[SEC_FIELD]]
            continue

        # Same levels
        past_matches.append(row[SEC_FIELD])