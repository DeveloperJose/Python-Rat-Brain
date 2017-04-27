# -*- coding: utf-8 -*-
import csv
import pylab as plt
filename = 'paxinos_watson_to_swanson.csv'
output = 'paxinos_watson_to_swanson_matches_only.csv'

with open(filename, 'r') as inputfile:
    with open(output, 'w') as outputfile:
        reader = csv.DictReader(inputfile, delimiter=',', lineterminator='\n')
        writer = csv.writer(outputfile, delimiter=',', lineterminator='\n')
        writer.writerow(('PW Level', 'SW', 'SW', 'SW', 'SW'))

        past_level = None
        past_matches = []

        #figure = plt.figure()
        #axes = figure.add_subplot(111)

        for row in reader:
            reader_level = row['PW Level']

            # Check if it's the first level
            if past_level is None:
                past_level = reader_level
                past_matches = [reader_level]
                past_matches.append(row['SW Level'])
                continue

            # The levels changed
            if reader_level != past_level:
                # Save the past level
                writer.writerow(tuple(past_matches))
                # Reset
                past_level = reader_level
                past_matches = [reader_level]
                past_matches.append(row['SW Level'])
                continue

            # Same levels
            past_matches.append(row['SW Level'])