# -*- coding: utf-8 -*-
# Representation of Knots in SL(2, Z)
import numpy as np
import itertools
import multiprocessing
import timing

import sys
import config
import logbook
logger = logbook.Logger(__name__)
logbook.StreamHandler(sys.stdout, level=logbook.DEBUG, format_string=config.LOGGER_FORMAT_STRING).push_application()

# Settings
ITERATION = 0
PROCESSORS = multiprocessing.cpu_count()
RESULTS = list()

def function(array1, array2):
    # A^2
    array1 = np.linalg.matrix_power(array1, 2)

    # Series selection
    # B^6n+1 {1, 7, 13, 19, 25}
    # B^6n+3
    # B^6n+5
    series = [6*n + 1 for n in range(4)]
    for power in series:
        test = np.linalg.matrix_power(array2, power)
        if not np.array_equal(array1, test):
            return False

    return True

def process(iter_array):
    global ITERATION
    global RESULTS
    ITERATION += 1

    # Convert from itertool product to numpy 2x2 array
    array = np.fromiter(iter_array, np.uint32, 8)
    array1 = np.reshape(array[:4], (2, 2))
    array2 = np.reshape(array[4:], (2, 2))

    if np.array_equal(array1, array2):
        return

    if function(array1, array2):
        # Converted to list to print in a single line
        logger.debug("Successful with {0} and {1}", array1.tolist(), array2.tolist())
        RESULTS.append((array1, array2))


# Generate all 2x2 array combinations in a given range
# Output: (0,0,0,0,0,0,0,0)
arrays = itertools.product(range(3), repeat=8)

# Run all operations using multithreading
pool = multiprocessing.pool.ThreadPool(PROCESSORS)

timing.stopwatch()
pool.map(process, arrays)
timing.stopwatch("Finished Mapping")

# Clean-Up
pool.close()