# -*- coding: utf-8 -*-
# Author: Jose G Perez
# This code is not part of the Rat Brain program
#
# Representation of Knots in SL(2, Z)
# This tool uses bruteforce to find matrices such that
# A^2 = B^6n+1 for any n {1, 7, 13, 19, 25}
# A^2 = B^6n+3 for any n {3, 9, 15, 21, 27}
# A^2 = B^6n+5 for any n {5, 11, 17, 23, 29}
# Depending on what you select

import numpy as np
import itertools
import multiprocessing, multiprocessing.pool
import timing

import sys
import config
import logbook
logger = logbook.Logger(__name__)
logbook.StreamHandler(sys.stdout, level=logbook.DEBUG, format_string=config.LOGGER_FORMAT_STRING).push_application()

""" ===== Settings """
# Range of numbers that will be in the matrices
RANGE_NUM = range(-3, 3)

# Series of powers to use for checking
SERIES = [2*n + 1 for n in range(5)]
#SERIES = [6*n + 1 for n in range(3)]
#SERIES = [6*n + 3 for n in range(4)]
#SERIES = [6*n + 5 for n in range(4)]
#SERIES = [6*n + 1 for n in range(5)] + [6*n + 3 for n in range(5)] + [6*n + 5 for n in range(5)]

""" ===== Constants """
IDENTITY = np.identity(2, np.int32)

""" ===== Debugging """
ITERATION = 0
PROCESSORS = multiprocessing.cpu_count()
RESULTS = list()

def function(A, B):
    global SERIES
    # A^2
    A = np.linalg.matrix_power(A, 2)

    # A^2 is the zero matrix
    if not A.any():
        return False

    # Series selection
    # B^6n+1 {1, 7, 13, 19, 25}
    # B^6n+3
    # B^6n+5
    for power in SERIES:
        test = np.linalg.matrix_power(B, power)
        if not np.array_equal(A, test):
            return False

    return True

def process(iter_array):
    global ITERATION
    global RESULTS
    ITERATION += 1

    # Convert from itertool product to numpy 2x2 array
    array = np.fromiter(iter_array, np.int32, 8)
    A = np.reshape(array[:4], (2, 2))
    B = np.reshape(array[4:], (2, 2))

    # Ignore redundant repetitions
    if np.array_equal(A, B):
        return

    # Arrays of zeroes are ignored
    if not A.any() or not B.any():
        return

    # Check if any of them is the identity matrix
    if np.array_equal(A, IDENTITY) or np.array_equal(B, IDENTITY):
        return

    # Also ignore negative identity matrices
    if np.array_equal(A, -IDENTITY) or np.array_equal(B, -IDENTITY):
        return

    if function(A, B):
        RESULTS.append((A, B))
        # Converted to list to print in a single line
        result = ''' Successful with
                    A={}, Det={}, Trace={}
                    B={}, Det={}, Trace={}
                    A^2={}
                    [6n+1]B^7={}, B^13={}
                    [6n+3]B^9={}, B^15={}
                    [6n+5]B^11={}, B^17={}'''
        logger.debug(result, A.tolist(), np.linalg.det(A), np.trace(A),
                     B.tolist(), np.linalg.det(B), np.trace(B),
                     np.linalg.matrix_power(A, 2).tolist(),
                     np.linalg.matrix_power(B, 7).tolist(),
                     np.linalg.matrix_power(B, 13).tolist(),
                     np.linalg.matrix_power(B, 9).tolist(),
                     np.linalg.matrix_power(B, 15).tolist(),
                     np.linalg.matrix_power(B, 11).tolist(),
                     np.linalg.matrix_power(B, 17).tolist())

# Generate all 2x2 array combinations in a given range
# Output Example: (0,0,0,0,0,0,0,0)
arrays = list(itertools.product(RANGE_NUM, repeat=8))

# Run all operations using multithreading
pool = multiprocessing.pool.ThreadPool(PROCESSORS)

timing.stopwatch()
pool.map(process, arrays)
timing.stopwatch("Finished Mapping")

logger.debug("Total Iterations: {}", ITERATION)

# Clean-Up
pool.close()