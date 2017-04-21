# -*- coding: utf-8 -*-
import atexit
from time import clock
from datetime import timedelta

def secondsToStr(t):
    return str(timedelta(seconds=t))

line = "="*40

def log(s, elapsed=None):
    print(line)
    print(secondsToStr(clock()), '-', s)
    if elapsed:
        print("Elapsed time:", elapsed)
    print(line)
    print()

stopwatch_start = None
def stopwatch(message="Stopwatch: "):
    global stopwatch_start
    if (stopwatch_start is None):
        stopwatch_start = clock()
    else:

        elapsed = clock() - stopwatch_start
        log(message, secondsToStr(elapsed))
        stopwatch_start = None

def endlog():
    end = clock()
    elapsed = end-start
    log("End Program", secondsToStr(elapsed))

def now():
    return secondsToStr(clock())

start = clock()
atexit.register(endlog)
log("Start Program")