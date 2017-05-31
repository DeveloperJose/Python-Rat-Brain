# -*- coding: utf-8 -*-
import feature
import thread

im_region = feature.nissl_load(34)

thread_match = thread.MatchingThread()
thread_match.set_im(im_region)
thread_match.run()