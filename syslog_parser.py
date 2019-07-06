#!/usr/bin/env python
# -*-coding: utf-8 -*-
import os
import sys
import time

from parser import *

sys.path.append("../")

input_dir = 'data/logs/'  # The input directory of log file
output_dir = 'log_result/'  # The output directory of parsing results
log_file = 'auth.log'  # The input log file name
log_format = '<Month> <Date> <Time> <Level> <ComponentAndPID>: <Content>'
tau = 0.5  # Message type threshold (default: 0.5)
regex = []  # Regular expression list for optional preprocessing (default: [])
log_file_list = ['auth.log']

if __name__ == '__main__':
    print("current pid", os.getpid())
    for log_file in log_file_list:
        print("begin parse file {}, time: {}".format(log_file,
                                                     time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time.time()))))
        parser = LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex)
        parser.parse(log_file)
        # parser.simple_lcs_parse(log_file)
        print("finish parse file {}, time: {}".format(log_file,
                                                      time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time.time()))))
