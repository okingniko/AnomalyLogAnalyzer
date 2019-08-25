#!/usr/bin/env python
# -*-coding: utf-8 -*-
import os
import sys
import time

sys.path.append("../")
from parser import *

input_dir = 'data/logs/'  # The input directory of log file
output_dir = 'log_result/'  # The output directory of parsing results
log_file = 'HDFS.log'  # The input log file name
log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
tau = 0.5  # Message type threshold (default: 0.5)
regex = []  # Regular expression list for optional preprocessing (default: [])

if __name__ == '__main__':
    print("current pid", os.getpid())
    # time.sleep(20)

    print("begin parse file {}, time: {}".format(log_file,
                                                 time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time.time()))))
    parser = LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex)
    parser.parse(log_file)
    # parser.simple_lcs_parse(log_file)
    print("finish parse file {}, time: {}".format(log_file,
                                                  time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time.time()))))
