#!/usr/bin/env python
# -*-coding: utf-8 -*-
import sys
import os
import time

sys.path.append("../")
from parser import *

input_dir = 'data/logs/'  # The input directory of log file
output_dir = 'log_result/'  # The output directory of parsing results
log_file = 'auth.log'  # The input log file name
log_format = '<Month> <Date> <Time> <Level> <ComponentAndPID>: <Content>'
tau = 0.5  # Message type threshold (default: 0.5)
regex = []  # Regular expression list for optional preprocessing (default: [])
# log_file_list = ['perf_2k.log', 'perf_5k.log', 'perf_1w.log', 'perf_2w.log', 'perf_5w.log', 'perf_10w.log',
#                  'perf_20w.log', 'perf_50w.log', 'perf_100w.log']
# log_file_list = ['auth_mix.log']
log_file_list = ['syslog1.log']

# nohup top -pid 88060 -i 1 > perf_parse.txt 2>&1 &

if __name__ == '__main__':
    print("current pid", os.getpid())
    # time.sleep(20)
    for log_file in log_file_list:
        print("begin parse file {}, time: {}".format(log_file, time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time.time()))))
        parser = LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex)
        parser.parse(log_file)
        # parser.simple_lcs_parse(log_file)
        print("finish parse file {}, time: {}".format(log_file, time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time.time()))))
