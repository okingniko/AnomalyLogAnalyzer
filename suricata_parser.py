#!/usr/bin/env python
# -*-coding: utf-8 -*-
import sys

sys.path.append("../")
from parser import *

input_dir = 'data/logs/'  # The input directory of log file
output_dir = 'log_result/'  # The output directory of parsing results
log_file = 'suricata.log'  # The input log file name
log_format = '<Date> -- <Time> - <Level> - <Content>'
tau = 0.5  # Message type threshold (default: 0.5)
regex = []  # Regular expression list for optional preprocessing (default: [])

if __name__ == '__main__':
    parser = LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex)
    parser.parse(log_file)
