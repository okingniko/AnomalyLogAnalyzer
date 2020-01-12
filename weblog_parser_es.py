#!/usr/bin/env python
# -*-coding: utf-8 -*-
import os
import sys
import time

sys.path.append("../")
from parser import *
from logreader import *

es_endpoint = 'http://47.96.231.21:9200'
index_name = 'njnet_access_mix'
output_dir = 'log_result/'  # The output directory of parsing results
log_format = '<IP> - - <Time> <Zone> <Content>'
tau = 0.5  # Message type threshold (default: 0.5)
regex = []  # Regular expression list for optional preprocessing (default: [])

if __name__ == '__main__':
    es_reader = ElasticReader(es_endpoint)
    print("current pid", os.getpid())
    # time.sleep(20)
    print("begin parse index {}, time: {}".format(index_name,
                                                  time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time.time()))))
    parser = LogParser(es_reader=es_reader, outdir=output_dir, log_format=log_format, tau=tau, rex=regex)
    parser.parse(index_name)
    print("finish parse file {}, time: {}".format(index_name,
                                                  time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time.time()))))