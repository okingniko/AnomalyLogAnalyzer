#!/usr/bin/env python
# -*-coding: utf-8 -*-

# out_file = 'data/logs/auth_mix.log'
# in_file_list = ['data/logs/auth_normal.log', 'data/logs/auth_hydra.log']

out_file = 'data/logs/jslab_logs/jslab_access_mix.log'
in_file_list = ['data/logs/jslab_logs/jslab_access.log.1', 'data/logs/jslab_logs/jslab_access.log.2',
                'data/logs/jslab_logs/jslab_access.log.3', 'data/logs/jslab_logs/jslab_access.log.4',
                'data/logs/jslab_logs/jslab_access.log.5', 'data/logs/jslab_logs/jslab_access.log.6',
                'data/logs/jslab_logs/jslab_access.log.7', 'data/logs/jslab_logs/jslab_access.log.8',
                'data/logs/jslab_logs/jslab_access.log.9', 'data/logs/jslab_logs/jslab_access.log.10',
                'data/logs/jslab_logs/jslab_access.log.11', 'data/logs/jslab_logs/jslab_access.log.12',
                'data/logs/jslab_logs/jslab_access.log.13', 'data/logs/jslab_logs/jslab_access.log.14'
                ]

if __name__ == '__main__':
    with open(out_file, 'w') as outf:
        for in_file in in_file_list:
            with open(in_file, 'r') as inf:
                outf.writelines(inf.readlines())
