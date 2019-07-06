#!/usr/bin/env python
# -*-coding: utf-8 -*-

import re
import os
import pandas as pd
from datetime import datetime


class LCSObject:
    """ Class object to store a log group with the same template
    """

    def __init__(self, log_template, log_idl=list()):
        self.log_template = log_template
        self.log_idl = log_idl


class Node:
    """ A node in prefix tree data structure
    """

    def __init__(self, token='', template_no=0):
        self.log_cluster = None
        self.token = token
        self.template_no = template_no
        self.childD = dict()


class LogParser:
    """ LogParser class

    Attributes
    ----------
        path : the path of the input file
        log_name : the file name of the input file
        save_path : the path of the output file
        tau : how much percentage of tokens matched to merge a log message
    """

    def __init__(self, indir='./', outdir='./result/', log_format=None, tau=0.5, rex=list()):
        self.path = indir
        self.log_name = None
        self.save_path = outdir
        self.tau = tau
        self.log_format = log_format
        self.df_log = None
        self.rex = rex

    def lcs(self, seq1, seq2):
        lengths = [[0 for j in range(len(seq2) + 1)] for i in range(len(seq1) + 1)]
        # row 0 and column 0 are initialized to 0 already
        for i in range(len(seq1)):
            for j in range(len(seq2)):
                if seq1[i] == seq2[j]:
                    lengths[i + 1][j + 1] = lengths[i][j] + 1
                else:
                    lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])

        # read the substring out from the matrix
        result = []
        seq1_len, seq2_len = len(seq1), len(seq2)
        while seq1_len != 0 and seq2_len != 0:
            if lengths[seq1_len][seq2_len] == lengths[seq1_len - 1][seq2_len]:
                seq1_len -= 1
            elif lengths[seq1_len][seq2_len] == lengths[seq1_len][seq2_len - 1]:
                seq2_len -= 1
            else:
                assert seq1[seq1_len - 1] == seq2[seq2_len - 1]
                result.insert(0, seq1[seq1_len - 1])
                seq1_len -= 1
                seq2_len -= 1
        return result

    def simple_loop_match(self, log_cluster_list, seq):
        retlog_cluster = None

        for log_cluster in log_cluster_list:
            if float(len(log_cluster.log_template)) < 0.5 * len(seq):
                continue

            # If the template is a subsequence of seq
            if all(token in seq or token == '*' for token in log_cluster.log_template):
                return log_cluster

        return retlog_cluster

    def prefix_tree_match(self, parentn, seq, idx):
        retlog_cluster = None
        length = len(seq)
        for i in range(idx, length):
            if seq[i] in parentn.childD:
                childn = parentn.childD[seq[i]]
                if childn.log_cluster is not None:
                    const_lm = [w for w in childn.log_cluster.log_template if w != '*']
                    if float(len(const_lm)) >= self.tau * length:
                        return childn.log_cluster
                else:
                    return self.prefix_tree_match(childn, seq, i + 1)

        return retlog_cluster

    def lcs_match(self, log_cluster_list, seq):
        retlog_cluster = None

        max_len = -1
        # max_lcs = []
        max_cluster = None
        set_seq = set(seq)
        size_seq = len(seq)
        for log_cluster in log_cluster_list:
            set_template = set(log_cluster.log_template)
            if len(set_seq & set_template) < 0.5 * size_seq:
                continue
            lcs = self.lcs(seq, log_cluster.log_template)
            if len(lcs) > max_len or (
                    len(lcs) == max_len and len(log_cluster.log_template) < len(max_cluster.log_template)):
                max_len = len(lcs)
                # max_lcs = lcs
                max_cluster = log_cluster

        # lcs should be large then tau * len(itself)
        if float(max_len) >= self.tau * size_seq:
            retlog_cluster = max_cluster

        return retlog_cluster

    def get_template(self, lcs, seq):
        ret_val = []
        if not lcs:
            return ret_val

        lcs = lcs[::-1]
        i = 0
        for token in seq:
            i += 1
            if token == lcs[-1]:
                ret_val.append(token)
                lcs.pop()
            else:
                ret_val.append('*')
            if not lcs:
                break
        if i < len(seq):
            ret_val.append('*')
        return ret_val

    def add_seq_to_prefix_tree(self, rootn, new_cluster):
        parentn = rootn
        seq = new_cluster.log_template
        seq = [w for w in seq if w != '*']

        for i in range(len(seq)):
            token_in_seq = seq[i]
            # Match
            if token_in_seq in parentn.childD:
                parentn.childD[token_in_seq].template_no += 1
                # Do not Match
            else:
                parentn.childD[token_in_seq] = Node(token=token_in_seq, template_no=1)
            parentn = parentn.childD[token_in_seq]

        if parentn.log_cluster is None:
            parentn.log_cluster = new_cluster

    def remove_seq_from_prefix_tree(self, rootn, new_cluster):
        parentn = rootn
        seq = new_cluster.log_template
        seq = [w for w in seq if w != '*']

        for token_in_seq in seq:
            if token_in_seq in parentn.childD:
                matched_node = parentn.childD[token_in_seq]
                if matched_node.template_no == 1:
                    del parentn.childD[token_in_seq]
                    break
                else:
                    matched_node.template_no -= 1
                    parentn = matched_node

    def output_result(self, log_cluster_list):

        templates = [0] * self.df_log.shape[0]
        ids = [0] * self.df_log.shape[0]
        df_event = []

        for idx, log_cluster in enumerate(log_cluster_list):
            template_str = ' '.join(log_cluster.log_template)
            # eid = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]
            eid = "Event" + str(idx)
            for log_id in log_cluster.log_idl:
                templates[log_id - 1] = template_str
                ids[log_id - 1] = eid
            df_event.append([eid, template_str, len(log_cluster.log_idl)])

        df_event = pd.DataFrame(df_event, columns=['EventId', 'EventTemplate', 'Occurrences'])

        self.df_log['EventId'] = ids
        self.df_log['EventTemplate'] = templates
        self.df_log.to_csv(os.path.join(self.save_path, self.log_name + '_structured.csv'), index=False)
        df_event.to_csv(os.path.join(self.save_path, self.log_name + '_templates.csv'), index=False)

    def print_tree(self, node, dep):
        p_str = ''
        for i in range(dep):
            p_str += '\t'

        if node.token == '':
            p_str += 'Root'
        else:
            p_str += node.token
            if node.log_cluster is not None:
                p_str += '-->' + ' '.join(node.log_cluster.log_template)
        print(p_str + ' (' + str(node.template_no) + ')')

        for child in node.childD:
            self.print_tree(node.childD[child], dep + 1)

    def parse(self, log_name):
        start_time = datetime.now()
        print('Parsing file: ' + os.path.join(self.path, log_name))
        self.log_name = log_name
        self.load_data()
        root_node = Node()
        log_cluster_list = []  # lcsMap: 存放所有已经parsed出的LCSObjects.

        count = 0
        for idx, line in self.df_log.iterrows():
            log_id = line['LineId']
            # log_msg_list = list(filter(lambda x: x != '', re.split(r'[\s=:,]', self.preprocess(line['Content']))))
            log_msg_list = self.preprocess(line['Content']).strip().split()
            const_log_msg_list = [w for w in log_msg_list if w != '*']
            # print("const_log_msg_list", const_log_msg_list)
            # Find an existing matched log cluster
            match_cluster = self.prefix_tree_match(root_node, const_log_msg_list, 0)

            if match_cluster is None:
                match_cluster = self.simple_loop_match(log_cluster_list, const_log_msg_list)

                if match_cluster is None:
                    match_cluster = self.lcs_match(log_cluster_list, log_msg_list)

                    # Match no existing log cluster
                    if match_cluster is None:
                        new_cluster = LCSObject(log_template=log_msg_list, log_idl=[log_id])
                        log_cluster_list.append(new_cluster)
                        self.add_seq_to_prefix_tree(root_node, new_cluster)
                    # Add the new log message to the existing cluster
                    else:
                        new_template = self.get_template(self.lcs(log_msg_list, match_cluster.log_template),
                                                         match_cluster.log_template)
                        if ' '.join(new_template) != ' '.join(match_cluster.log_template):
                            self.remove_seq_from_prefix_tree(root_node, match_cluster)
                            match_cluster.log_template = new_template
                            self.add_seq_to_prefix_tree(root_node, match_cluster)
            if match_cluster:
                match_cluster.log_idl.append(log_id)
            count += 1
            if count % 1000 == 0 or count == len(self.df_log):
                print('Processed {0:.1f}% of log lines.'.format(count * 100.0 / len(self.df_log)))

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.output_result(log_cluster_list)
        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))

    def simple_lcs_parse(self, log_name):
        start_time = datetime.now()
        print('Parsing file: ' + os.path.join(self.path, log_name))
        self.log_name = log_name
        self.load_data()
        log_cluster_list = []  # lcsMap: 存放所有已经parsed出的LCSObjects.

        count = 0
        for idx, line in self.df_log.iterrows():
            log_id = line['LineId']
            log_msg_list = self.preprocess(line['Content']).strip().split()
            # Find an existing matched log cluster
            match_cluster = self.lcs_match(log_cluster_list, log_msg_list)

            if match_cluster is None:
                new_cluster = LCSObject(log_template=log_msg_list, log_idl=[log_id])
                log_cluster_list.append(new_cluster)
            # Add the new log message to the existing cluster
            else:
                new_template = self.get_template(self.lcs(log_msg_list, match_cluster.log_template),
                                                 match_cluster.log_template)
                if ' '.join(new_template) != ' '.join(match_cluster.log_template):
                    match_cluster.log_template = new_template
            if match_cluster:
                match_cluster.log_idl.append(log_id)
            count += 1
            if count % 1000 == 0 or count == len(self.df_log):
                print('Processed {0:.1f}% of log lines.'.format(count * 100.0 / len(self.df_log)))

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.output_result(log_cluster_list)
        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))

    def load_data(self):
        headers, regex = self.generate_log_format_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.log_name), regex, headers, self.log_format)

    def preprocess(self, line):
        for currentRex in self.rex:
            line = re.sub(currentRex, '*', line)
        return line

    def log_to_dataframe(self, log_file, regex, headers, log_format):
        """ Function to transform log file to dataframe
        """
        log_messages = []
        linecount = 0
        with open(log_file, 'r') as fin:
            for line in fin.readlines():
                # print('line:', line)
                line = re.sub(r'[^\x00-\x7F]+', '<NASCII>', line)
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    # print(message)
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        # print(logdf)
        return logdf

    def generate_log_format_regex(self, log_format):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', log_format)
        # print(splitters)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(r' +', r'\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        # print(headers, regex)
        return headers, regex
