#!/usr/bin/env python
# -*-coding: utf-8 -*-

import pandas as pd

out_file = 'log_result/auth_mix_online.log_structured.csv'
in_file_list = 'log_result/auth_mix.log_structured.csv'
data_instance_file = 'ssh_data_instances.csv'


def split_file():
    id_map = dict()
    tmp_id_map = dict()

    with open(in_file_list, "r") as inf:
        with open(out_file, "w") as outf:
            outf.write(inf.readline())
            for line in inf:
                cid = line.split(",")[5]
                if cid in id_map:
                    continue
                else:
                    tmp_id_map[cid] = 0

                tmp_id_map = {k: v + 1 for k, v in tmp_id_map.items() if v < 5}
                print(tmp_id_map)
                for k in tmp_id_map.keys():
                    if tmp_id_map[k] >= 5:
                        id_map[k] = True
                print("tmp_id_map: ", len(tmp_id_map), ", id_map: ", len(id_map))
                outf.write(line)


def cal_unique():
    data_df = pd.read_csv(data_instance_file, engine='c', na_filter=False, memory_map=True, converters={"EventSequence": eval})
    x_data = data_df['EventSequence'].values
    print(x_data[0])
    uniq_dict = dict()
    for event in x_data:
        contents = ",".join(event)
        uniq_dict[contents] = True
    # with open(data_instance_file, "r") as inf:
    #     inf.readline()
    #     for line in inf:
    #         message = eval(line.split(",")[1])
    #         print(message)
    print("Has unique: ", len(uniq_dict))

if __name__ == '__main__':
    cal_unique()
