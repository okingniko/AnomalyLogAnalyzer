#!/usr/bin/env python
# -*-coding: utf-8 -*-

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Document
from elasticsearch_dsl import Search


class ElasticReader:
    def __init__(self, es_endpoint):
        self.client = Elasticsearch(es_endpoint)

    # can use helpers.bulk to accelerate
    def put_mock_log(self, index_name):
        with open('../data/logs/njnert_logs/njnet_access_mix.log') as f:
            for no, line in enumerate(f):
                d = Document(message=line)
                d.save(using=self.client, index=index_name)
                if no % 1000 == 0:
                    print('put {} rows'.format(no))

    def matchall(self, index_name, size=5000):
        begin_index = -1
        while True:
            body = {
                "size": size,
                "query": {
                    "match_all": {
                    }
                },
                "search_after": [begin_index],
                "sort": ['_doc']
            }
            s = Search.from_dict(body).using(self.client).index(index_name)
            response = s.execute()
            for hit in response:
                yield hit.message
            if len(response) == 0:
                break
            else:
                begin_index = response[-1].meta.sort[0]


if __name__ == '__main__':
    es_reader = ElasticReader('http://47.96.231.21:9200')
    # es_reader.put_mock_log('njnet_access_mix')
    contents = es_reader.matchall("njnet_access_mix")
    for idx, line in enumerate(contents):
        # print(line)
        if idx % 10000 == 0:
            print(idx)
