# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:00:39 2019

@author: pc
"""

"""
看一下数据长什么样，用notepad也可以看
permission denied: 读不了数据，也没有占用这些数据
"""

from torch.utils.data import Dataset
import torch
import os
file_path=DATA_PATH=r'C:\Users\pc\Desktop\KnowledgeGraphEmbedding-master\data\FB15k-237'
with open(os.path.join(DATA_PATH, 'entities.dict')) as fin:
    entity2id = dict()
    for line in fin:
        eid, entity = line.strip().split('\t')
        entity2id[entity] = int(eid)

with open(os.path.join(DATA_PATH, 'relations.dict')) as fin:
    relation2id = dict()
    for line in fin:
        rid, relation = line.strip().split('\t')
        relation2id[relation] = int(rid)
def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples
read_triple(file_path,entity2id,relation2id)        

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        
    def __len__(self):
        return self.len
