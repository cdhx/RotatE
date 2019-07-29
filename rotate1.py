# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:11:23 2019

@author: pc
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import json
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import KGEModel
from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator

'''
transE,没有自对抗负采样，cuda,日志，三种模式,checkpoint
'''
DATA_PATH = r'C:\Users\pc\Desktop\编程\KnowledgeGraphEmbedding-master\data\FB15k'
SAVE_PATH = r'C:\Users\pc\Desktop\编程\KnowledgeGraphEmbedding-master\save_model'
INIT_CHECKPOINT=r'C:\Users\pc\Desktop\编程\KnowledgeGraphEmbedding-master\save_model'
MODEL='TransE'
HIDDEN_DIM=200
gamma=12.0
BATCH_SIZE=1024
LR=0.001
SAVE_CHECKPOINT=10
def obj_dic(d):
    top = type('new', (object,), d)
    seqs = tuple, list, set, frozenset
    for i, j in d.items():
    	if isinstance(j, dict):
    	    setattr(top, i, obj_dic(j))
    	elif isinstance(j, seqs):
    	    setattr(top, i, 
    		    type(j)(obj_dic(sj) if isinstance(sj, dict) else sj for sj in j))
    	else:
    	    setattr(top, i, j)
    return top
def save_model(model,optimizer,save_variable_list,args):
    #save hyper para
    arg_par=vars(arg)
    with open(os.path.join(SAVE_PATH,'config.json'),'w')as fjson:
        json.dump(arg_par,fjson)
    #save model        
    torch.save({
        **save_variable_list,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        },
        os.path.join(arg.SAVE_PATH,'checkpoint')
    )
    entity_embedding=model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(arg.SAVE_PATH,'entity_embedding'),
        entity_embedding
    )
    relation_embedding=model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(arg.SAVE_PATH,'relation_embedding'),
        relation_embedding
    )
    
def read_triple(file_path,entity2id,relation2id):
    triples=[]
    with open(file_path) as fin:
        for line in fin:
            h,r,t=line.strip().split('\t')
            triples.append((entity2id[h],relation2id[r],entity2id[t]))
    return triples

def log_metrics(mode,step,metrics):
    '''
    Print evaluation message 
    '''
    for metric in metric:
        logging.info('%s %s at step %d:%f'%(mode,metric,step,metrics[metric]))

def main(arg):
    
    with open(r'C:\Users\pc\Desktop\编程\KnowledgeGraphEmbedding-master\data\FB15k\entities.dict') as fin:
        entity2id = dict()
        for line in fin:
            eid,entity=line.strip().split('\t')
            entity2id[entity] = int(eid)
    with open(r'C:\Users\pc\Desktop\编程\KnowledgeGraphEmbedding-master\data\FB15k\relations.dict') as fin:
        relation2id = dict()
        for line in fin:
            rid,relation=line.strip().split('\t')
            relation2id[relation] = int(rid)
    nentity = len(entity2id)
    nrelation = len(relation2id)                  
                    
    arg.nentity=nentity
    arg.nrelation=nrelation
    
    logging.info('Model:%s'%arg.MODEL)
    logging.info('Data Path:%s'%arg.DATA_PATH)
    logging.info('entity:%d'%arg.nentity)
    logging.info('relation:%d'%arg.nrelation)
    
    #extract data from file
    train_triples=read_triple(os.path.join(arg.DATA_PATH,'train.txt'), entity2id, relation2id)
    logging.info('#train:%d'%len(train_triples))
    valid_triples=read_triple(os.path.join(arg.DATA_PATH,'valid.txt'), entity2id, relation2id)
    logging.info('#valid:%d'%len(valid_triples))
    test_triples=read_triple(os.path.join(arg.DATA_PATH,'test.txt'), entity2id, relation2id)
    logging.info('#test:%d'%len(test_triples))
    #all true triples
    all_true_triples=train_triples+valid_triples+test_triples
    
    #construct model
    kge_model = KGEModel(
        model_name=arg.MODEL,
        nentity=arg.nentity,
        nrelation=arg.nrelation,
        hidden_dim=arg.HIDDEN_DIM,
        gamma=arg.gamma,
        double_entity_embedding=arg.double_entity_embedding,
        double_relation_embedding=arg.double_relation_embedding
    )
    
    #print model para configuration
    logging.info('Model Parameter Configuration')
    for name,para in kge_model.named_parameters():
        #print(name,para.size(),para.requires_grad
        logging.info('Parameter %s:%s,require_grad=%s'%(name,str(para.size()),str(para.requires_grad)))
        
    #do train
    train_dataloader_head=DataLoader(
        TrainDataset(train_triples,nentity,nrelation,arg.negative_sample_size,'head-bath'),
        batch_size=arg.BATCH_SIZE,
        shuffle=True,
        num_workers=max(1,arg.cpu_num//2),
        collate_fn=TrainDataset.collate_fn
    )
    train_dataloader_tail=DataLoader(
        TrainDataset(train_triples,nentity,nrelation,arg.negative_sample_size,'tail-batch'),
        batch_size=arg.BATCH_SIZE,
        shuffle=True,
        num_workers=max(1,arg.cpu_num//2),
        collate_fn=TrainDataset.collate_fn
    )
    
    train_iterator = BidirectionalOneShotIterator(train_dataloader_head,train_dataloader_tail)
    
    #set train configuration
    current_learning_rate = arg.LR
    optimizer = torch.optim.Adam(
        filter(lambda p:p.requires_grad,kge_model.parameters()),
        lr=current_learning_rate
    )
    
    
    warm_up_steps=arg.warm_up_steps if arg.warm_up_steps else arg.max_steps//2
    init_step=0
    step=init_step
    
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('learning_rate = %d' % current_learning_rate)
    logging.info('batch_size = %d' % arg.BATCH_SIZE)
    #logging.info('negative_adversarial_sampling = %d' % arg.negative_adversarial_sampling'])
    logging.info('hidden_dim = %d' % arg.HIDDEN_DIM)
    logging.info('gamma = %f' % arg.gamma)
    #logging.info('negative_adversarial_sampling = %s' % str(arg.negative_adversarial_sampling']))
  
    #start training
    training_logs=[]
    
    for step in range(init_step,arg.max_steps):
        log=kge_model.train_step(kge_model,optimizer,train_iterator,arg)
        training_logs.append(log)
        #update warm-up-step
        if step >= warm_up_steps:#大于warm_up_steps后学习率变为原来的1/10
            current_learning_rate = current_learning_rate / 10
            logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, kge_model.parameters()),
                lr=current_learning_rate#更新优化器里的学习率
            )
            warm_up_steps = warm_up_steps * 3#更新warm_up_steps
        #save model
        if step % arg.save_checkpoint_steps == 0:
            save_variable_list = {
                'step': step,
                'current_learning_rate': current_learning_rate,
                'warm_up_steps': warm_up_steps
            }
            save_model(kge_model, optimizer, save_variable_list, arg)
    #save after last time
    save_variable_list = {
        'step': step,
        'current_learning_rate': current_learning_rate,
        'warm_up_steps': warm_up_steps
    }
    save_model(kge_model, optimizer, save_variable_list, args)
if __name__ == '__main__':
    
    arg_par={}
    arg_par['DATA_PATH']= r'C:\Users\pc\Desktop\编程\KnowledgeGraphEmbedding-master\data\FB15k'
    arg_par['SAVE_PATH']= r'C:\Users\pc\Desktop\编程\KnowledgeGraphEmbedding-master\save_model'
    arg_par['INIT_CHECKPOINT']=r'C:\Users\pc\Desktop\编程\KnowledgeGraphEmbedding-master\save_model'
    arg_par['MODEL']='TransE'
    arg_par['HIDDEN_DIM']=200
    arg_par['gamma']=12.0
    arg_par['BATCH_SIZE']=1024
    arg_par['LR']=0.001
    arg_par['SAVE_CHECKPOINT']=10
    arg_par['cpu_num']=4
    arg_par['warm_up_steps']=0
    arg_par['max_steps']=100
    arg_par['negative_sample_size']=False
    arg_par['double_relation_embedding']=False
    arg_par['double_entity_embedding']=False
    arg_par['save_checkpoint_steps']=10
    arg_par['cuda']=torch.cuda.is_available()
    
    arg_par['adversarial_temperature']=1.0
    arg_par['uni_weight']=False
    arg_par['regularization']=0.0
    arg_par['countries']=False
    arg=obj_dic(arg_par)#转换成下标访问，但还不是parser
    main(arg)