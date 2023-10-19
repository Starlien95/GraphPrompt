import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import re
import os
import sys
import json
from torch.optim.lr_scheduler import LambdaLR
from collections import OrderedDict
from multiprocessing import Pool
from tqdm import tqdm
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
import random
from tqdm import trange
from sklearn.metrics import precision_recall_fscore_support
import functools
import dgl

#drop==True means drop nodes of class drop when split train,val, test;but can only drop the biggest class(ex 0,1,2 can only drop label 2)
def few_shot_split_nodelevel(graph,tasknum,trainshot,valshot,labelnum,seed=0, drop=False):
    train=[]
    val=[]
    test=[]
    if drop:
        labelnum=labelnum-1
    nodenum=graph.number_of_nodes()
    random.seed(seed)
    for count in range(tasknum):
        index = random.sample(range(0, nodenum), nodenum)
        trainindex=[]
        valindex=[]
        testindex=[]
        traincount = torch.zeros(labelnum)
        valcount = torch.zeros(labelnum)
        for i in index:
            label=graph.ndata["label"][i]
            if drop:
                if label==labelnum:
                    continue
            if traincount[label]<trainshot:
                trainindex.append(i)
                traincount[label]+=1
            elif valcount[label]<valshot:
                valcount[label]+=1
                valindex.append(i)
            else:
                testindex.append(i)
        train.append(trainindex)
        val.append(valindex)
        test.append(testindex)
    return train,val,test
train_config={
    "trainshot": 10,
    "valshot": 4,
    "labelnum": 3,
    "tasknum": 10,
    "save_data_dir": "../data/PROTEINS/all",
    "graph_num":245,
    "seed":0,
    "drop": False
}

def count(config):
    total_node=0
    total_edge=0
    for num in trange(config["graph_num"]):
        save_path=os.path.join(config["save_data_dir"],str(num))
        graph=dgl.load_graphs(save_path)[0][0]
        node_num=graph.number_of_nodes()
        total_node+=node_num
        total_edge+=graph.number_of_edges()
    print("avg node:",total_node/train_config["graph_num"])
    print("avg edge:",total_edge/train_config["graph_num"])

if __name__ == "__main__":
    count(train_config)





