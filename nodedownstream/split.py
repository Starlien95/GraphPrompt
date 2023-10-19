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
    "save_graph_path": "../data",
    "save_fewshot_path":"../data",
    "seed":0,
    "drop": False
}
if __name__ == "__main__":
    for i in range(1, len(sys.argv), 2):
        arg = sys.argv[i]
        value = sys.argv[i + 1]

        if arg.startswith("--"):
            arg = arg[2:]
        if arg not in train_config:
            print("Warning: %s is not surported now." % (arg))
            continue
        train_config[arg] = value
        try:
            value = eval(value)
            if isinstance(value, (int, float)):
                train_config[arg] = value
        except:
            pass
    # graph=_read_graphs_from_dir(nci1_data_path)
    # dglgraph=dglpreprocess(graph)
    # dgl.data.utils.save_graphs(os.path.join(save_path,"graph"),dglgraph)
    graph=dgl.load_graphs(train_config["save_graph_path"])[0][0]
    count=torch.zeros(train_config["labelnum"])
    for i in graph.ndata["label"]:
        count[i]+=1
    trainset,valset,testset=few_shot_split_nodelevel(graph,train_config["tasknum"],train_config["trainshot"],
                                                     train_config["valshot"],train_config["labelnum"],
                                                     train_config["seed"],train_config["drop"])
    trainset = np.array(trainset)
    valset = np.array(valset)
    testset = np.array(testset)
    fewshot_dir=os.path.join(train_config["save_fewshot_path"],"%s_trainshot_%s_valshot_%s_tasks" %
                             (train_config["trainshot"],train_config["valshot"],train_config["tasknum"]))
    if os.path.exists(train_config["save_fewshot_path"])!=True:
        os.mkdir(train_config["save_fewshot_path"])
    if os.path.exists(fewshot_dir)!=True:
        os.mkdir(fewshot_dir)

    np.save(os.path.join(fewshot_dir, "train_dgl_dataset"), trainset)
    np.save(os.path.join(fewshot_dir, "val_dgl_dataset"), valset)
    np.save(os.path.join(fewshot_dir, "test_dgl_dataset"), testset)


def split(config):
    for num in trange(config["graph_num"]):
        save_path=os.path.join(config["save_data_dir"],str(num))
        graph=dgl.load_graphs(save_path)[0][0]
        max_nlabel=graph.ndata["label"].max()
        trainset,valset,testset=few_shot_split_nodelevel(graph,config["few_shot_tasknum"],config["train_shotnum"],
                                                         config["val_shotnum"],max_nlabel+1,
                                                         config["seed"],config["split_drop"])
        trainset = np.array(trainset)
        valset = np.array(valset)
        testset = np.array(testset)
        fewshot_dir=os.path.join(config["save_fewshot_dir"],"%s_trainshot_%s_valshot_%s_tasks" %
                                 (config["train_shotnum"],config["val_shotnum"],config["few_shot_tasknum"]))
        if os.path.exists(config["save_fewshot_dir"])!=True:
            os.mkdir(config["save_fewshot_dir"])
        if os.path.exists(fewshot_dir)!=True:
            os.mkdir(fewshot_dir)
        temp=os.path.join(fewshot_dir,str(num))
        if os.path.exists(temp)!=True:
            os.mkdir(os.path.join(fewshot_dir,str(num)))
        np.save(os.path.join(fewshot_dir, str(num),"train_dgl_dataset"), trainset)
        np.save(os.path.join(fewshot_dir, str(num),"val_dgl_dataset"), valset)
        np.save(os.path.join(fewshot_dir, str(num),"test_dgl_dataset"), testset)




