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
from sklearn.metrics import precision_recall_fscore_support
import functools
import dgl


def igraph_node_feature2dgl_node_feature(label,input):
    a=input
    b = a.split('[')[1]
    c = b.split(']')[0]
    d = c.split('\\n\', ')
    fix=d[len(d)-1]
    fix=fix.split('\\')[0]
    d[len(d)-1]=fix
    nodefeaturestring = []
    for nodefeature in d:
        temp = nodefeature.split('\'')[1]
        temp = temp.split(',')
        nodefeaturestring.append(temp)

    numbers_float = []
    for num in nodefeaturestring:
        temp = []
        for data in num:
            temp.append(float(data))
        numbers_float.append(temp)
    return label,numbers_float

def FUCK_U_IGraphLoad(path,graph_attr_num):
    with open(path, "r") as f:
        data=f.readlines()
        count=0
        for line in data:
            gattr=line.split()
            if gattr[0]=="label":
                label=int(gattr[1])
                count+=1
            if gattr[0]=="feature":
                feature=line.split("feature")[1]
                count+=1
            if count==graph_attr_num:
                return label, feature

def FUCK_IGraphLoad(path,graph_attr_num):
    label,feature=FUCK_U_IGraphLoad(path,graph_attr_num)
    return igraph_node_feature2dgl_node_feature(label,feature)

def ReSetNodeId(startid,edgelist):
    count=0
    for edge in edgelist:
        src,dst=edge
        src+=startid
        dst+=startid
        edgelist[count]=(src,dst)
        count+=1
    return edgelist


def _read_graphs_from_dir(dirpath):
    import igraph as ig
    graph = ig.Graph()
    count=0
    for filename in os.listdir(dirpath):
        if not os.path.isdir(os.path.join(dirpath, filename)):
            names = os.path.splitext(os.path.basename(filename))
            if names[1] != ".gml":
                continue
            try:
                if count==0:
                    _graph = ig.read(os.path.join(dirpath, filename))
                    label,feature=FUCK_IGraphLoad(os.path.join(dirpath, filename),2)
                    _graph.vs["label"] = [int(x) for x in _graph.vs["label"]]
                    _graph.es["label"] = [int(x) for x in _graph.es["label"]]
                    _graph.es["key"] = [int(x) for x in _graph.es["key"]]
                    _graph["feature"]=feature
                    graph=_graph
                    count+=1
                else:
                    _graph = ig.read(os.path.join(dirpath, filename))
                    label,feature=FUCK_IGraphLoad(os.path.join(dirpath, filename),2)
                    _graph.vs["label"] = [int(x) for x in _graph.vs["label"]]
                    _graph.es["label"] = [int(x) for x in _graph.es["label"]]
                    _graph.es["key"] = [int(x) for x in _graph.es["key"]]
                    _graph["feature"]=feature
                    _graph_nodelabel=_graph.vs["label"]
                    graph_nodelabel=graph.vs["label"]
                    new_nodelabel=graph_nodelabel+_graph_nodelabel
                    _graph_edgelabel=_graph.es["label"]
                    graph_edgelabel=graph.es["label"]
                    new_edgelabel=graph_edgelabel+_graph_edgelabel
                    _graph_edgekey=_graph.es["key"]
                    graph_edgekey=graph.es["key"]
                    new_edgekey=graph_edgekey+_graph_edgekey

                    graph_nodenum=graph.vcount()
                    _graph_nodenum=_graph.vcount()
                    graph.add_vertices(_graph_nodenum)
                    _graphedge=_graph.get_edgelist()
                    _graphedge=ReSetNodeId(graph_nodenum,_graphedge)
                    graph.add_edges(_graphedge)
                    graph.vs["label"]=new_nodelabel
                    graph.es["label"]=new_edgelabel
                    graph.es["key"]=new_edgekey
                    graph["feature"]=graph["feature"]+_graph["feature"]

            except BaseException as e:
                print(e)
                break
    return graph

def graph2dglgraph(graph):
    dglgraph = dgl.DGLGraph(multigraph=True)
    dglgraph.add_nodes(graph.vcount())
    edges = graph.get_edgelist()
    dglgraph.add_edges([e[0] for e in edges], [e[1] for e in edges])
    dglgraph.readonly(True)
    return dglgraph

def dglpreprocess(x):
    graph = x
    graph_dglgraph = graph2dglgraph(graph)
    graph_dglgraph.ndata["indeg"] = torch.tensor(np.array(graph.indegree(), dtype=np.float32))
    graph_dglgraph.ndata["label"] = torch.tensor(np.array(graph.vs["label"], dtype=np.int64))
    graph_dglgraph.ndata["id"] = torch.tensor(np.arange(0, graph.vcount(), dtype=np.int64))
    nodefeature=graph["feature"]
    graph_dglgraph.ndata["feature"]=torch.tensor(np.array(nodefeature, dtype=np.float32))
    return graph_dglgraph

def read_graphs_from_dir(dirpath):
    import igraph as ig
    ret=[]
    for filename in os.listdir(dirpath):
        if not os.path.isdir(os.path.join(dirpath, filename)):
            names = os.path.splitext(os.path.basename(filename))
            if names[1] != ".gml":
                continue
            try:
                graph = ig.read(os.path.join(dirpath, filename))
                label,feature=FUCK_IGraphLoad(os.path.join(dirpath, filename),2)
                graph.vs["label"] = [int(x) for x in graph.vs["label"]]
                graph.es["label"] = [int(x) for x in graph.es["label"]]
                graph.es["key"] = [int(x) for x in graph.es["key"]]
                graph["label"]=label
                graph["feature"]=feature
                ret.append(graph)
            except BaseException as e:
                print(e)
                break
    return ret



if __name__ == "__main__":
    assert len(sys.argv) == 2
    nci1_data_path = sys.argv[1]
    save_path="../data/ENZYMES/test_allinone"
    graph=_read_graphs_from_dir(nci1_data_path)
    dglgraph=dglpreprocess(graph)
    dgl.data.utils.save_graphs(os.path.join(save_path,"graph"),dglgraph)
    g=dgl.load_graphs(os.path.join(save_path,"graph"))[0][0]
    print(g)
    print(g.number_of_nodes())

def Raw2OneGraph(raw_data,save_data):
    nci1_data_path = raw_data
    save_path=save_data
    graphs=read_graphs_from_dir(nci1_data_path)
    count=0
    for graph in graphs:
        print("process graph ",count)
        dglgraph=dglpreprocess(graph)
        if countlabelnum(dglgraph)!=1:
            dgl.data.utils.save_graphs(os.path.join(save_path,str(count)),dglgraph)
            count+=1
    return count

def countlabelnum(graph):
    count=torch.zeros(3)
    for i in graph.ndata["label"]:
        count[i]=1
    return count.count_nonzero()