import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import copy
from functools import partial
from dgl.nn.pytorch.conv import RelGraphConv
from basemodel import GraphAdjModel
from utils import map_activation_str_to_layer, split_and_batchify_graph_feats,GetAdj


class GIN(torch.nn.Module):
    def __init__(self, config):
        super(GIN, self).__init__()

        self.act=torch.nn.ReLU()
        self.g_net, self.bns, g_dim = self.create_net(
            name="graph", input_dim=config["node_feature_dim"], hidden_dim=config["gcn_hidden_dim"],
            num_layers=config["gcn_graph_num_layers"], num_bases=config["gcn_num_bases"], regularizer=config["gcn_regularizer"])
        self.num_layers_num=config["gcn_graph_num_layers"]
        self.dropout=torch.nn.Dropout(p=config["dropout"])

    def create_net(self, name, input_dim, **kw):
        num_layers = kw.get("num_layers", 1)
        hidden_dim = kw.get("hidden_dim", 64)
        num_rels = kw.get("num_rels", 1)
        num_bases = kw.get("num_bases", 8)
        regularizer = kw.get("regularizer", "basis")
        dropout = kw.get("dropout", 0.5)


        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_layers):

            if i:
                nn = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), self.act, torch.nn.Linear(hidden_dim, hidden_dim))
            else:
                nn = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), self.act, torch.nn.Linear(hidden_dim, hidden_dim))
            conv = dgl.nn.pytorch.conv.GINConv(apply_func=nn,aggregator_type='sum')
            bn = torch.nn.BatchNorm1d(hidden_dim)

            self.convs.append(conv)
            self.bns.append(bn)

        return self.convs, self.bns, hidden_dim


    def forward(self, graph, graph_len):
        graph_output = graph.ndata["feature"]
        xs = []
        for i in range(self.num_layers_num):
            graph_output = F.relu(self.convs[i](graph,graph_output))
            graph_output = self.bns[i](graph_output)
            graph_output = self.dropout(graph_output)
            xs.append(graph_output)
        xpool= []
        for x in xs:
            graph_embedding = split_and_batchify_graph_feats(x, graph_len)[0]
            graph_embedding = torch.sum(graph_embedding, dim=1)
            xpool.append(graph_embedding)
        x = torch.cat(xpool, -1)
        return x,torch.cat(xs, -1)
