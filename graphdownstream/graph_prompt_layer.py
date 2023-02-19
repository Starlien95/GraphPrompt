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

class graph_prompt_layer_mean(nn.Module):
    def __init__(self):
        super(graph_prompt_layer_mean, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(2, 2))
    def forward(self, graph_embedding, graph_len):
        graph_embedding=split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        graph_prompt_result=graph_embedding.mean(dim=1)
        return graph_prompt_result

class graph_prompt_layer_linear_mean(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(graph_prompt_layer_linear_mean, self).__init__()
        self.linear=torch.nn.Linear(input_dim,output_dim)

    def forward(self, graph_embedding, graph_len):
        graph_embedding=self.linear(graph_embedding)

        graph_embedding=split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        graph_prompt_result=graph_embedding.mean(dim=1)
        graph_prompt_result=torch.nn.functional.normalize(graph_prompt_result,dim=1)
        return graph_prompt_result

class graph_prompt_layer_linear_sum(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(graph_prompt_layer_linear_sum, self).__init__()
        self.linear=torch.nn.Linear(input_dim,output_dim)

    def forward(self, graph_embedding, graph_len):
        graph_embedding=self.linear(graph_embedding)

        graph_embedding=split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        graph_prompt_result=graph_embedding.sum(dim=1)
        graph_prompt_result=torch.nn.functional.normalize(graph_prompt_result,dim=1)
        return graph_prompt_result



#sum result is same as mean result
class graph_prompt_layer_sum(nn.Module):
    def __init__(self):
        super(graph_prompt_layer_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(2, 2))
    def forward(self, graph_embedding, graph_len):
        graph_embedding=split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        graph_prompt_result=graph_embedding.sum(dim=1)
        return graph_prompt_result



class graph_prompt_layer_weighted(nn.Module):
    def __init__(self,max_n_num):
        super(graph_prompt_layer_weighted, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,max_n_num))
        self.max_n_num=max_n_num
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding, graph_len):
        graph_embedding=split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        weight = self.weight[0][0:graph_embedding.size(1)]
        temp1 = torch.ones(graph_embedding.size(0), graph_embedding.size(2), graph_embedding.size(1)).to(graph_embedding.device)
        temp1 = weight * temp1
        temp1 = temp1.permute(0, 2, 1)
        graph_embedding=graph_embedding*temp1
        graph_prompt_result=graph_embedding.sum(dim=1)
        return graph_prompt_result

class graph_prompt_layer_feature_weighted_mean(nn.Module):
    def __init__(self,input_dim):
        super(graph_prompt_layer_feature_weighted_mean, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.max_n_num=input_dim
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding, graph_len):
        graph_embedding=split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        graph_embedding=graph_embedding*self.weight
        graph_prompt_result=graph_embedding.mean(dim=1)
        return graph_prompt_result

class graph_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,input_dim):
        super(graph_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.max_n_num=input_dim
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding, graph_len):
        graph_embedding=split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        graph_embedding=graph_embedding*self.weight
        graph_prompt_result=graph_embedding.sum(dim=1)
        return graph_prompt_result

class graph_prompt_layer_weighted_matrix(nn.Module):
    def __init__(self,max_n_num,input_dim):
        super(graph_prompt_layer_weighted_matrix, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(input_dim,max_n_num))
        self.max_n_num=max_n_num
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding, graph_len):
        graph_embedding=split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        weight = self.weight.permute(1, 0)[0:graph_embedding.size(1)]
        weight = weight.expand(graph_embedding.size(0), weight.size(0), weight.size(1))
        graph_embedding = graph_embedding * weight
        #prompt: mean
        graph_prompt_result=graph_embedding.sum(dim=1)
        return graph_prompt_result

class graph_prompt_layer_weighted_linear(nn.Module):
    def __init__(self,max_n_num,input_dim,output_dim):
        super(graph_prompt_layer_weighted_linear, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,max_n_num))
        self.linear=nn.Linear(input_dim,output_dim)
        self.max_n_num=max_n_num
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding, graph_len):
        graph_embedding=self.linear(graph_embedding)
        graph_embedding=split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        weight = self.weight[0][0:graph_embedding.size(1)]
        temp1 = torch.ones(graph_embedding.size(0), graph_embedding.size(2), graph_embedding.size(1)).to(graph_embedding.device)
        temp1 = weight * temp1
        temp1 = temp1.permute(0, 2, 1)
        graph_embedding=graph_embedding*temp1
        graph_prompt_result = graph_embedding.mean(dim=1)
        return graph_prompt_result

class graph_prompt_layer_weighted_matrix_linear(nn.Module):
    def __init__(self,max_n_num,input_dim,output_dim):
        super(graph_prompt_layer_weighted_matrix_linear, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(output_dim,max_n_num))
        self.linear=nn.Linear(input_dim,output_dim)
        self.max_n_num=max_n_num
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding, graph_len):
        graph_embedding=self.linear(graph_embedding)
        graph_embedding=split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        weight = self.weight.permute(1, 0)[0:graph_embedding.size(1)]
        weight = weight.expand(graph_embedding.size(0), weight.size(0), weight.size(1))
        graph_embedding = graph_embedding * weight
        graph_prompt_result=graph_embedding.mean(dim=1)
        return graph_prompt_result
