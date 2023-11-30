import torch
import os
import numpy as np
import dgl
import logging
import datetime
import math
import sys
import gc
import json
import time
import torch.nn.functional as F
import warnings
import itertools
from functools import partial
from collections import OrderedDict
from torch.utils.data import DataLoader
from dgllife.utils import ConsecutiveSplitter



try:
    from torch.utils.tensorboard import SummaryWriter
except BaseException as e:
    from tensorboardX import SummaryWriter
from dataset import Sampler, EdgeSeqDataset, GraphAdjDataset, GraphAdjDataset_DGL_Input
from utils import anneal_fn, get_enc_len, pretrain_load_data, get_linear_schedule_with_warmup,\
    bp_compute_abmae,compareloss,label2onehot,correctness_GPU,correctness,macrof1,weightf1,few_shot_split_graphlevel,\
    distance2center,center_embedding,index2mask,mask_select_emb
'''from cnn import CNN
from rnn import RNN
from txl import TXL
from rgcn import RGCN
from rgin import RGIN
from gat import GAT
from gcn import GCN
from graphsage import GraphSage
from gin import GIN'''
# from gat import GAT
# from gcn_onehot import GCN
from gin import GIN
# from gcn import GCN
# from gat import GAT
# from graphsage import Graphsage
# from graphsage import Graphsage
from node_prompt_layer import node_prompt_layer_linear_mean,node_prompt_layer_linear_sum,\
    node_prompt_layer_feature_weighted_mean,node_prompt_layer_feature_weighted_sum,node_prompt_layer_sum

warnings.filterwarnings("ignore")
INF = float("inf")

train_config = {
    "max_npv": 8,  # max_number_pattern_vertices: 8, 16, 32
    "max_npe": 8,  # max_number_pattern_edges: 8, 16, 32
    "max_npvl": 8,  # max_number_pattern_vertex_labels: 8, 16, 32
    "max_npel": 8,  # max_number_pattern_edge_labels: 8, 16, 32

    "max_ngv": 64,  # max_number_graph_vertices: 64, 512,4096
    "max_nge": 256,  # max_number_graph_edges: 256, 2048, 16384
    "max_ngvl": 16,  # max_number_graph_vertex_labels: 16, 64, 256
    "max_ngel": 16,  # max_number_graph_edge_labels: 16, 64, 256

    "base": 2,

    "gpu_id": -1,
    "num_workers": 12,

    "epochs": 200,
    "batch_size": 512,
    "update_every": 1,  # actual batch_sizer = batch_size * update_every
    "print_every": 100,
    "init_emb": "Equivariant",  # None, Orthogonal, Normal, Equivariant
    "share_emb": True,  # sharing embedding requires the same vector length
    "share_arch": True,  # sharing architectures
    "dropout": 0.2,
    "dropatt": 0.2,

    "reg_loss": "MSE",  # MAE, MSEl
    "bp_loss": "MSE",  # MAE, MSE
    "bp_loss_slp": "anneal_cosine$1.0$0.01",  # 0, 0.01, logistic$1.0$0.01, linear$1.0$0.01, cosine$1.0$0.01,
    # cyclical_logistic$1.0$0.01, cyclical_linear$1.0$0.01, cyclical_cosine$1.0$0.01
    # anneal_logistic$1.0$0.01, anneal_linear$1.0$0.01, anneal_cosine$1.0$0.01
    "lr": 0.001,
    "weight_decay": 0.00001,
    "max_grad_norm": 8,

    "pretrain_model": "GCN",

    "emb_dim": 128,
    "activation_function": "leaky_relu",  # sigmoid, softmax, tanh, relu, leaky_relu, prelu, gelu

    "filter_net": "MaxGatedFilterNet",  # None, MaxGatedFilterNet
    "predict_net": "SumPredictNet",  # MeanPredictNet, SumPredictNet, MaxPredictNet,
    "predict_net_add_enc": True,
    "predict_net_add_degree": True,

    # MeanAttnPredictNet, SumAttnPredictNet, MaxAttnPredictNet,
    # MeanMemAttnPredictNet, SumMemAttnPredictNet, MaxMemAttnPredictNet,
    # DIAMNet
    # "predict_net_add_enc": True,
    # "predict_net_add_degree": True,
    "txl_graph_num_layers": 3,
    "txl_pattern_num_layers": 3,
    "txl_d_model": 128,
    "txl_d_inner": 128,
    "txl_n_head": 4,
    "txl_d_head": 4,
    "txl_pre_lnorm": True,
    "txl_tgt_len": 64,
    "txl_ext_len": 0,  # useless in current settings
    "txl_mem_len": 64,
    "txl_clamp_len": -1,  # max positional embedding index
    "txl_attn_type": 0,  # 0 for Dai et al, 1 for Shaw et al, 2 for Vaswani et al, 3 for Al Rfou et al.
    "txl_same_len": False,

    "gcn_num_bases": 8,
    "gcn_regularizer": "bdd",  # basis, bdd
    "gcn_graph_num_layers": 3,
    "gcn_hidden_dim": 128,
    "gcn_ignore_norm": False,  # ignorm=True -> RGCN-SUM

    "graph_dir": "../data/debug/graphs",
    "save_data_dir": "../data/debug",
    "save_model_dir": "../dumps/debug",
    "save_pretrain_model_dir": "../dumps/MUTAGPreTrain/GCN",
    "graphslabel_dir":"../data/debug/graphs",
    "downstream_graph_dir": "../data/debug/graphs",
    "downstream_save_data_dir": "../data/debug",
    "downstream_save_model_dir": "../dumps/debug",
    "downstream_graphslabel_dir":"../data/debug/graphs",
    "temperature": 0.01,
    "graph_finetuning_input_dim": 8,
    "graph_finetuning_output_dim": 2,
    "graph_label_num":2,
    "seed": 0,
    "update_pretrain": False,
    "dropout": 0.5,
    "gcn_output_dim": 8,

    "prompt": "SUM",
    "prompt_output_dim": 2,
    "scalar": 1e3,

    "dataset_seed": 0,
    "train_shotnum": 10,
    "val_shotnum": 5,
    "few_shot_tasknum": 5,

    "save_fewshot_dir": "../data/MUTAGGraphClassification/fewshot",

    "downstream_dropout": 0,
    "node_feature_dim": 18,
    "train_label_num": 6,
    "val_label_num": 6,
    "test_label_num": 6,
    "nhop_neighbour": 1
}

def pre_train(model, graph, device, config,self_weight = None):
    epoch_step = 1
    total_step = config["epochs"] * epoch_step
    total_reg_loss = 0
    total_bp_loss = 0
    #total_cnt = 1e-6


    model.eval()
    total_time=0
    graph=graph.to(device)
    graph_len=torch.tensor([[89250]], device=device)
    graph_len = graph_len.to(device)
    s=time.time()
    x,pred = model(graph, graph_len)
    #########################################################################
    ############为了解决在NCI1上loss出现0的情况###################################
    #需要约束embedding为>0的值，可以通过加一个可以通过relu，sigmoid实现
    pred=F.sigmoid(pred)

    #修改后的计算过程
    adj = graph.adjacency_matrix()
    size_ = adj.size(0)
    p = [i for i in range(size_)]
    x = torch.tensor([p,p])
    q = [self_weight for i in range(size_)]
    tt = torch.sparse_coo_tensor(x,q,(size_,size_))
    adj = (adj + tt).to(device)
    '''print('---------------------------------------------------')
    print('adj: ',adj.size())
    print('pred: ',adj.size())
    print('---------------------------------------------------')'''
    if config["nhop_neighbour"]==0:
        pred=pred
    else:
        for i in range(config["nhop_neighbour"]):
            pred = torch.matmul(adj, pred)
    return pred
from fvcore.nn import FlopCountAnalysis, parameter_count_table


#pretrain_embedding是该部分需要的embedding；node_label也是该部分节点对应的nodelabel
def train(model, optimizer, scheduler, data_type, device, config, epoch, label_num, pretrain_embedding, node_label, logger=None, writer=None):
    epoch_step = 1
    total_step = config["epochs"] * epoch_step
    total_reg_loss = 0
    total_bp_loss = 0
    batchcnt = 0
    total_acc=0
    total_cnt = 1e-6

    if config["reg_loss"] == "MAE":
        reg_crit = lambda pred, target: F.l1_loss(F.relu(pred), target)
    elif config["reg_loss"] == "MSE":

        reg_crit = lambda pred, target: F.mse_loss(F.relu(pred), target)
    elif config["reg_loss"] == "SMSE":
        reg_crit = lambda pred, target: F.smooth_l1_loss(F.relu(pred), target)
    elif config["reg_loss"] == "NLL":
        reg_crit = lambda pred, target: F.nll_loss(F.relu(pred), target)
    elif config["reg_loss"] == "ABMAE":
        reg_crit = lambda pred, target: bp_compute_abmae(F.leaky_relu(pred), target) + 0.8 * F.l1_loss(F.relu(pred),
                                                                                                       target)
    else:
        raise NotImplementedError

    if config["bp_loss"] == "MAE":
        bp_crit = lambda pred, target, neg_slp: F.l1_loss(F.leaky_relu(pred, neg_slp), target)
    elif config["bp_loss"] == "MSE":
        bp_crit = lambda pred, target, neg_slp: F.mse_loss(F.leaky_relu(pred, neg_slp), target)
    elif config["bp_loss"] == "SMSE":
        
        bp_crit = lambda pred, target, neg_slp: F.smooth_l1_loss(F.leaky_relu(pred, neg_slp), target)
    elif config["bp_loss"] == "NLL":
        bp_crit = lambda pred, target,neg_slp: F.nll_loss(pred, target)
    elif config["bp_loss"]=="CROSS":
        bp_crit = lambda pred, target, neg_slp: F.cross_entropy(pred, target)
    elif config["bp_loss"] == "ABMAE":
        bp_crit = lambda pred, target, neg_slp: bp_compute_abmae(F.leaky_relu(pred, neg_slp), target) + 0.8 * F.l1_loss(
            F.leaky_relu(pred, neg_slp), target, reduce="none")
    else:
        raise NotImplementedError

    model.train()
    total_time = 0
    label_num=torch.tensor(label_num).to(device)
    #for batch_id, batch in enumerate(data_loader):
    # print(batch)
    batchcnt+=1
    s = time.time()

    embedding=model(pretrain_embedding,0)

    node_label=node_label
    c_embedding=center_embedding(embedding,node_label,label_num)
    distance=distance2center(embedding,c_embedding)
    #print(distance)

    distance = 1/F.normalize(distance, dim=1)

    #distance=distance2center2(embedding,c_embedding)

    #print('distance: ',distance )
    pred=F.log_softmax(distance,dim=1)
    #----------------------------------------
    #reg_loss = reg_crit(pred, graph_label_onehot)
    #对NLL LOSS用这个公式，否则用上面的
    reg_loss = reg_crit(pred, node_label.squeeze().type(torch.LongTensor).to(device))
    #------------------------------------------------
    reg_loss.requires_grad_(True)
    _pred = torch.argmax(pred, dim=1, keepdim=True)
    accuracy = correctness_GPU(_pred, node_label)
    total_acc+=accuracy
    if isinstance(config["bp_loss_slp"], (int, float)):
        neg_slp = float(config["bp_loss_slp"])
    else:
        bp_loss_slp, l0, l1 = config["bp_loss_slp"].rsplit("$", 3)
        neg_slp = anneal_fn(bp_loss_slp, 0 + epoch * epoch_step, T=total_step // 4, lambda0=float(l0),
                            lambda1=float(l1))

    #bp_loss = bp_crit(pred.float(), graph_label_onehot.float(), neg_slp)
    #对NLL LOSS用这个公式，否则用上面的
    bp_loss = bp_crit(pred.float(), node_label.squeeze().type(torch.LongTensor).to(device),neg_slp)
    bp_loss.requires_grad_(True)

    # float
    reg_loss_item = reg_loss.item()
    bp_loss_item = bp_loss.item()
    total_reg_loss += reg_loss_item
    total_bp_loss += bp_loss_item

    if writer:
        writer.add_scalar("%s/REG-%s" % (data_type, config["reg_loss"]), reg_loss_item,
                          epoch * epoch_step + 0)
        writer.add_scalar("%s/BP-%s" % (data_type, config["bp_loss"]), bp_loss_item, epoch * epoch_step + 0)

    # if logger and (0 % config["print_every"] == 0 or 0 == epoch_step - 1):
    #     logger.info(
    #         "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:0>5d}/{:0>5d}\treg loss: {:0>5.8f}\tbp loss: {:0>5.8f}".format(
    #             epoch, config["epochs"], data_type, 0, epoch_step,
    #             reg_loss_item, bp_loss_item))
    bp_loss.backward(retain_graph=True)
    '''for name, parms in model.named_parameters():
        print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
              ' -->grad_value:', parms.grad)'''
    '''for name, parms in model.named_parameters():
        print('-->name:', name, ' -->value:', parms)'''
    if (config["update_every"] < 2 or 0 % config["update_every"] == 0 or 0 == epoch_step - 1):
        if config["max_grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        if scheduler is not None:
            scheduler.step(epoch * epoch_step + 0)
        optimizer.step()
        optimizer.zero_grad()
    e=time.time()
    total_time+=e-s

    # mean_reg_loss = total_reg_loss / total_cnt
    # mean_bp_loss = total_bp_loss / total_cnt
    # mean_acc=total_acc/batchcnt
    if writer:
        writer.add_scalar("%s/REG-%s-epoch" % (data_type, config["reg_loss"]), reg_loss.item(), epoch)
        writer.add_scalar("%s/BP-%s-epoch" % (data_type, config["bp_loss"]), bp_loss.item(), epoch)
    # if logger:
    #     #logger.info("epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\treg loss: {:0>5.8f}\tbp loss: {:0>5.8f}\tmean_acc: {:0>1.3f}".format(
    #         #epoch, config["epochs"], data_type, reg_loss.item(), bp_loss.item(),accuracy))
    #         pass
    gc.collect()
    return reg_loss, bp_loss, total_time,accuracy,c_embedding


def evaluate(model, data_type, device, config, epoch, c_embedding, label_num, pretrain_embedding, node_label, debug=False,logger=None, writer=None):
    epoch_step = 1
    total_reg_loss = 0
    total_step = config["epochs"] * epoch_step
    total_bp_loss = 0
    batchcnt=0
    total_acc=0
    total_macrof=0
    total_weighted=0
    total_cnt = 1e-6

    evaluate_results = {"data": {"id": list(), "counts": list(), "pred": list()},
                        "error": {"mae": INF, "mse": INF},
                        "time": {"avg": list(), "total": 0.0}}

    if config["reg_loss"] == "MAE":
        reg_crit = lambda pred, target: F.l1_loss(F.relu(pred), target, reduce="none")
    elif config["reg_loss"] == "MSE":
        reg_crit = lambda pred, target: F.mse_loss(F.relu(pred), target, reduce="none")
    elif config["reg_loss"] == "SMSE":
        reg_crit = lambda pred, target: F.smooth_l1_loss(F.relu(pred), target, reduce="none")
    elif config["reg_loss"] == "NLL":
        reg_crit = lambda pred, target: F.nll_loss(F.relu(pred), target)
    elif config["reg_loss"] == "ABMAE":
        reg_crit = lambda pred, target: bp_compute_abmae(F.relu(pred), target) + 0.8 * F.l1_loss(F.relu(pred), target,
                                                                                                 reduce="none")
    else:
        raise NotImplementedError

    if config["bp_loss"] == "MAE":
        bp_crit = lambda pred, target, neg_slp: F.l1_loss(F.leaky_relu(pred, neg_slp), target, reduce="none")
    elif config["bp_loss"] == "MSE":
        bp_crit = lambda pred, target, neg_slp: F.mse_loss(F.leaky_relu(pred, neg_slp), target, reduce="none")
    elif config["bp_loss"] == "SMSE":
        bp_crit = lambda pred, target, neg_slp: F.smooth_l1_loss(F.leaky_relu(pred, neg_slp), target, reduce="none")
    elif config["bp_loss"] == "NLL":
        bp_crit = lambda pred, target,neg_slp: F.nll_loss(pred, target)
    elif config["bp_loss"] == "CROSS":
        bp_crit = lambda pred, target, neg_slp: F.cross_entropy(pred, target)
    elif config["bp_loss"] == "ABMAE":
        bp_crit = lambda pred, target, neg_slp: bp_compute_abmae(F.leaky_relu(pred, neg_slp), target) + 0.8 * F.l1_loss(
            F.leaky_relu(pred, neg_slp), target, reduce="none")
    else:
        raise NotImplementedError

    model.eval()
    l2onehot=label2onehot(train_config["graph_label_num"],device)
    label_num=torch.tensor(label_num).to(device)
    total_time = 0
    batchcnt+=1

    s = time.time()
    # if debug:
    #     print("####################")
    #     print("pretrain embedding:",embedding)
    embedding = model(pretrain_embedding, 0)*train_config["scalar"]
    #print('embedding:', embedding)
    #c_embedding = center_embedding(embedding, graph_label, label_num)
    #print('c_embedding:', c_embedding)
    #print('distance:', distance)
    node_label=node_label
    c_embedding = center_embedding(embedding, node_label, label_num,debug)

    distance=distance2center(embedding,c_embedding)
    distance=-1*F.normalize(distance,dim=1)
    # if debug:
    #     print("embedding:",embedding)
    #     print("c_embedding:",c_embedding)
    #     print("distance:",distance)
    #distance = distance2center2(embedding, c_embedding)

    pred=F.log_softmax(distance,dim=1)
    # if debug:
    #     print("pred:",pred)

    #reg_loss = reg_crit(pred, graph_label_onehot)
    # 对NLL LOSS用这个公式，否则用上面的
    reg_loss = reg_crit(pred, node_label.squeeze().type(torch.LongTensor).to(device))

    if isinstance(config["bp_loss_slp"], (int, float)):
        neg_slp = float(config["bp_loss_slp"])
    else:
        bp_loss_slp, l0, l1 = config["bp_loss_slp"].rsplit("$", 3)
        # neg_slp = anneal_fn(bp_loss_slp, 0 + epoch * epoch_step, T=total_step // 4, lambda0=float(l0),
        #                     lambda1=float(l1))
        neg_slp=0.2
    #bp_loss = bp_crit(pred.float(), graph_label_onehot.float(), neg_slp)
    # 对NLL LOSS用这个公式，否则用上面的
    bp_loss = bp_crit(pred, node_label.squeeze().type(torch.LongTensor).to(device), neg_slp)

    #graph_label_onehot=l2onehot(graph_label)
    _pred = torch.argmax(pred, dim=1, keepdim=True)
    accuracy = correctness_GPU(_pred, node_label)
    eval_pred=_pred.cpu().numpy()
    eval_graph_label=node_label.cpu().numpy()
    acc=correctness(eval_pred,eval_graph_label)
    macrof=macrof1(eval_pred,eval_graph_label)
    weightf=weightf1(eval_pred,eval_graph_label)
    total_acc+=acc
    total_macrof+=macrof
    total_weighted+=weightf


    # float
    reg_loss_item = reg_loss.item()
    bp_loss_item = bp_loss.item()
    #evaluate_results["time"]["total"] += (et-st)
    #avg_t = (et-st) / (cnt + 1e-8)
    #evaluate_results["time"]["avg"].extend([avg_t]*cnt)

    #evaluate_results["error"]["mae"] += F.l1_loss(F.leaky_relu(pred.float(),neg_slp), graph_label_onehot.float(), reduce="none").sum().item()
    #evaluate_results["error"]["mse"] += F.mse_loss(F.leaky_relu(pred.float(),neg_slp), graph_label_onehot.float(), reduce="none").sum().item()
    if writer:
        writer.add_scalar("%s/REG-%s" % (data_type, config["reg_loss"]), reg_loss_item,
                          epoch * epoch_step + 0)
        writer.add_scalar("%s/BP-%s" % (data_type, config["bp_loss"]), bp_loss_item,
                          epoch * epoch_step + 0)

    if logger and 0 == epoch_step - 1:
        pass
    if writer:
        writer.add_scalar("%s/REG-%s-epoch" % (data_type, config["reg_loss"]), reg_loss.item(), epoch)
        writer.add_scalar("%s/BP-%s-epoch" % (data_type, config["bp_loss"]), bp_loss.item(), epoch)
    # if logger:
    #     logger.info("epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\treg loss: {:0>5.8f}\tbp loss: {:0>5.8f}\tacc:{:0>1.3f}".format(
    #         epoch, config["epochs"], data_type, reg_loss_item, bp_loss_item,acc))

        #evaluate_results["error"]["mae"] = evaluate_results["error"]["mae"] / total_cnt
        #evaluate_results["error"]["mse"] = evaluate_results["error"]["mse"] / total_cnt
        #accuracy = correctness(evaluate_results["data"]["pred"], evaluate_results["data"]["counts"])
        #macrof=macrof1(evaluate_results["data"]["pred"],evaluate_results["data"]["counts"])
        #weightedf=weightf1(evaluate_results["data"]["pred"],evaluate_results["data"]["counts"])

    gc.collect()
    #return mean_reg_loss, mean_bp_loss, evaluate_results, total_time,mean_acc.cpu(),macrof,weightedf,c_embedding
    return mean_reg_loss, mean_bp_loss, evaluate_results, total_time,acc,macrof,weightf,c_embedding


if __name__ == "__main__":
    #adj_self_weight = [1,0,0.1,0.3,0.5,0.7,0.9,1,1.1,1.3,1.5,1.7,1.9,2]
    adj_self_weight = [1,0,0.1,0.3,0.5,0.7]
 

    record = {1:"1",0.1:"01",0.3:"03",0.5:"05",0.7:"07",0.9:"09",0:"0",1.1:"11",1.3:"13",1.5:"15",1.7:"17",1.9:"19",2:"2"}
    for i in range(1, len(sys.argv), 2):
        arg = sys.argv[i]
        value = sys.argv[i + 1]

        if arg.startswith("--"):
            arg = arg[2:]
        if arg not in train_config:
            # print("Warning: %s is not surported now." % (arg))
            continue
        train_config[arg] = value
        try:
            value = eval(value)
            if isinstance(value, (int, float)):
                train_config[arg] = value
        except:
            pass
    for weight in range(len(adj_self_weight)):
        for times in range(5):
            print("the times%d for weight%s"%(times,adj_self_weight[weight]))
            torch.set_printoptions(precision=10)

            torch.manual_seed(train_config["seed"])
            np.random.seed(train_config["seed"])

            ts = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            pretrain_model_name = "%s_%s_%s" % (train_config["pretrain_model"], train_config["predict_net"], ts)
            save_model_dir = train_config["downstream_save_model_dir"]
            save_pretrain_model_dir=train_config["save_pretrain_model_dir"]
            os.makedirs(save_model_dir, exist_ok=True)

            # save config
            with open(os.path.join(save_model_dir, "train_config.json"), "w") as f:
                json.dump(train_config, f)

            # set logger
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %H:%M:%S')
            console = logging.StreamHandler()
            console.setFormatter(fmt)
            logger.addHandler(console)
            logfile = logging.FileHandler(os.path.join(save_model_dir, "train_log.txt"), 'w')
            logfile.setFormatter(fmt)
            logger.addHandler(logfile)

            # set device
            device = torch.device("cuda:%d" % train_config["gpu_id"] if train_config["gpu_id"] != -1 else "cpu")
            if train_config["gpu_id"] != -1:
                torch.cuda.set_device(device)

            # reset the pattern parameters
            if train_config["share_emb"]:
                train_config["max_npv"], train_config["max_npvl"], train_config["max_npe"], train_config["max_npel"] = \
                    train_config["max_ngv"], train_config["max_ngvl"], train_config["max_nge"], train_config["max_ngel"]

            if train_config["pretrain_model"] == "GCN":
                pre_train_model = GCN(train_config)
            if train_config["pretrain_model"] == "GIN":
                pre_train_model = GIN(train_config)
            if train_config["pretrain_model"] == "GAT":
                pre_train_model = GAT(train_config)
            if train_config["pretrain_model"] == "GraphSage":
                pre_train_model = Graphsage(train_config)


            # 先把整张图读进来，得到embedding后再用实现划分好的index来mask embedding
            os.makedirs(train_config["save_data_dir"], exist_ok=True)



            #-------------start here-----------------------------------------------------------
            #把划分好的task保存下来，使得以后不需要再重新划分tasks
            fewshot_dir=os.path.join(train_config["save_fewshot_dir"],"%s_trainshot_%s_valshot_%s_tasks" %
                                    (train_config["train_shotnum"],train_config["val_shotnum"],train_config["few_shot_tasknum"]))
            print(os.path.exists(fewshot_dir))
            print("Load Few Shot")
            trainset = np.load(os.path.join(fewshot_dir, "train_dgl_dataset.npy"),allow_pickle=True)
            valset = np.load(os.path.join(fewshot_dir, "val_dgl_dataset.npy"),allow_pickle=True)
            testset = np.load(os.path.join(fewshot_dir, "test_dgl_dataset.npy"),allow_pickle=True)
            trainset=torch.tensor(trainset,dtype=int)
            valset=torch.tensor(valset,dtype=int)
            testset=torch.tensor(testset,dtype=int)
            graph_dir=os.path.join(train_config["graph_dir"],str(train_config["node_feature_dim"]),"graph")
            graph=dgl.load_graphs(graph_dir)[0][0]
            nodelabel=graph.ndata["label"]


            acc = list()
            macroF = list()
            weightedF = list()
            total_rec = list()
            for count in range(train_config["few_shot_tasknum"]):
                #graph = dgl.load_graphs(os.path.join(train_config["save_data_dir"], "graph"))[0][0]
                pre_train_model = pre_train_model.to(device)
                pre_train_model.load_state_dict(torch.load(os.path.join(save_pretrain_model_dir, "best_local_%s_times%d.pt"%(record[adj_self_weight[weight]],times))))
                pretrain_embedding = pre_train(pre_train_model, graph, device, train_config,self_weight=adj_self_weight[weight])
                # logger.info("num of pretrain parameters: %d" % (sum(p.numel() for p in pre_train_model.parameters()if p.requires_grad )))


     
                _trainset=trainset[count]
                _valset=valset[count]
                _testset=testset[count]

                trainmask = index2mask(_trainset, train_config["max_ngv"])
                valmask = index2mask(_valset, train_config["max_ngv"])
                testmask = index2mask(_testset, train_config["max_ngv"])
                nodelabel=nodelabel.to(device)
                pretrain_embedding=pretrain_embedding.to(device)
                trainmask,valmask,testmask=trainmask.to(device),valmask.to(device),testmask.to(device)
                trainlabel = torch.masked_select(nodelabel,torch.tensor(trainmask,dtype=bool)).unsqueeze(1)
                vallabel = torch.masked_select(nodelabel,torch.tensor(valmask,dtype=bool)).unsqueeze(1)
                testlabel = torch.masked_select(nodelabel,torch.tensor(testmask,dtype=bool)).unsqueeze(1)
                trainemb=mask_select_emb(pretrain_embedding,trainmask,device)
                valemb=mask_select_emb(pretrain_embedding,valmask,device)
                testemb=mask_select_emb(pretrain_embedding,testmask,device)

                #np.random.seed()
                #torch.random.seed()
                # if train_config["prompt"] == "MEAN":
                #     model = graph_prompt_layer_mean()
                if train_config["prompt"] == "SUM":
                    model = node_prompt_layer_sum()
                if train_config["prompt"] == "LINEAR-MEAN":
                    model = node_prompt_layer_linear_mean(
                        train_config["gcn_hidden_dim"] * train_config["gcn_graph_num_layers"],
                        train_config["prompt_output_dim"])
                if train_config["prompt"] == "LINEAR-SUM":
                    model = node_prompt_layer_linear_sum(train_config["gcn_hidden_dim"] * train_config["gcn_graph_num_layers"],
                                                        train_config["prompt_output_dim"])
                if train_config["prompt"] == "FEATURE-WEIGHTED-SUM":
                    model = node_prompt_layer_feature_weighted_sum(
                        train_config["gcn_hidden_dim"] * train_config["gcn_graph_num_layers"])
                if train_config["prompt"] == "FEATURE-WEIGHTED-MEAN":
                    model = node_prompt_layer_feature_weighted_mean(
                        train_config["gcn_hidden_dim"] * train_config["gcn_graph_num_layers"])
                model = model.to(device)
                # logger.info(model)
                # logger.info("num of parameters: %d" % (sum(p.numel() for p in model.parameters() if p.requires_grad)))
                # optimizer and losses
                # writer = SummaryWriter(save_model_dir)
                writer = None
                # if train_config["update_pretrain"]:
                #     optimizer = torch.optim.AdamW(itertools.chain(pre_train_model.parameters(), model.parameters()),
                #                                   lr=train_config["lr"],
                #                                   weight_decay=train_config["weight_decay"], amsgrad=True)
                #else:
                optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["lr"],
                                            weight_decay=train_config["weight_decay"], amsgrad=True)

                optimizer.zero_grad()
                scheduler = None
                # scheduler = get_linear_schedule_with_warmup(optimizer,
                #     len(data_loaders["train"]), train_config["epochs"]*len(data_loaders["train"]), min_percent=0.0001)
                best_bp_losses = {"train": INF, "dev": INF, "test": INF}
                best_bp_epochs = {"train": -1, "dev": -1, "test": -1}
                best_acc = {"train": -1, "dev": -1, "test": -1}

                total_train_time = 0
                total_dev_time = 0
                total_test_time = 0
                best_c_embedding = None
                c_embedding = None
                for epoch in range(train_config["epochs"]):
                    mean_reg_loss, mean_bp_loss, _time, accfold,c_embedding = train(model, optimizer, scheduler, "train", device,
                                                            train_config, epoch, train_config["train_label_num"], trainemb,trainlabel, logger=logger, writer=writer)
                    total_train_time += _time
                    torch.save(model.state_dict(), os.path.join(save_model_dir, 'epoch%d.pt' % (epoch)))
                    if train_config["update_pretrain"] == True:
                        torch.save(pre_train_model.state_dict(),os.path.join(save_model_dir, 'pretrain_epoch%d.pt' % (epoch)))
                    if accfold >= best_acc["train"] or mean_bp_loss <= best_bp_losses["train"]:
                        # if epoch_accuracy > best_acc[data_type]:
                        # if mean_bp_loss <= best_bp_losses[data_type]:
                        # ------------------------------------------------------------
                        # 这代码只针对or来选择时候生效
                        if accfold >= best_acc["train"]:
                            best_acc["train"] = accfold
                        if mean_bp_loss < best_bp_losses["train"]:
                            best_bp_losses["train"] = mean_bp_loss
                        # ------------------------------------------------------------
                        # best_bp_losses[data_type] = mean_bp_loss
                        # best_acc[data_type]=epoch_accuracy
                        best_bp_epochs["train"] = epoch
                        #logger.info(
                            #"data_type: {:<5s}\tbest mean loss: {:.3f}\t best acc: {:.3f}\t (epoch: {:0>3d})".format(
                                #"train", mean_bp_loss, accfold, epoch))



                    mean_reg_loss, mean_bp_loss, evaluate_results, _time,accfold, macroFfold, weightedFfold,c_embedding = \
                        evaluate(model, "val", device, train_config, epoch, c_embedding,
                                train_config["val_label_num"], valemb,vallabel,logger=logger,writer=writer)
                    total_dev_time += _time
                    with open(os.path.join(save_model_dir, '%s%d.json' % ("val", epoch)), "w") as f:
                        json.dump(evaluate_results, f)
                    if accfold >= best_acc["dev"] or mean_bp_loss <= best_bp_losses["dev"]:
                        # if epoch_accuracy > best_acc[data_type]:
                        # if mean_bp_loss <= best_bp_losses[data_type]:
                        # ------------------------------------------------------------
                        # 这代码只针对or来选择时候生效
                        if accfold >= best_acc["dev"]:
                            best_acc["dev"] = accfold
                        if mean_bp_loss < best_bp_losses["dev"]:
                            best_bp_losses["dev"] = mean_bp_loss
                        # ------------------------------------------------------------
                        best_c_embedding = c_embedding

                        # best_bp_losses[data_type] = mean_bp_loss
                        # best_acc[data_type]=epoch_accuracy
                        best_bp_epochs["dev"] = epoch
                        #logger.info(
                            #"data_type: {:<5s}\tbest mean loss: {:.3f}\t best acc: {:.3f}\t (epoch: {:0>3d})".format(
                                #"dev", mean_bp_loss, accfold, epoch))



                best_epoch=best_bp_epochs["dev"]
                data_loaders = OrderedDict({"test": None})
                data_loaders["test"]=testset[count]
                model.load_state_dict(torch.load(os.path.join(save_model_dir, 'epoch%d.pt' % (best_epoch))))
                if train_config["update_pretrain"] == True:
                    pre_train_model.load_state_dict(
                        torch.load(os.path.join(save_model_dir, 'pretrain_epoch%d.pt' % (best_epoch))))
                mean_reg_loss, mean_bp_loss, evaluate_results, _time, acctest, macroFtest, weightedFtest,c_embedding = \
                    evaluate(model, "test", device, train_config, epoch, best_c_embedding,
                                    train_config["val_label_num"], testemb,testlabel,logger=logger,writer=writer)


                acc.append(acctest)
                macroF.append(macroFtest)
                weightedF.append(weightedFtest)



                # for data_type in data_loaders.keys():
                #     logger.info(
                #         "data_type: {:<5s}\tbest mean loss: {:.3f} (epoch: {:0>3d})".format(data_type, best_bp_losses[data_type],
                #                                                                             best_bp_epochs[data_type]))

              
            acc=np.array(acc)
            macroF=np.array(macroF)
            weightedF=np.array(weightedF)

            total_rec.append('%sacc mean:'%(record[adj_self_weight[weight]]))
            total_rec.append(np.mean(acc))
            
            total_rec.append('%sacc std: '%(record[adj_self_weight[weight]]))
            total_rec.append(np.std(acc))
            total_rec.append('%sMF:'%(record[adj_self_weight[weight]]))
            total_rec.append(np.mean(macroF))
            
            total_rec.append('%sMFstd: '%(record[adj_self_weight[weight]]))
            total_rec.append(np.std(macroF))
            #np.savetxt('total_rec%s_enzymes.txt'%(record[adj_self_weight[weight]]),total_rec,fmt='%s')
            np.savetxt('total_rec_flickr_local%s_times%d.txt'%(record[adj_self_weight[weight]],times),total_rec,fmt='%s')

         
            # _________________________________________________________________________________________________________________

