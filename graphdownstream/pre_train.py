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
from functools import partial
from collections import OrderedDict
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random


try:
    from torch.utils.tensorboard import SummaryWriter
except BaseException as e:
    from tensorboardX import SummaryWriter
from dataset import Sampler, EdgeSeqDataset, GraphAdjDataset
from utils import anneal_fn, get_enc_len, pretrain_load_data, \
    get_linear_schedule_with_warmup,bp_compute_abmae,compareloss,split_and_batchify_graph_feats
'''from cnn import CNN
from rnn import RNN
from txl import TXL
from rgcn import RGCN
from rgin import RGIN

from gin import GIN'''
from gin import GIN

warnings.filterwarnings("ignore")
INF = float("inf")

train_config = {
    "max_npv": 8,  # max_number_pattern_vertices: 8, 16, 32
    "max_npe": 8,  # max_number_pattern_edges: 8, 16, 32
    "max_npvl": 8,  # max_number_pattern_vertex_labels: 8, 16, 32
    "max_npel": 8,  # max_number_pattern_edge_labels: 8, 16, 32

    "max_ngv": 620,  # max_number_graph_vertices: 64, 512,4096
    "max_nge": 2098,  # max_number_graph_edges: 256, 2048, 16384
    "max_ngvl": 2,  # max_number_graph_vertex_labels: 16, 64, 256
    "max_ngel": 2,  # max_number_graph_edge_labels: 16, 64, 256

    "base": 2,

    "gpu_id": -1,
    "num_workers": 12,

    "epochs": 400,
    "batch_size": 1024,
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
    "lr": 0.1,
    "weight_decay": 0.00001,
    "max_grad_norm": 8,

    "model": "CNN",  # CNN, RNN, TXL, RGCN, RGIN, RSIN

    "predict_net": "SumPredictNet",  # MeanPredictNet, SumPredictNet, MaxPredictNet,
    # MeanAttnPredictNet, SumAttnPredictNet, MaxAttnPredictNet,
    # MeanMemAttnPredictNet, SumMemAttnPredictNet, MaxMemAttnPredictNet,
    # DIAMNet
    # "predict_net_add_enc": True,
    # "predict_net_add_degree": True,
    "predict_net_add_enc": True,
    "predict_net_add_degree": True,

    "predict_net_hidden_dim": 128,
    "predict_net_num_heads": 4,
    "predict_net_mem_len": 4,
    "predict_net_mem_init": "mean",
    # mean, sum, max, attn, circular_mean, circular_sum, circular_max, circular_attn, lstm
    "predict_net_recurrent_steps": 3,

    "emb_dim": 128,
    "activation_function": "leaky_relu",  # sigmoid, softmax, tanh, relu, leaky_relu, prelu, gelu

    "filter_net": "MaxGatedFilterNet",  # None, MaxGatedFilterNet
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
    "gcn_hidden_dim": 32,
    "gcn_ignore_norm": False,  # ignorm=True -> RGCN-SUM

    "graph_dir": "../data/PROTEINS/raw",
    "save_data_dir": "../data/PROTEINSPreTrain",
    "save_model_dir": "../dumps/PROTEINSPreTrain/GIN",
    "save_pretrain_model_dir": "../dumps/MUTAGPreTrain/GCN",
    "graphslabel_dir":"../data/PROTEINS/PROTEINS_graph_labels.txt",
    "downstream_graph_dir": "../data/debug/graphs",
    "downstream_save_data_dir": "../data/debug",
    "downstream_save_model_dir": "../dumps/debug",
    "downstream_graphslabel_dir":"../data/debug/graphs",
    "train_num_per_class": 3,
    "shot_num": 2,
    "temperature": 1,
    "graph_finetuning_input_dim": 8,
    "graph_finetuning_output_dim": 2,
    "graph_label_num": 3,
    "seed": 0,
    "model": "GIN",
    "dropout": 0.5,
    "node_feature_dim": 1,
    "pretrain_hop_num": 1
}

def train(model, optimizer, scheduler, data_type, data_loader, device, config, epoch, logger=None, writer=None):
    epoch_step = len(data_loader)
    total_step = config["epochs"] * epoch_step
    total_reg_loss = 0
    total_bp_loss = 0
    #total_cnt = 1e-6

    if config["reg_loss"] == "MAE":
        reg_crit = lambda pred, target: F.l1_loss(F.relu(pred), target)
    elif config["reg_loss"] == "MSE":
        reg_crit = lambda pred, target: F.mse_loss(F.relu(pred), target)
    elif config["reg_loss"] == "SMSE":
        reg_crit = lambda pred, target: F.smooth_l1_loss(F.relu(pred), target)
    elif config["reg_loss"] == "ABMAE":
        reg_crit = lambda pred, target: bp_compute_abmae(F.leaky_relu(pred), target)+0.8*F.l1_loss(F.relu(pred), target)
    else:
        raise NotImplementedError

    if config["bp_loss"] == "MAE":
        bp_crit = lambda pred, target, neg_slp: F.l1_loss(F.leaky_relu(pred, neg_slp), target)
    elif config["bp_loss"] == "MSE":
        bp_crit = lambda pred, target, neg_slp: F.mse_loss(F.leaky_relu(pred, neg_slp), target)
    elif config["bp_loss"] == "SMSE":
        bp_crit = lambda pred, target, neg_slp: F.smooth_l1_loss(F.leaky_relu(pred, neg_slp), target)
    elif config["bp_loss"] == "ABMAE":
        bp_crit = lambda pred, target, neg_slp: bp_compute_abmae(F.leaky_relu(pred, neg_slp), target)+0.8*F.l1_loss(F.leaky_relu(pred, neg_slp), target, reduce="none")
    else:
        raise NotImplementedError

    model.train()
    total_time=0
    for batch_id, batch in enumerate(data_loader):
        ids, graph_label, graph, graph_len= batch
        # print(batch)
        graph=graph.to(device)
        graph_label=graph_label.to(device)
        graph_len = graph_len.to(device)
        s=time.time()
        x,pred = model(graph, graph_len)
        pred=F.sigmoid(pred)

        adj = graph.adjacency_matrix()
        adj = adj.to(device)
        if train_config["pretrain_hop_num"]==0:
            pred=pred
        else:
            for count in range(train_config["pretrain_hop_num"]):
                pred = torch.matmul(adj, pred)
        #print(pred.size())
        _pred=split_and_batchify_graph_feats(pred, graph_len)[0]
        sample = graph.ndata['sample']
        _sample=split_and_batchify_graph_feats(sample, graph_len)[0]
        sample_=_sample.reshape(_sample.size(0),-1,1)
        #print(_pred.size())
        #print(sample_.size())
        _pred=torch.gather(input=_pred,dim=1,index=sample_)
        #print(_pred.size())
        _pred=_pred.resize_as(_sample)
        #print(_pred.size())

        reg_loss = compareloss(_pred,train_config["temperature"])
        reg_loss.requires_grad_(True)
        #      print(reg_loss.size())

        if isinstance(config["bp_loss_slp"], (int, float)):
            neg_slp = float(config["bp_loss_slp"])
        else:
            bp_loss_slp, l0, l1 = config["bp_loss_slp"].rsplit("$", 3)
            neg_slp = anneal_fn(bp_loss_slp, batch_id + epoch * epoch_step, T=total_step // 4, lambda0=float(l0),
                                lambda1=float(l1))
        bp_loss = reg_loss
        bp_loss.requires_grad_(True)


        # float
        reg_loss_item = reg_loss.item()
        bp_loss_item = bp_loss.item()
        total_reg_loss += reg_loss_item
        total_bp_loss += bp_loss_item

        if writer:
            writer.add_scalar("%s/REG-%s" % (data_type, config["reg_loss"]), reg_loss_item,
                              epoch * epoch_step + batch_id)
            writer.add_scalar("%s/BP-%s" % (data_type, config["bp_loss"]), bp_loss_item, epoch * epoch_step + batch_id)

        if logger and (batch_id % config["print_every"] == 0 or batch_id == epoch_step - 1):
            logger.info(
                "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:0>5d}/{:0>5d}\treg loss: {:0>10.3f}\tbp loss: {:0>16.3f}".format(
                    epoch, config["epochs"], data_type, batch_id, epoch_step,
                    reg_loss_item, bp_loss_item))
        print(bp_loss.grad)
        bp_loss.backward()
        if (config["update_every"] < 2 or batch_id % config["update_every"] == 0 or batch_id == epoch_step - 1):
            if config["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            if scheduler is not None:
                scheduler.step(epoch * epoch_step + batch_id)
            optimizer.step()
            optimizer.zero_grad()
        e=time.time()
        total_time+=e-s
    #mean_reg_loss = total_reg_loss / total_cnt
    #mean_bp_loss = total_bp_loss / total_cnt
    mean_reg_loss = total_reg_loss
    mean_bp_loss = total_bp_loss
    if writer:
        writer.add_scalar("%s/REG-%s-epoch" % (data_type, config["reg_loss"]), mean_reg_loss, epoch)
        writer.add_scalar("%s/BP-%s-epoch" % (data_type, config["bp_loss"]), mean_bp_loss, epoch)
    if logger:
        logger.info("epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\treg loss: {:0>10.3f}\tbp loss: {:0>16.3f}".format(
            epoch, config["epochs"], data_type, mean_reg_loss, mean_bp_loss))

    gc.collect()
    return mean_reg_loss, mean_bp_loss, total_time


def evaluate(model, data_type, data_loader, device, config, epoch, logger=None, writer=None):
    epoch_step = len(data_loader)
    total_reg_loss = 0
    #total_cnt = 1e-6

    evaluate_results = {"data": {"id": list(), "counts": list(), "pred": list()},
                        "error": {"mae": INF, "mse": INF},
                        "time": {"avg": list(), "total": 0.0}}

    if config["reg_loss"] == "MAE":
        reg_crit = lambda pred, target: F.l1_loss(F.relu(pred), target, reduce="none")
    elif config["reg_loss"] == "MSE":
        reg_crit = lambda pred, target: F.mse_loss(F.relu(pred), target, reduce="none")
    elif config["reg_loss"] == "SMSE":
        reg_crit = lambda pred, target: F.smooth_l1_loss(F.relu(pred), target, reduce="none")
    elif config["reg_loss"] == "ABMAE":
        reg_crit = lambda pred, target: bp_compute_abmae(F.relu(pred), target)+0.8*F.l1_loss(F.relu(pred), target, reduce="none")
    else:
        raise NotImplementedError

    if config["bp_loss"] == "MAE":
        bp_crit = lambda pred, target, neg_slp: F.l1_loss(F.leaky_relu(pred, neg_slp), target, reduce="none")
    elif config["bp_loss"] == "MSE":
        bp_crit = lambda pred, target, neg_slp: F.mse_loss(F.leaky_relu(pred, neg_slp), target, reduce="none")
    elif config["bp_loss"] == "SMSE":
        bp_crit = lambda pred, target, neg_slp: F.smooth_l1_loss(F.leaky_relu(pred, neg_slp), target, reduce="none")
    elif config["bp_loss"] == "ABMAE":
        bp_crit = lambda pred, target, neg_slp: bp_compute_abmae(F.leaky_relu(pred, neg_slp), target)+0.8*F.l1_loss(F.leaky_relu(pred, neg_slp), target, reduce="none")
    else:
        raise NotImplementedError

    model.eval()
    total_time=0
    with torch.no_grad():
        for batch_id, batch in enumerate(data_loader):
            ids, graph_label, graph, graph_len= batch
            #cnt = counts.shape[0]
            #total_cnt += cnt

            graph = graph.to(device)
            graph_label=graph_label.to(device)
            graph_len = graph_len.to(device)
            st = time.time()
            pred = model(graph, graph_len)
            adj = graph.adjacency_matrix()
            adj = adj.to(device)
            pred = torch.matmul(adj, pred)
            sample = graph.ndata['sample']
            _sample = sample.reshape(-1, 1)
            pred = torch.gather(input=pred, dim=0, index=_sample)
            pred = pred.resize_as(sample)

            et=time.time()
            evaluate_results["time"]["total"] += (et - st)
            #avg_t = (et - st) / (cnt + 1e-8)
            #evaluate_results["time"]["avg"].extend([avg_t] * cnt)
            #evaluate_results["data"]["pred"].extend(pred.cpu().view(-1).tolist())

            reg_loss = compareloss(pred, train_config["temperature"])
            reg_loss_item = reg_loss.item()

            if writer:
                writer.add_scalar("%s/REG-%s" % (data_type, config["reg_loss"]), reg_loss_item,
                                  epoch * epoch_step + batch_id)
                '''writer.add_scalar("%s/BP-%s" % (data_type, config["bp_loss"]), bp_loss_item,
                                  epoch * epoch_step + batch_id)'''

            if logger and batch_id == epoch_step - 1:
                logger.info(
                    "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:0>5d}/{:0>5d}\treg loss: {:0>10.3f}".format(
                        epoch, config["epochs"], data_type, batch_id, epoch_step,
                        reg_loss_item))
            et=time.time()
            total_time+=et-st
            total_reg_loss+=reg_loss_item
        mean_reg_loss = total_reg_loss
        #mean_bp_loss = total_bp_loss / total_cnt
        if writer:
            writer.add_scalar("%s/REG-%s-epoch" % (data_type, config["reg_loss"]), mean_reg_loss, epoch)
            #writer.add_scalar("%s/BP-%s-epoch" % (data_type, config["bp_loss"]), mean_bp_loss, epoch)
        if logger:
            logger.info("epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\treg loss: {:0>10.3f}".format(
                epoch, config["epochs"], data_type, mean_reg_loss))

        evaluate_results["error"]["loss"] = mean_reg_loss
        #evaluate_results["error"]["mse"] = evaluate_results["error"]["mse"] / total_cnt

    gc.collect()
    return mean_reg_loss,0,evaluate_results, total_time


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

    torch.manual_seed(train_config["seed"])
    np.random.seed(train_config["seed"])

    ts = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    model_name = "%s_%s_%s" % (train_config["model"], train_config["predict_net"], ts)
    save_model_dir = train_config["save_model_dir"]
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


    if train_config["model"] == "GCN":
        model = GCN(train_config)
    if train_config["model"] == "GIN":
        model = GIN(train_config)
    if train_config["model"] == "GAT":
        model = GAT(train_config)
    if train_config["model"] == "GraphSage":
        model = Graphsage(train_config)

    model = model.to(device)
    logger.info(model)
    logger.info("num of parameters: %d" % (sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # load data
    os.makedirs(train_config["save_data_dir"], exist_ok=True)
    data_loaders = OrderedDict({"train": None, "dev": None})
    if all([os.path.exists(os.path.join(train_config["save_data_dir"],
                                        "%s_%s_dataset.pt" % (
                                                data_type, "dgl" if train_config["model"] in ["RGCN", "RGIN",
                                                                                              "GAT","GCN","GraphSage","GIN"] else "edgeseq")))
            for data_type in data_loaders]):

        logger.info("loading data from pt...")
        for data_type in data_loaders:
            if train_config["model"] in ["RGCN", "RGIN", "GAT","GCN","GraphSage","GIN"]:
                dataset = GraphAdjDataset(list())
                dataset.load(os.path.join(train_config["save_data_dir"], "%s_dgl_dataset.pt" % (data_type)))

                sampler = Sampler(dataset, group_by=["graph"], batch_size=train_config["batch_size"],
                                  shuffle=data_type == "train", drop_last=False)
                data_loader = DataLoader(dataset,
                                         batch_sampler=sampler,
                                         collate_fn=GraphAdjDataset.batchify,
                                         pin_memory=data_type == "train")
            else:
                dataset = EdgeSeqDataset(list())
                dataset.load(os.path.join(train_config["save_data_dir"], "%s_edgeseq_dataset.pt" % (data_type)))
                sampler = Sampler(dataset, group_by=["graph", "pattern"], batch_size=train_config["batch_size"],
                                  shuffle=data_type == "train", drop_last=False)
                data_loader = DataLoader(dataset,
                                         batch_sampler=sampler,
                                         collate_fn=EdgeSeqDataset.batchify,
                                         pin_memory=data_type == "train")
            data_loaders[data_type] = data_loader
            logger.info("data (data_type: {:<5s}, len: {}) generated".format(data_type, len(dataset.data)))
            logger.info(
                "data_loader (data_type: {:<5s}, len: {}, batch_size: {}) generated".format(data_type, len(data_loader),
                                                                                            train_config["batch_size"]))
    else:
        data = pretrain_load_data(train_config["graph_dir"], train_config["graphslabel_dir"], num_workers=train_config["num_workers"])
        logger.info("{}/{}/{} data loaded".format(len(data["train"]), len(data["dev"]), len(data["test"])))
        for data_type, x in data.items():
            if train_config["model"] in ["RGCN", "RGIN", "GAT","GCN","GraphSage","GIN"]:
                if os.path.exists(os.path.join(train_config["save_data_dir"], "%s_dgl_dataset.pt" % (data_type))):
                    dataset = GraphAdjDataset(list())
                    dataset.load(os.path.join(train_config["save_data_dir"], "%s_dgl_dataset.pt" % (data_type)))
                else:
                    if(data_type=="test"):
                        test_time_start=time.time()
                    elif(data_type=="train"):
                        train_time_start = time.time()
                    else:
                        val_time_start = time.time()
                    dataset = GraphAdjDataset(x)
                    if(data_type=="test"):
                        test_time_end=time.time()
                        test_time=test_time_end-test_time_start
                        logger.info(
                            "preprocess test time: {:.3f}".format(test_time))
                    elif(data_type=="train"):
                        train_time_end=time.time()
                        train_time=train_time_end-train_time_start
                        logger.info(
                            "preprocess train time: {:.3f}".format(train_time))
                    else:
                        val_time_end=time.time()
                        val_time=val_time_end-val_time_start
                        logger.info(
                            "preprocess val time: {:.3f}".format(val_time))
                    dataset.save(os.path.join(train_config["save_data_dir"], "%s_dgl_dataset.pt" % (data_type)))
                sampler = Sampler(dataset, group_by=["graph"], batch_size=train_config["batch_size"],
                                  shuffle=data_type == "train", drop_last=False)
                data_loader = DataLoader(dataset,
                                         batch_sampler=sampler,
                                         collate_fn=GraphAdjDataset.batchify,
                                         pin_memory=data_type == "train")
            else:
                if os.path.exists(os.path.join(train_config["save_data_dir"], "%s_edgeseq_dataset.pt" % (data_type))):
                    dataset = EdgeSeqDataset(list())
                    dataset.load(os.path.join(train_config["save_data_dir"], "%s_edgeseq_dataset.pt" % (data_type)))
                else:
                    dataset = EdgeSeqDataset(x)
                    dataset.save(os.path.join(train_config["save_data_dir"], "%s_edgeseq_dataset.pt" % (data_type)))
                sampler = Sampler(dataset, group_by=["graph", "pattern"], batch_size=train_config["batch_size"],
                                  shuffle=data_type == "train", drop_last=False)
                data_loader = DataLoader(dataset,
                                         batch_sampler=sampler,
                                         collate_fn=EdgeSeqDataset.batchify,
                                         pin_memory=data_type == "train")
            data_loaders[data_type] = data_loader
            logger.info("data (data_type: {:<5s}, len: {}) generated".format(data_type, len(dataset.data)))
            logger.info(
                "data_loader (data_type: {:<5s}, len: {}, batch_size: {}) generated".format(data_type, len(data_loader),
                                                                                            train_config["batch_size"]))

    print('data_loaders', data_loaders.items())

    # optimizer and losses
    writer = SummaryWriter(save_model_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["lr"], weight_decay=train_config["weight_decay"],
                                  amsgrad=True)
    optimizer.zero_grad()
    scheduler = None
    best_reg_losses = {"train": INF, "dev": INF, "test": INF}
    best_reg_epochs = {"train": -1, "dev": -1, "test": -1}
    
    total_train_time=0
    total_dev_time=0
    total_test_time=0

    plt_x=list()
    plt_y=list()

    for epoch in range(train_config["epochs"]):
        for data_type, data_loader in data_loaders.items():

            if data_type == "train":
                mean_reg_loss, mean_bp_loss, _time = train(model, optimizer, scheduler, data_type, data_loader, device,
                                                    train_config, epoch, logger=logger, writer=writer)
                total_train_time+=_time
                torch.save(model.state_dict(), os.path.join(save_model_dir, 'epoch%d.pt' % (epoch)))
            else:
                mean_reg_loss, mean_bp_loss, evaluate_results, _time = evaluate(model, data_type, data_loader, device,
                                                                         train_config, epoch, logger=logger,
                                                                         writer=writer)
                total_dev_time+=_time
                with open(os.path.join(save_model_dir, '%s%d.json' % (data_type, epoch)), "w") as f:
                    json.dump(evaluate_results, f)
            if mean_reg_loss <= best_reg_losses[data_type]:
                best_reg_losses[data_type] = mean_reg_loss
                best_reg_epochs[data_type] = epoch
                logger.info(
                    "data_type: {:<5s}\tbest mean loss: {:.3f} (epoch: {:0>3d})".format(data_type, mean_reg_loss,
                                                                                        epoch))
            if data_type == "train":
                plt_x.append(epoch)
                plt_y.append(mean_reg_loss)

        plt.figure(1)
        plt.plot(plt_x,plt_y)
        plt.savefig('epoch_loss.png')
    for data_type in data_loaders.keys():
        logger.info(
            "data_type: {:<5s}\tbest mean loss: {:.3f} (epoch: {:0>3d})".format(data_type, best_reg_losses[data_type],
                                                                                best_reg_epochs[data_type]))

    best_epoch = train_config["epochs"]-1
    model.load_state_dict(torch.load(os.path.join(save_model_dir, 'epoch%d.pt' % (best_epoch))))
    torch.save(model.state_dict(), os.path.join(save_model_dir, "best.pt"))
