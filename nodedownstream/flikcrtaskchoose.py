
import os
import numpy as np



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
    "train_shotnum": 50,
    "val_shotnum": 50,
    "few_shot_tasknum": 10,

    "save_fewshot_dir": "../data/FlickrPreTrainNodeClassification/fewshot",
    "select_fewshot_dir": ".../data/FlickrPreTrainNodeClassification/select",
    "None": True,

    "downstream_dropout": 0,
    "node_feature_dim": 18,
    "train_label_num": 6,
    "val_label_num": 6,
    "test_label_num": 6,
    "nhop_neighbour": 1
}


fewshot_dir = os.path.join(train_config["save_fewshot_dir"], "%s_trainshot_%s_valshot_%s_tasks" %
                           (train_config["train_shotnum"], train_config["val_shotnum"],
                            train_config["few_shot_tasknum"]))
print(os.path.exists(fewshot_dir))
print("Load Few Shot")
trainset = np.load(os.path.join(fewshot_dir, "train_dgl_dataset.npy"), allow_pickle=True).tolist()
valset = np.load(os.path.join(fewshot_dir, "val_dgl_dataset.npy"), allow_pickle=True).tolist()
testset = np.load(os.path.join(fewshot_dir, "test_dgl_dataset.npy"), allow_pickle=True).tolist()
save=[0,1,3,8]
rettrain=[]
retval=[]
rettest=[]
for i in save:
    rettrain.append(trainset[i])
    retval.append(valset[1])
    rettest.append(testset[1])

selectdir = os.path.join(train_config["save_fewshot_dir"], "%s_trainshot_%s_valshot_%s_tasks" %
                           (train_config["train_shotnum"], train_config["val_shotnum"],
                            train_config["few_shot_tasknum"]))

if train_config["None"]:
    rettrain = np.array(rettrain)
    retval = np.array(retval)
    rettest = np.array(rettest)

else:
    trainset = np.load(os.path.join(selectdir, "train_dgl_dataset.npy"), allow_pickle=True).tolist()
    valset = np.load(os.path.join(selectdir, "val_dgl_dataset.npy"), allow_pickle=True).tolist()
    testset = np.load(os.path.join(selectdir, "test_dgl_dataset.npy"), allow_pickle=True).tolist()
    rettrain=rettrain.append(trainset)
    retval=rettrain.append(valset)
    rettest=rettrain.append(testset)
    rettrain = np.array(rettrain)
    retval = np.array(retval)
    rettest = np.array(rettest)

np.save(os.path.join(fewshot_dir, "train_dgl_dataset"), rettrain)
np.save(os.path.join(fewshot_dir, "val_dgl_dataset"), retval)
np.save(os.path.join(fewshot_dir, "test_dgl_dataset"), rettest)

