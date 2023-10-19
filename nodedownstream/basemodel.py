import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
from utils import int2onehot
from utils import get_enc_len, split_and_batchify_graph_feats, batch_convert_len_to_mask
from embedding import OrthogonalEmbedding, NormalEmbedding, EquivariantEmbedding
from filternet import MaxGatedFilterNet
from predictnet import MeanPredictNet, SumPredictNet, MaxPredictNet, \
    MeanAttnPredictNet, SumAttnPredictNet, MaxAttnPredictNet, \
    MeanMemAttnPredictNet, SumMemAttnPredictNet, MaxMemAttnPredictNet, \
    DIAMNet

class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()

        self.act_func = config["activation_function"]
        self.init_emb = config["init_emb"]
        self.share_emb = config["share_emb"]
        self.share_arch = config["share_arch"]
        self.base = config["base"]
        self.max_ngv = config["max_ngv"]
        self.max_ngvl = config["max_ngvl"]
        self.max_nge = config["max_nge"]
        self.max_ngel = config["max_ngel"]
        self.max_npv = config["max_npv"]
        self.max_npvl = config["max_npvl"]
        self.max_npe = config["max_npe"]
        self.max_npel = config["max_npel"]
        
        self.emb_dim = config["emb_dim"]
        self.dropout = config["dropout"]
        self.dropatt = config["dropatt"]
        self.add_enc = config["predict_net_add_enc"]

        # create encoding layer
        # create filter layers
        # create embedding layers
        # create networks
        #self.p_net, self.g_net = None, None
        self.g_net = None

        # create predict layers
        self.predict_net = None

    def get_emb_dim(self):
        if self.init_emb == "None":
            return self.get_enc_dim()
        else:
            return self.emb_dim

    def get_enc(self, graph, graph_len):
        raise NotImplementedError

    def get_emb(self, graph, graph_len):
        raise NotImplementedError

    def get_filter_gate(self, graph, graph_len):
        raise NotImplementedError

    def create_filter(self, filter_type):
        if filter_type == "None":
            filter_net = None
        elif filter_type == "MaxGatedFilterNet":
            filter_net = MaxGatedFilterNet()
        else:
            raise NotImplementedError("Currently, %s is not supported!" % (filter_type))
        return filter_net

    def create_enc(self, max_n, base):
        enc_len = get_enc_len(max_n-1, base)
        enc_dim = enc_len * base
        enc = nn.Embedding(max_n, enc_dim)
        enc.weight.data.copy_(torch.from_numpy(int2onehot(np.arange(0, max_n), enc_len, base)))
        enc.weight.requires_grad=False
        return enc

    def create_emb(self, input_dim, emb_dim, init_emb="Orthogonal"):
        if init_emb == "None":
            emb = None
        elif init_emb == "Orthogonal":
            emb = OrthogonalEmbedding(input_dim, emb_dim)
        elif init_emb == "Normal":
            emb = NormalEmbedding(input_dim, emb_dim)
        elif init_emb == "Equivariant":
            emb = EquivariantEmbedding(input_dim, emb_dim)
        else:
            raise NotImplementedError
        return emb

    def create_net(self, name, input_dim, **kw):
        raise NotImplementedError

    def create_predict_net(self, predict_type, pattern_dim, graph_dim, **kw):
        if predict_type == "None":
            predict_net = None
        elif predict_type == "MeanPredictNet":
            hidden_dim = kw.get("hidden_dim", 64)
            predict_net = MeanPredictNet(pattern_dim, graph_dim, hidden_dim,
                act_func=self.act_func, dropout=self.dropout)
        elif predict_type == "SumPredictNet":
            hidden_dim = kw.get("hidden_dim", 64)
            predict_net = SumPredictNet(pattern_dim, graph_dim, hidden_dim,
                act_func=self.act_func, dropout=self.dropout)
        elif predict_type == "MaxPredictNet":
            hidden_dim = kw.get("hidden_dim", 64)
            predict_net = MaxPredictNet(pattern_dim, graph_dim, hidden_dim,
                act_func=self.act_func, dropout=self.dropout)
        elif predict_type == "MeanAttnPredictNet":
            hidden_dim = kw.get("hidden_dim", 64)
            recurrent_steps = kw.get("recurrent_steps", 1)
            num_heads = kw.get("num_heads", 1)
            predict_net = MeanAttnPredictNet(pattern_dim, graph_dim, hidden_dim,
                act_func=self.act_func,
                num_heads=num_heads, recurrent_steps=recurrent_steps,
                dropout=self.dropout, dropatt=self.dropatt)
        elif predict_type == "SumAttnPredictNet":
            hidden_dim = kw.get("hidden_dim", 64)
            recurrent_steps = kw.get("recurrent_steps", 1)
            num_heads = kw.get("num_heads", 1)
            predict_net = SumAttnPredictNet(pattern_dim, graph_dim, hidden_dim,
                act_func=self.act_func,
                num_heads=num_heads, recurrent_steps=recurrent_steps,
                dropout=self.dropout, dropatt=self.dropatt)
        elif predict_type == "MaxAttnPredictNet":
            hidden_dim = kw.get("hidden_dim", 64)
            recurrent_steps = kw.get("recurrent_steps", 1)
            num_heads = kw.get("num_heads", 1)
            predict_net = MaxAttnPredictNet(pattern_dim, graph_dim, hidden_dim,
                act_func=self.act_func,
                num_heads=num_heads, recurrent_steps=recurrent_steps,
                dropout=self.dropout, dropatt=self.dropatt)
        elif predict_type == "MeanMemAttnPredictNet":
            hidden_dim = kw.get("hidden_dim", 64)
            recurrent_steps = kw.get("recurrent_steps", 1)
            num_heads = kw.get("num_heads", 1)
            mem_len = kw.get("mem_len", 4)
            predict_net = MeanMemAttnPredictNet(pattern_dim, graph_dim, hidden_dim,
                act_func=self.act_func,
                num_heads=num_heads, recurrent_steps=recurrent_steps,
                mem_len=mem_len,
                dropout=self.dropout, dropatt=self.dropatt)
        elif predict_type == "SumMemAttnPredictNet":
            hidden_dim = kw.get("hidden_dim", 64)
            recurrent_steps = kw.get("recurrent_steps", 1)
            num_heads = kw.get("num_heads", 1)
            mem_len = kw.get("mem_len", 4)
            predict_net = SumMemAttnPredictNet(pattern_dim, graph_dim, hidden_dim,
                act_func=self.act_func,
                num_heads=num_heads, recurrent_steps=recurrent_steps,
                mem_len=mem_len,
                dropout=self.dropout, dropatt=self.dropatt)
        elif predict_type == "MaxMemAttnPredictNet":
            hidden_dim = kw.get("hidden_dim", 64)
            recurrent_steps = kw.get("recurrent_steps", 1)
            num_heads = kw.get("num_heads", 1)
            mem_len = kw.get("mem_len", 4)
            predict_net = MaxMemAttnPredictNet(pattern_dim, graph_dim, hidden_dim,
                act_func=self.act_func,
                num_heads=num_heads, recurrent_steps=recurrent_steps,
                mem_len=mem_len,
                dropout=self.dropout, dropatt=self.dropatt)
        elif predict_type == "DIAMNet":
            hidden_dim = kw.get("hidden_dim", 64)
            recurrent_steps = kw.get("recurrent_steps", 1)
            num_heads = kw.get("num_heads", 1)
            mem_len = kw.get("mem_len", 4)
            mem_init = kw.get("mem_init", "mean")
            predict_net = DIAMNet(pattern_dim, graph_dim, hidden_dim,
                act_func=self.act_func,
                num_heads=num_heads, recurrent_steps=recurrent_steps, 
                mem_len=mem_len, mem_init=mem_init,
                dropout=self.dropout, dropatt=self.dropatt)
        else:
            raise NotImplementedError("Currently, %s is not supported!" % (predict_type))
        return predict_net

    def increase_input_size(self, config):
        assert config["base"] == self.base
        assert config["max_npv"] >= self.max_npv
        assert config["max_npvl"] >= self.max_npvl
        assert config["max_npe"] >= self.max_npe
        assert config["max_npel"] >= self.max_npel
        assert config["max_ngv"] >= self.max_ngv
        assert config["max_ngvl"] >= self.max_ngvl
        assert config["max_nge"] >= self.max_nge
        assert config["max_ngel"] >= self.max_ngel
        assert config["predict_net_add_enc"] or not self.add_enc
        assert config["predict_net_add_degree"] or not self.add_degree
        
        # create encoding layers
        # increase embedding layers
        # increase predict network
        # set new parameters

    def increase_net(self, config):
        raise NotImplementedError


class EdgeSeqModel(BaseModel):
    def __init__(self, config):
        super(EdgeSeqModel, self).__init__(config)
        # create encoding layer
        self.g_v_enc, self.g_vl_enc, self.g_el_enc = \
            [self.create_enc(max_n, self.base) for max_n in [self.max_ngv, self.max_ngvl, self.max_ngel]]
        self.g_u_enc, self.g_ul_enc = self.g_v_enc, self.g_vl_enc
        if self.share_emb:
            self.p_v_enc, self.p_vl_enc, self.p_el_enc = \
                self.g_v_enc, self.g_vl_enc, self.g_el_enc
        else:
            self.p_v_enc, self.p_vl_enc, self.p_el_enc = \
                [self.create_enc(max_n, self.base) for max_n in [self.max_npv, self.max_npvl, self.max_npel]]
        self.p_u_enc, self.p_ul_enc = self.p_v_enc, self.p_vl_enc

        # create filter layers
        self.ul_flt, self.el_flt, self.vl_flt = [self.create_filter(config["filter_net"]) for _ in range(3)]

        # create embedding layers
        self.g_u_emb, self.g_v_emb, self.g_ul_emb, self.g_el_emb, self.g_vl_emb = \
                [self.create_emb(enc.embedding_dim, self.emb_dim, init_emb=self.init_emb) \
                    for enc in [self.g_u_enc, self.g_v_enc, self.g_ul_enc, self.g_el_enc, self.g_vl_enc]]
        if self.share_emb:
            self.p_u_emb, self.p_v_emb, self.p_ul_emb, self.p_el_emb, self.p_vl_emb = \
                self.g_u_emb, self.g_v_emb, self.g_ul_emb, self.g_el_emb, self.g_vl_emb
        else:
            self.p_u_emb, self.p_v_emb, self.p_ul_emb, self.p_el_emb, self.p_vl_emb = \
                [self.create_emb(enc.embedding_dim, self.emb_dim, init_emb=self.init_emb) \
                    for enc in [self.p_u_enc, self.p_v_enc, self.p_ul_enc, self.p_el_enc, self.p_vl_enc]]

        # create networks
        # create predict layers

    def get_enc_dim(self):
        #get_enc_len返回math.floor(math.log(x, base)+1.0)
        #base默认为2
        g_dim = self.base * (get_enc_len(self.max_ngv-1, self.base) * 2 + \
            get_enc_len(self.max_ngvl-1, self.base) * 2 + \
            get_enc_len(self.max_ngel-1, self.base))
        if self.share_emb:
            return g_dim, g_dim
        else:
            p_dim = self.base * (get_enc_len(self.max_npv-1, self.base) * 2 + \
                get_enc_len(self.max_npvl-1, self.base) * 2 + \
                get_enc_len(self.max_npel-1, self.base))
            return p_dim, g_dim

    def get_emb_dim(self):
        if self.init_emb == "None":
            return self.get_enc_dim()
        else:
            return self.emb_dim, self.emb_dim

    def get_enc(self, pattern, pattern_len, graph, graph_len):
        pattern_u, pattern_v, pattern_ul, pattern_el, pattern_vl = \
            self.p_u_enc(pattern.u), self.p_v_enc(pattern.v), self.p_ul_enc(pattern.ul), self.p_el_enc(pattern.el), self.p_vl_enc(pattern.vl)
        graph_u, graph_v, graph_ul, graph_el, graph_vl = \
            self.g_u_enc(graph.u), self.g_v_enc(graph.v), self.g_ul_enc(graph.ul), self.g_el_enc(graph.el), self.g_vl_enc(graph.vl)

        p_enc = torch.cat([
            pattern_u,
            pattern_v,
            pattern_ul,
            pattern_el,
            pattern_vl], dim=2)
        g_enc = torch.cat([
            graph_u,
            graph_v,
            graph_ul,
            graph_el,
            graph_vl], dim=2)
        return p_enc, g_enc

    def get_emb(self, pattern, pattern_len, graph, graph_len):
        bsz = pattern_len.size(0)
        pattern_u, pattern_v, pattern_ul, pattern_el, pattern_vl = \
            self.p_u_enc(pattern.u), self.p_v_enc(pattern.v), self.p_ul_enc(pattern.ul), self.p_el_enc(pattern.el), self.p_vl_enc(pattern.vl)
        graph_u, graph_v, graph_ul, graph_el, graph_vl = \
            self.g_u_enc(graph.u), self.g_v_enc(graph.v), self.g_ul_enc(graph.ul), self.g_el_enc(graph.el), self.g_vl_enc(graph.vl)

        if self.init_emb == "None":
            p_emb = torch.cat([pattern_u, pattern_v, pattern_ul, pattern_el, pattern_vl], dim=2)
            g_emb = torch.cat([graph_u, graph_v, graph_ul, graph_el, graph_vl], dim=2)
        else:
            p_emb = self.p_u_emb(pattern_u) + \
                self.p_v_emb(pattern_v) + \
                self.p_ul_emb(pattern_ul) + \
                self.p_el_emb(pattern_el) + \
                self.p_vl_emb(pattern_vl)
            g_emb = self.g_u_emb(graph_u) + \
                self.g_v_emb(graph_v) + \
                self.g_ul_emb(graph_ul) + \
                self.g_el_emb(graph_el) + \
                self.g_vl_emb(graph_vl)
        return p_emb, g_emb

    def get_filter_gate(self, pattern, pattern_len, graph, graph_len):
        gate = None
        if self.ul_flt is not None:
            if gate is not None:
                gate &= self.ul_flt(pattern.ul, graph.ul)
            else:
                gate = self.ul_flt(pattern.ul, graph.ul)
        if self.el_flt is not None:
            if gate is not None:
                gate &= self.el_flt(pattern.el, graph.el)
            else:
                gate = self.el_flt(pattern.el, graph.el)
        if self.vl_flt is not None:
            if gate is not None:
                gate &= self.vl_flt(pattern.vl, graph.vl)
            else:
                gate = self.vl_flt(pattern.vl, graph.vl)
        return gate

    def increase_input_size(self, config):
        super(EdgeSeqModel, self).increase_input_size(config)

        # create encoding layers
        new_g_v_enc, new_g_vl_enc, new_g_el_enc = \
            [self.create_enc(max_n, self.base) for max_n in [config["max_ngv"], config["max_ngvl"], config["max_ngel"]]]
        if self.share_emb:
            new_p_v_enc, new_p_vl_enc, new_p_el_enc = \
                new_g_v_enc, new_g_vl_enc, new_g_el_enc
        else:
            new_p_v_enc, new_p_vl_enc, new_p_el_enc = \
                [self.create_enc(max_n, self.base) for max_n in [config["max_npv"], config["max_npvl"], config["max_npel"]]]
        del self.g_v_enc, self.g_vl_enc, self.g_el_enc, self.g_u_enc, self.g_ul_enc
        del self.p_v_enc, self.p_vl_enc, self.p_el_enc, self.p_u_enc, self.p_ul_enc
        self.g_v_enc, self.g_vl_enc, self.g_el_enc = new_g_v_enc, new_g_vl_enc, new_g_el_enc
        self.g_u_enc, self.g_ul_enc = self.g_v_enc, self.g_vl_enc
        self.p_v_enc, self.p_vl_enc, self.p_el_enc = new_p_v_enc, new_p_vl_enc, new_p_el_enc
        self.p_u_enc, self.p_ul_enc = self.p_v_enc, self.p_vl_enc

        # increase embedding layers
        self.g_u_emb.increase_input_size(self.g_u_enc.embedding_dim)
        self.g_v_emb.increase_input_size(self.g_v_enc.embedding_dim)
        self.g_ul_emb.increase_input_size(self.g_ul_enc.embedding_dim)
        self.g_vl_emb.increase_input_size(self.g_vl_enc.embedding_dim)
        self.g_el_emb.increase_input_size(self.g_el_enc.embedding_dim)
        if not self.share_emb:
            self.p_u_emb.increase_input_size(self.p_u_enc.embedding_dim)
            self.p_v_emb.increase_input_size(self.p_v_enc.embedding_dim)
            self.p_ul_emb.increase_input_size(self.p_ul_enc.embedding_dim)
            self.p_vl_emb.increase_input_size(self.p_vl_enc.embedding_dim)
            self.p_el_emb.increase_input_size(self.p_el_enc.embedding_dim)

        # increase predict network

        # set new parameters
        self.max_npv = config["max_npv"]
        self.max_npvl = config["max_npvl"]
        self.max_npe = config["max_npe"]
        self.max_npel = config["max_npel"]
        self.max_ngv = config["max_ngv"]
        self.max_ngvl = config["max_ngvl"]
        self.max_nge = config["max_nge"]
        self.max_ngel = config["max_ngel"]



class GraphAdjModel(BaseModel):
    def __init__(self, config):
        super(GraphAdjModel, self).__init__(config)
        
        self.add_degree = config["predict_net_add_degree"]

        # create encoding layer
        self.g_v_enc, self.g_vl_enc = \
            [self.create_enc(max_n, self.base) for max_n in [self.max_ngv, self.max_ngvl]]
        '''if self.share_emb:
            self.p_v_enc, self.p_vl_enc = \
                self.g_v_enc, self.g_vl_enc
        else:
            self.p_v_enc, self.p_vl_enc = \
                [self.create_enc(max_n, self.base) for max_n in [self.max_npv, self.max_npvl]]'''

        # create filter layers
        self.vl_flt = self.create_filter(config["filter_net"])

        # create embedding layers
        self.g_vl_emb = self.create_emb(self.g_vl_enc.embedding_dim, self.emb_dim, init_emb=self.init_emb)
        '''if self.share_emb:
            self.p_vl_emb = self.g_vl_emb
        else:
            self.p_vl_emb = self.create_emb(self.p_vl_enc.embedding_dim, self.emb_dim, init_emb=self.init_emb)'''

        # create networks
        # create predict layers

    def get_enc_dim(self):
        g_dim = self.base * (get_enc_len(self.max_ngv-1, self.base) + get_enc_len(self.max_ngvl-1, self.base))
        '''if self.share_emb:
            return g_dim, g_dim
        else:
            p_dim = self.base * (get_enc_len(self.max_npv-1, self.base) + get_enc_len(self.max_npvl-1, self.base))
            return p_dim, g_dim'''
        return g_dim

    #def get_enc(self, pattern, pattern_len, graph, graph_len):
    def get_enc(self, graph, graph_len):
        graph_v, graph_vl = self.g_v_enc(graph.ndata["id"]), self.g_vl_enc(graph.ndata["label"])
        #p_enc = torch.cat([pattern_v, pattern_vl], dim=1)
        g_enc = torch.cat([graph_v, graph_vl], dim=1)
        #return p_enc, g_enc
        return g_enc

    #def get_emb(self, pattern, pattern_len, graph, graph_len):
    def get_emb(self, graph, graph_len):

        graph_v, graph_vl = self.g_v_enc(graph.ndata["id"]), self.g_vl_enc(graph.ndata["label"])

        if self.init_emb == "None":
            g_emb = graph_vl
        else:
            g_emb = self.g_vl_emb(graph_vl)
        return g_emb

    def get_filter_gate(self, graph, graph_len):

        gate = None
        if self.vl_flt is not None:
            gate = self.vl_flt(split_and_batchify_graph_feats(graph.ndata["label"].unsqueeze(-1), graph_len)[0])

        if gate is not None:
            bsz = graph_len.size(0)
            max_g_len = graph_len.max()
            if bsz * max_g_len != graph.number_of_nodes():
                graph_mask = batch_convert_len_to_mask(graph_len) # bsz x max_len
                gate = gate.masked_select(graph_mask.unsqueeze(-1)).view(-1, 1)
            else:
                gate = gate.view(-1, 1)
        return gate
    
    def increase_input_size(self, config):
        super(GraphAdjModel, self).increase_input_size(config)

        # create encoding layers
        new_g_v_enc, new_g_vl_enc = \
            [self.create_enc(max_n, self.base) for max_n in [config["max_ngv"], config["max_ngvl"]]]
        del self.g_v_enc, self.g_vl_enc
        #del self.p_v_enc, self.p_vl_enc
        self.g_v_enc, self.g_vl_enc = new_g_v_enc, new_g_vl_enc
        self.g_vl_emb.increase_input_size(self.g_vl_enc.embedding_dim)

        self.max_ngv = config["max_ngv"]
        self.max_ngvl = config["max_ngvl"]
        self.max_nge = config["max_nge"]
        self.max_ngel = config["max_ngel"]
