import pandas as pd
import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    SAGEConv,
    GINConv,
    TransformerConv,
    RGCNConv,
    ClusterGCNConv,
    TAGConv,
)
import copy
import collections


# 获取所有节点的邻居


class GCNEncoder(torch.nn.Module):
    def __init__(self, input_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, 128)
        # self.bn1 = torch.nn.BatchNorm1d(128)
        self.conv2 = GCNConv(128, 16)

    def forward(self, x, train_pos_edge_index):
        x = x
        train_pos_edge_index = train_pos_edge_index
        x = self.conv1(x, train_pos_edge_index)
        x = x.relu()
        # x = self.bn1(x)
        x = self.conv2(x, train_pos_edge_index)
        return x


class Sample_GAT(torch.nn.Module):
    def __init__(self, input_dim):
        super(Sample_GAT, self).__init__()
        self.conv1 = GATConv(input_dim, 128, heads=1, dropout=0.3)
        self.conv2 = GATConv(128, 16, heads=1, dropout=0.3)

    def forward(self, x, train_pos_edge_index, edge_weight):
        x = x
        train_pos_edge_index = train_pos_edge_index
        x = self.conv1(
            x,
            train_pos_edge_index,
            torch.squeeze(
                edge_weight,
                1,
            ).float(),
        )
        # x=F.dropout(x, 0.5)
        x = x.relu()
        # x = self.bn1(x)
        x = self.conv2(x, train_pos_edge_index, torch.squeeze(edge_weight, 1).float())
        return x


class Sample_GCN(torch.nn.Module):
    def __init__(self, input_dim):
        super(Sample_GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, 128)
        # self.bn1 = torch.nn.BatchNorm1d(128)
        self.conv2 = GCNConv(128, 16)

    def forward(self, x, train_pos_edge_index, edge_weight):
        x = x
        train_pos_edge_index = train_pos_edge_index
        x = self.conv1(
            x,
            train_pos_edge_index,
            torch.squeeze(
                edge_weight,
                1,
            ).float(),
        )
        # x=F.dropout(x, 0.5)
        x = x.relu()
        # x = self.bn1(x)
        x = self.conv2(x, train_pos_edge_index, torch.squeeze(edge_weight, 1).float())
        return x


# class GCNEncoder(torch.nn.Module):
#     def __init__(self, input_dim,num_layer=3,drop_ratio=0.1):
#         super(GCNEncoder, self).__init__()
#         self.conv1 = GCNConv(input_dim, 128)
#         # self.bn1 = torch.nn.BatchNorm1d(128)
#         self.conv2 = GCNConv(128, 16)
#         # self.conv3 = GCNConv(64, 16)
#         self.num_layer = num_layer

#         self.drop_ratio = drop_ratio
#         self.gnns = nn.ModuleList()
#         for layer in range(num_layer):
#             self.gnns.append(GCNConv(16,16, aggr="add"))

#         # List of batchnorms
#         self.batch_norms = nn.ModuleList()
#         for layer in range(num_layer):
#             self.batch_norms.append(nn.BatchNorm1d(16))

#     def forward(self, x, train_pos_edge_index):
#         x=x.cuda()
#         train_pos_edge_index=train_pos_edge_index.cuda()
#         x = self.conv1(x, train_pos_edge_index)
#         # x = x.relu()
#         #x = self.bn1(x)
#         x = self.conv2(x, train_pos_edge_index)
#         # x=F.dropout(x, self.drop_ratio)
#         # x = self.conv3(x, train_pos_edge_index)
#         # x = x.relu()
#         x = F.dropout(F.relu(x), self.drop_ratio)
#         for layer in range(3):
#             h = self.gnns[layer](x, train_pos_edge_index)
#             h = self.batch_norms[layer](h)
#             if layer == self.num_layer - 1:
#                 h = F.dropout(h, self.drop_ratio)
#             else:
#                 h = F.dropout(F.relu(h), self.drop_ratio)
#         # x = self.conv2(h, train_pos_edge_index)

#         return h
class TransformEncoder(torch.nn.Module):
    def __init__(self, input_dim):
        super(TransformEncoder, self).__init__()
        self.conv1 = TransformerConv(input_dim, 128, heads=1, dropout=0.3)
        self.conv2 = TransformerConv(128, 16, heads=1, dropout=0.3)
        # self.bn1 = torch.nn.BatchNorm1d(128)
        # self.conv2 = GCNConv(128, 16)

    def forward(self, x, train_pos_edge_index):
        x = x.cuda()
        train_pos_edge_index = train_pos_edge_index.cuda()
        x = self.conv1(x, train_pos_edge_index)
        x = x.relu()
        # x = self.bn1(x)
        x = self.conv2(x, train_pos_edge_index)
        return x


class GATEncoder(torch.nn.Module):
    def __init__(self, input_dim):
        super(GATEncoder, self).__init__()
        self.conv1 = GATConv(input_dim, 128, heads=1, dropout=0.3)
        self.conv2 = GATConv(128, 16, heads=1, dropout=0.3)
        # self.bn1 = torch.nn.BatchNorm1d(128)
        # self.conv2 = GCNConv(128, 16)

    def forward(self, x, train_pos_edge_index):
        x = x
        train_pos_edge_index = train_pos_edge_index
        x = self.conv1(x, train_pos_edge_index)
        x = x.relu()
        # x = self.bn1(x)
        x = self.conv2(x, train_pos_edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, input_dim):
        super(GAT, self).__init__()

        self.hid = 16
        self.in_head = 8
        self.out_head = 1

        self.conv1 = GATConv(
            in_channels=input_dim,
            out_channels=self.hid,
            heads=self.in_head,
            dropout=0.6,
        )
        self.conv2 = GATConv(
            in_channels=self.hid * self.in_head,
            out_channels=16,
            concat=False,
            heads=self.out_head,
            dropout=0.6,
        )

    def forward(self, x, edge_index):

        # x = F.dropout(x, p=0.6)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.1)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class SAGEConvEncoder(torch.nn.Module):
    def __init__(self, input_dim):
        super(SAGEConvEncoder, self).__init__()

        self.conv1 = SAGEConv(input_dim, 128, 1)
        # self.bn1 = torch.nn.BatchNorm1d(128)
        self.conv2 = SAGEConv(128, 16, 1)
        # self.num_neighbors = neighbor_samples

    def forward(self, x, train_pos_edge_index):
        x = x
        # neighbours = neighbors(train_pos_edge_index)
        # num_samples = self.num_neighbors
        train_pos_edge_index = train_pos_edge_index
        # self.num_neighbors = num_neighbors
        x = self.conv1(x, train_pos_edge_index)
        x = x.relu()
        # x = self.bn1(x)
        x = self.conv2(x, train_pos_edge_index)
        # tmp = np.save(
        #     '/home/intern/SyntheticLethal/final_code/Model/2.npy',
        #     x.detach().cpu().numpy(),
        # )

        return x


class ClusterGCNConvEncoder(torch.nn.Module):
    def __init__(self, input_dim):
        super(ClusterGCNConvEncoder, self).__init__()
        self.conv1 = ClusterGCNConv(input_dim, 128)
        # self.bn1 = torch.nn.BatchNorm1d(128)
        self.conv2 = ClusterGCNConv(128, 16)

    def forward(self, x, train_pos_edge_index):
        x = x.cuda()
        train_pos_edge_index = train_pos_edge_index.cuda()
        x = self.conv1(x, train_pos_edge_index)
        x = x.relu()
        # x = self.bn1(x)
        x = self.conv2(x, train_pos_edge_index)
        return x


class DNAConvEncoder(torch.nn.Module):
    def __init__(self, input_dim):
        super(DNAConvEncoder, self).__init__()
        self.conv1 = TAGConv(input_dim, 128)
        # self.conv2 = TAGConv(256, 64)
        # self.bn1 = torch.nn.BatchNorm1d(128)
        self.conv2 = TAGConv(128, 64)
        self.conv3 = TAGConv(64, 16)

    def forward(self, x, train_pos_edge_index):
        x = x
        train_pos_edge_index = train_pos_edge_index
        x = self.conv1(x, train_pos_edge_index)
        # x = F.elu(x)
        x = F.dropout(x, p=0.5)

        # x = self.bn1(x)
        x = self.conv2(x, train_pos_edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.5)
        x = self.conv3(x, train_pos_edge_index)

        # x = F.dropout(x, p=0.2)
        # x = self.conv2(x, train_pos_edge_index)
        # x = F.elu(x)

        return x


class MFFSL(torch.nn.Module):
    def __init__(
        self,
        ablation,
        model_type,
        args,
        n_graph,
        node_emb_dim,
        sl_input_dim,
        d_model=16,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        esm_in_dimension=1280,
    ):
        super(MFFSL, self).__init__()
        self.model_type = model_type
        self.data_aba = ablation
        self.delete_EM = args.delete_EM
        self.delete_Es = args.delete_Es
        self.delete_par = args.delete_par
        self.delete_L1000 = args.delete_L1000
        if model_type == 'GCN':

            self.encode_sl = GCNEncoder(input_dim=sl_input_dim)
            self.encode_isme = Sample_GCN(input_dim=1)
            self.encode_low = Sample_GCN(input_dim=1)
            self.encode_par = Sample_GCN(input_dim=1)
            self.fold_change = Sample_GCN(input_dim=1)
        if model_type == 'GAT':
            self.encode_sl = GAT(input_dim=sl_input_dim)
            self.encode_isme = GAT(input_dim=1)
            self.encode_low = GAT(input_dim=1)
            self.encode_par = GAT(input_dim=1)
            self.fold_change = GAT(input_dim=1)
        if model_type == 'GraphSage':
            # GraphSAGE(in_channels=16, hidden_channels=32, out_channels=64, num_layers=3, neighbor_samples=[5, 10])
            # self.encode_sl = SAGEConvEncoder(
            #     input_dim=sl_input_dim, neighbor_samples=None
            # )
            # self.encode_isme = SAGEConvEncoder(input_dim=1, neighbor_samples=None)
            # self.encode_low = SAGEConvEncoder(input_dim=1, neighbor_samples=None)
            # self.encode_par = SAGEConvEncoder(input_dim=1, neighbor_samples=None)
            # self.fold_change = SAGEConvEncoder(input_dim=1, neighbor_samples=None)
            self.encode_sl = SAGEConvEncoder(input_dim=sl_input_dim)
            self.encode_isme = SAGEConvEncoder(input_dim=1)
            self.encode_low = SAGEConvEncoder(input_dim=1)
            self.encode_par = SAGEConvEncoder(input_dim=1)
            self.fold_change = SAGEConvEncoder(input_dim=1)
        if model_type == 'TAG':
            self.encode_sl = DNAConvEncoder(input_dim=sl_input_dim)
            self.encode_isme = DNAConvEncoder(input_dim=1)
            self.encode_low = DNAConvEncoder(input_dim=1)
            self.encode_par = DNAConvEncoder(input_dim=1)
            self.fold_change = DNAConvEncoder(input_dim=1)
        if model_type == 'Transformer':
            self.encode_sl = TransformEncoder(input_dim=sl_input_dim)
            self.encode_isme = TransformEncoder(input_dim=1)
            self.encode_low = TransformEncoder(input_dim=1)
            self.encode_par = TransformEncoder(input_dim=1)
            self.fold_change = TransformEncoder(input_dim=1)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward=8)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

        decoder_layer2 = TransformerDecoderLayer2(d_model, nhead, dim_feedforward=8)
        self.decoder2 = TransformerDecoder2(decoder_layer2, num_decoder_layers)

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward=8)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        self.d_model = d_model
        self.nhead = nhead

        self.aggragator = AggregateLayer(d_model=2 * d_model)
        self.predictor = GlobalPredictor(d_model=2 * d_model, d_h=64, d_out=1)
        self.predictor0 = GlobalPredictor(d_model=2 * d_model, d_h=48, d_out=1)
        self._reset_parameters()

        # self.linear1 = torch.nn.Linear(n_graph*node_emb_dim, 32)
        self.linear4 = torch.nn.Linear(80, 32)
        self.linear1 = torch.nn.Linear(64, 32)
        self.linear0 = torch.nn.Linear(48, 32)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.linear2 = torch.nn.Linear(32, 16)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.linear3 = torch.nn.Linear(16, 1)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def decode(self, z, pos_edge_index, neg_edge_index):
        # z=z.cuda()
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        emb_features = z[edge_index[0]] + z[edge_index[1]]
        return torch.Tensor(emb_features.cpu())

    def forward(
        self,
        x,
        train_pos_edge_index,
        sl_pospre,
        sl_pos_back,
        sl_neg,
        isme,
        low_exp,
        para,
        emb_fold_change,
        label,
    ):
        if label == 'train':
            sl_pos = train_pos_edge_index
            emb_sl = self.encode_sl(x, sl_pos)
        else:
            emb_sl = self.encode_sl(x, sl_pospre)

        # emb_sl = self.encode_sl(x, sl_pospre)
        if self.model_type == 'GCN':

            emb_isme = self.encode_isme(x, isme.train_pos_edge_index, isme.edge_attr)
            emb_low = self.encode_low(
                x, low_exp.train_pos_edge_index, low_exp.edge_attr
            )
            emb_par = self.encode_par(x, para.train_pos_edge_index, para.edge_attr)
            emb_fold_change = self.fold_change(
                x, emb_fold_change.train_pos_edge_index, emb_fold_change.edge_attr
            )
        else:
            emb_isme = self.encode_isme(x, isme.train_pos_edge_index)
            emb_low = self.encode_low(x, low_exp.train_pos_edge_index)
            emb_par = self.encode_par(x, para.train_pos_edge_index)
            emb_fold_change = self.fold_change(x, emb_fold_change.train_pos_edge_index)

        new_features = []
        features_alls = torch.zeros((x.shape[0], 1))

        features_alls = features_alls.detach().cpu().numpy()
        if self.data_aba:
            if self.delete_EM:

                emb_all = [emb_sl, emb_low, emb_par, emb_fold_change]
            if self.delete_Es:

                emb_all = [emb_sl, emb_isme, emb_par, emb_fold_change]
            if self.delete_par:

                emb_all = [emb_sl, emb_isme, emb_low, emb_fold_change]
            if self.delete_L1000:

                emb_all = [emb_sl, emb_isme, emb_low, emb_par]
            emb_isme = torch.unsqueeze(emb_all[1], 1)
            emb_low = torch.unsqueeze(emb_all[2], 1)
            emb_par = torch.unsqueeze(emb_all[3], 1)
            print(emb_isme.shape)
            print(emb_low.shape)
            emb_low = self.decoder2(emb_isme, emb_low)
            emb_par = self.decoder2(emb_isme, emb_par)

            out1 = self.decoder(emb_low, emb_isme)
            out2 = self.decoder(emb_par, emb_isme)
            out1 = torch.squeeze(out1, 1)
            out2 = torch.squeeze(out2, 1)

            out = torch.cat([out1, out2], dim=-1)
            # out = self.aggragator(out)
            # output = self.predictor(out)
            out = torch.squeeze(out, 1)
            # emb_all = [emb_sl,out1,out2]
            emb_all = [emb_sl, out1, out2]
            if label == 'train':
                features_all = torch.zeros((sl_pos.shape[1] + sl_neg.shape[1], 1))
                new_feature = []
                for emb in emb_all:

                    emb_feature = self.decode(emb, sl_pos, sl_neg)
                    # z = self.decoder2(emb_feature,emb_feature)
                    new_feature.append(emb_feature)
                    features_all = torch.cat((features_all, emb_feature), dim=1)
            else:
                features_all = torch.zeros((sl_pos_back.shape[1] + sl_neg.shape[1], 1))
                new_feature = []
                for emb in emb_all:

                    emb_feature = self.decode(emb, sl_pos_back, sl_neg)
                    # z = self.decoder2(emb_feature,emb_feature)
                    new_feature.append(emb_feature)
                    features_all = torch.cat((features_all, emb_feature), dim=1)

            # features_all = torch.zeros((sl_pos_back.shape[1] + sl_neg.shape[1], 1))
            # new_feature = []
            # for emb in emb_all:

            #     emb_feature = self.decode(emb, sl_pos_back, sl_neg)
            #     # z = self.decoder2(emb_feature,emb_feature)
            #     new_feature.append(emb_feature)
            #     features_all = torch.cat((features_all, emb_feature), dim=1)

            features_all = features_all[:, 1:]
            hidden = self.linear0(features_all)
            # hidden = F.relu(hidden)

            hidden = self.bn1(hidden)
            hidden = self.linear2(hidden)
            # hidden = F.relu(hidden)
            hidden = F.dropout(hidden, p=0.5, training=self.training)
            hidden = self.bn2(hidden)
            hidden = self.linear3(hidden)
            y_pred = torch.sigmoid(hidden.squeeze(1))
        else:
            emb_all = [emb_sl, emb_isme, emb_low, emb_par, emb_fold_change]
            # emb_all_cp=copy.deepcopy(emb_all)
            # emb_all = [emb_sl, emb_isme, emb_low,emb_fold_change, emb_par]
            # emb_all=[emb_all[0],emb_all[4],emb_all[3],emb_all[1],emb_all[4]]
            # emb_all=[emb_all[0],emb_all[4],emb_all[3],emb_all[2],emb_all[3]]
            # emb_all=[emb_all[0],emb_all[3],emb_all[1],emb_all[4],emb_all[4]]
            # emb_all=[emb_all[0],emb_all[1],emb_all[2],emb_all[3],emb_all[4]]
            # emb_all = [emb_sl]

            emb_isme = torch.unsqueeze(emb_all[1], 1)
            emb_low = torch.unsqueeze(emb_all[2], 1)
            emb_par = torch.unsqueeze(emb_all[3], 1)
            print(emb_isme.shape)
            print(emb_low.shape)
            emb_low = self.decoder2(emb_isme, emb_low)
            emb_par = self.decoder2(emb_isme, emb_par)

            out1 = self.decoder(emb_low, emb_isme)
            out2 = self.decoder(emb_par, emb_isme)
            out1 = torch.squeeze(out1, 1)
            out2 = torch.squeeze(out2, 1)

            out = torch.cat([out1, out2], dim=-1)
            # out = self.aggragator(out)
            # output = self.predictor(out)
            out = torch.squeeze(out, 1)
            # emb_all = [emb_sl,out1,out2]
            emb_all = [emb_sl, out1, out2, emb_fold_change]
            if label == 'train':
                # emb_all=emb_all_cp
                features_all = torch.zeros((sl_pos.shape[1] + sl_neg.shape[1], 1))
                new_feature = []
                for emb in emb_all:

                    emb_feature = self.decode(emb, sl_pos, sl_neg)
                    # z = self.decoder2(emb_feature,emb_feature)
                    new_feature.append(emb_feature)
                    features_all = torch.cat((features_all, emb_feature), dim=1)
            else:
                features_all = torch.zeros((sl_pos_back.shape[1] + sl_neg.shape[1], 1))
                new_feature = []
                # emb_all=emb_all_cp
                for emb in emb_all:

                    emb_feature = self.decode(emb, sl_pos_back, sl_neg)
                    # z = self.decoder2(emb_feature,emb_feature)
                    new_feature.append(emb_feature)
                    features_all = torch.cat((features_all, emb_feature), dim=1)

            features_all = features_all[:, 1:]
            hidden = self.linear1(features_all)
            # hidden = F.relu(hidden)

            hidden = self.bn1(hidden)
            hidden = self.linear2(hidden)
            # hidden = F.relu(hidden)
            hidden = F.dropout(hidden, p=0.5, training=self.training)
            hidden = self.bn2(hidden)
            hidden = self.linear3(hidden)
            y_pred = torch.sigmoid(hidden.squeeze(1))
        return y_pred

    def get_emb(
        self,
        x,
        sl_pos,
        sl_neg,
        kg_ppi,
        kg_reactome,
        kg_corum,
        fold_change,
        dit,
        rank_id,
    ):
        emb_sl = self.encode_sl(x, sl_pos)
        esl, eslindex = torch.sort(torch.sum(torch.abs(emb_sl), 1), descending=True)
        sl_label = [eslindex.detach().cpu().numpy().tolist().index(i) for i in rank_id]
        emb_ppi = self.encode_ppi(x, kg_ppi)
        eppi, eppiindex = torch.sort(torch.sum(torch.abs(emb_ppi), 1), descending=True)
        ppi_label = [
            eppiindex.detach().cpu().numpy().tolist().index(i) for i in rank_id
        ]
        emb_reactome = self.encode_reactome(x, kg_reactome)
        ereac, emb_reactomeindex = torch.sort(
            torch.sum(torch.abs(emb_reactome), 1), descending=True
        )
        emb_reactomeindex_label = [
            emb_reactomeindex.detach().cpu().numpy().tolist().index(i) for i in rank_id
        ]
        emb_corum = self.encode_corum(x, kg_corum)
        ecorum, ecorumindex = torch.sort(
            torch.sum(torch.abs(emb_corum), 1), descending=True
        )
        ecorumindex_label = [
            ecorumindex.detach().cpu().numpy().tolist().index(i) for i in rank_id
        ]
        fold_changes = self.encode_corum(x, fold_change)
        ecorum, fold_changeindex = torch.sort(
            torch.sum(torch.abs(fold_changes), 1), descending=True
        )
        fold_change_label = [
            fold_changeindex.detach().cpu().numpy().tolist().index(i) for i in rank_id
        ]

        emb_all = [sl_pos, emb_ppi, emb_reactome, emb_corum, fold_changes]

        # concat all type of edge embedding
        features_all = torch.zeros((x.shape[0], 1))
        new_feature = []
        for emb in emb_all:
            # emb=emb.cuda()
            # sl_pos=sl_pos.cuda()
            # sl_neg=sl_neg.cuda()
            # emb_feature = self.decode(emb, sl_pos, sl_neg)
            # z = self.decoder2(emb_feature,emb_feature)
            new_feature.append(emb)
            features_all = torch.cat((features_all, emb), dim=1)

        features_all = features_all[:, 1:]
        # features_key = features_all[:,16:]
        features_all0 = features_all[:, 16:32]
        features_all0 = torch.sum(features_all0, 1)
        features_all0 = torch.unsqueeze(features_all0, 1)
        # features_all=torch.sum(features_all,1)
        # features_all=torch.unsqueeze(features_all,1)
        features_all1 = features_all[:, 16:32]
        features_all1 = torch.sum(features_all1, 1)
        features_all1 = torch.unsqueeze(features_all1, 1)
        # _,features_all1=torch.sort(torch.sum(features_all1,1), descending=True)
        features_all2 = features_all[:, 32:48]
        features_all2 = torch.sum(features_all2, 1)
        features_all2 = torch.unsqueeze(features_all2, 1)
        # _,features_all2=torch.sort(torch.sum(features_all2,1), descending=True)
        features_all3 = features_all[:, 48:64]
        features_all3 = torch.sum(features_all3, 1)
        features_all3 = torch.unsqueeze(features_all3, 1)
        features_all = torch.sum(features_all, 1)
        features_all = torch.unsqueeze(features_all, 1)
        # _,features_all3=torch.sort(torch.sum(features_all3,1), descending=True)
        # aaabig=torch.matmul(features_key,features_key.transpose(0, 1))
        aaa = torch.matmul(features_all, features_all.transpose(0, 1))
        aaa0 = torch.matmul(features_all1, features_all1.transpose(0, 1))
        aaa1 = torch.matmul(features_all1, features_all1.transpose(0, 1))
        aaa2 = torch.matmul(features_all2, features_all2.transpose(0, 1))
        aaa3 = torch.matmul(features_all3, features_all3.transpose(0, 1))

        return aaa, aaa0, aaa1, aaa2, aaa3


class GlobalPredictor(nn.Module):
    def __init__(self, d_model=None, d_h=None, d_out=None, dropout=0.5):
        super(GlobalPredictor, self).__init__()
        self.predict_layer = nn.Sequential(
            collections.OrderedDict(
                [
                    # ('batchnorm', nn.BatchNorm1d(d_model)),
                    ('fc1', nn.Linear(d_model, d_h)),
                    ('tanh', nn.Tanh()),
                    ('dropout', nn.Dropout(dropout)),
                    ('fc2', nn.Linear(d_h, d_out)),
                ]
            )
        )

    def forward(self, x):
        x = self.predict_layer(x)
        return x


class AggregateLayer(nn.Module):
    def __init__(self, d_model=None, dropout=0.1):
        super(AggregateLayer, self).__init__()
        self.attn = nn.Sequential(
            collections.OrderedDict(
                [
                    ('layernorm', nn.LayerNorm(d_model)),
                    ('fc', nn.Linear(d_model, 1, bias=False)),
                    ('dropout', nn.Dropout(dropout)),
                    ('softmax', nn.Softmax(dim=1)),
                ]
            )
        )

    def forward(self, context):
        '''
        Parameters
        ----------
        context: token embedding from encoder (Transformer/LSTM)
                (batch_size, seq_len, embed_dim)
        '''
        weight = self.attn(context)
        # (batch_size, seq_len, embed_dim).T * (batch_size, seq_len, 1) *  ->
        # (batch_size, embed_dim, 1)
        output = torch.bmm(context.transpose(1, 2), weight)
        output = output.squeeze(2)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, ab_seq, ag_seq):

        for layer in self.layers:
            ag_seq, attn = layer(ab_seq, ag_seq)
        return ag_seq


class TransformerDecoder2(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, ab_seq, ag_seq):
        ab_seq = ab_seq
        ag_seq = ag_seq

        for layer in self.layers:
            ag_seq, attn = layer(ab_seq, ag_seq)
        return ag_seq


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, ab_seq):

        for layer in self.layers:
            ab_seq = layer(ab_seq)

        return ab_seq


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="relu"):
        super().__init__()
        self.pos = PositionalEncoding(d_model)
        self.attn2 = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, bias=False
        )  # test later if use true of false.
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, ab_seq):
        ### encoder, self-attention
        k = self.pos(ab_seq)
        seq_out, attention = self.attn2(k, k, value=k)
        src = k + self.dropout1(seq_out)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        out = src + self.dropout2(src2)
        out = self.norm2(out)
        return out


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="relu"):
        super().__init__()
        self.pos = PositionalEncoding(d_model)

        self.attn2 = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, bias=False
        )  # test later if use true of false.
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, ab_seq, ag_seq):
        ### query is antigen
        # k = self.pos(ab_seq)
        # q = self.pos(ag_seq)
        k = ab_seq
        q = ag_seq

        seq_out, attention = self.attn2(q, k, value=k)
        src = q + self.dropout1(seq_out)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        out = src + self.dropout2(src2)
        out = self.norm2(out)
        return out, attention


class TransformerDecoderLayer2(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="relu"):
        super().__init__()
        self.pos = PositionalEncoding(d_model)

        self.attn2 = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, bias=False
        )  # test later if use true of false.
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, ab_seq, ag_seq):
        ### query is one of multi-omic data

        k = ab_seq
        q = ag_seq

        seq_out, attention = self.attn2(q, k, value=k)
        src = q + self.dropout1(seq_out)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        out = src + self.dropout2(src2)
        out = self.norm2(out)
        return out, attention


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len=2000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # pe.shape = torch.Size([1, 5000, d_model])
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return x


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
