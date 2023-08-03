import math
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import pandas as pd
import numpy as np


class SynlethDB(Data):
    def __init__(self, num_nodes, sl_data, nosl_data):
        num_nodes = num_nodes
        num_edges = sl_data.shape[0]
        neg_num_edges = nosl_data.shape[0]
        feat_node_dim = 1
        feat_edge_dim = 1
        self.x = torch.ones(num_nodes, feat_node_dim)
        self.y = torch.randint(0, 2, (num_nodes,))
        self.edge_index = torch.tensor(
            sl_data[['gene1', 'gene2']].T.values, dtype=torch.long
        )
        self.edge_attr = torch.ones(num_edges, feat_edge_dim)
        self.neg_edge_index = torch.tensor(
            nosl_data[['gene1', 'gene2']].T.values, dtype=torch.long
        )
        self.neg_edge_attr = torch.ones(neg_num_edges, feat_edge_dim)


# related knowledge graph
class SynlethDB_omic(Data):
    def __init__(self, omic_data, types):
        self.type = types
        num_nodes = 12926
        num_edges = omic_data.shape[0]
        feat_node_dim = 1
        feat_edge_dim = 1
        self.x = torch.ones(num_nodes, feat_node_dim)
        self.y = torch.randint(0, 2, (num_nodes,))
        self.edge_index = torch.tensor(
            omic_data[['gene1', 'gene2']].T.values, dtype=torch.long
        )
        self.edge_attr = torch.abs(
            torch.tensor(omic_data[[self.type]].values, dtype=torch.long)
        )
        self.edge_attrs = torch.tensor(
            omic_data[[self.type]].values, dtype=torch.float32
        )


# random negative sample
def get_k_fold_data_random_neg(data, tpos_edge_index, tneg_edge_index, k=10):

    num_nodes = data.num_nodes

    row, col = data.edge_index
    num_edges = row.size(0)
    mask = row < col
    row, col = row[mask], col[mask]

    neg_row, neg_col = data.neg_edge_index
    neg_num_edges = neg_row.size(0)
    mask = neg_row < neg_col
    neg_row, neg_col = neg_row[mask], neg_col[mask]

    assert k > 1
    fold_size = num_edges // k

    perm = torch.randperm(num_edges)
    row, col = row[perm], col[perm]

    neg_perm = torch.randperm(neg_num_edges)
    neg_row, neg_col = neg_row[neg_perm], neg_col[neg_perm]

    res_neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)

    res_neg_adj_mask = res_neg_adj_mask.triu(diagonal=1).to(torch.bool)
    res_neg_adj_mask[row, col] = 0
    res_neg_row, res_neg_col = res_neg_adj_mask.nonzero(as_tuple=False).t()

    for j in range(k):
        val_start = j * fold_size
        val_end = (j + 1) * fold_size
        if j == k - 1:
            val_row, val_col = row[val_start:], col[val_start:]
            train_row, train_col = row[:val_start], col[:val_start]
        else:
            val_row, val_col = row[val_start:val_end], col[val_start:val_end]
            train_row, train_col = torch.cat(
                [row[:val_start], row[val_end:]], 0
            ), torch.cat([col[:val_start], col[val_end:]], 0)

        # val
        data.val_pos_edge_index = torch.stack([val_row, val_col], dim=0)
        # train
        data.train_pos_edge_index = torch.stack([train_row, train_col], dim=0)

        add_val = data.val_pos_edge_index.shape[1]
        add_train = data.train_pos_edge_index.shape[1]
        perm = torch.randperm(res_neg_row.size(0))[: add_val + add_train]
        res_neg_row, res_neg_col = res_neg_row[perm], res_neg_col[perm]

        res_r, res_c = res_neg_row[:add_val], res_neg_col[:add_val]
        data.val_neg_edge_index = torch.stack([res_r, res_c], dim=0)

        res_r, res_c = (
            res_neg_row[add_val : add_val + add_train],
            res_neg_col[add_val : add_val + add_train],
        )
        data.train_neg_edge_index = torch.stack([res_r, res_c], dim=0)

        data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)
        data.train_neg_edge_index = to_undirected(data.train_neg_edge_index)
        # data.test_pos_edge_index = torch.stack([pos_set_IDa,pos_set_IDb], dim=0)
        # data.test_neg_edge_index = torch.stack([neg_set_IDa,neg_set_IDb], dim=0)
        # data.test_pos_edge_index = to_undirected(tpos_edge_index)
        # data.test_neg_edge_index = to_undirected(tneg_edge_index)
        data.test_pos_edge_index = tpos_edge_index
        data.test_neg_edge_index = tneg_edge_index
        yield data


def train_test_split_edges_cv2(data, tpos_edge_index, tneg_edge_index, test_ratio=0.1):
    num_nodes = data.num_nodes
    row, col = data.edge_index

    data.edge_index = None
    num_edges = row.size(0)

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    n_t = int(math.floor(test_ratio * num_edges))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    r, c = row[:n_t], col[:n_t]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row[n_t:], col[n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)

    data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))[:num_edges]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    row, col = neg_row[:n_t], neg_col[:n_t]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_t:], neg_col[n_t:]
    data.train_neg_edge_index = torch.stack([row, col], dim=0)

    data.train_neg_edge_index = to_undirected(data.train_neg_edge_index)
    data.test_pos_edge_index = to_undirected(tpos_edge_index)
    data.test_neg_edge_index = to_undirected(tneg_edge_index)
    yield data


def train_test_split_edges_omic(data, test_ratio=0.1):
    num_nodes = data.num_nodes
    row, col = data.edge_index

    data.edge_index = None
    num_edges = row.size(0)

    # Return upper triangular portion.
    # mask = row < col
    # row, col = row[mask], col[mask]

    n_t = int(math.floor(test_ratio * num_edges))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    r, c = row[:n_t], col[:n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row[n_t:], col[n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)

    # data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))[:num_edges]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    row, col = neg_row[:n_t], neg_col[:n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_t:], neg_col[n_t:]
    data.train_neg_edge_index = torch.stack([row, col], dim=0)

    # data.train_neg_edge_index = to_undirected(data.train_neg_edge_index)
    return data


def train_test_split_edges_cv3(data, tpos_edge_index, tneg_edge_index, test_ratio=0.1):
    num_nodes = data.num_nodes
    row, col = data.edge_index
    # listraw=[]
    # listcol=[]
    # index_list=[]
    # n_t = int(math.floor(test_ratio * num_edges))
    # rest_edge=num_edges-n_t
    # for index,(i,a) in enumerate(zip(row,col)):
    #     if int(i) not in listraw and len(listraw)== rest_edge:
    #         listraw.append(int(i))
    #         listcol.append(int(col[index]))
    #         index_list.append(index)
    # index_list=torch.Tensor(index_list)

    # row=torch.index_select(row, dim = 1, index =index_list)
    data.edge_index = None
    num_edges = row.size(0)

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    n_t = int(math.floor(test_ratio * num_edges))
    listraw = []
    listcol = []
    index_list = []
    n_t = int(math.floor(test_ratio * num_edges))
    rest_edge = num_edges - n_t
    for index, (i, a) in enumerate(zip(row, col)):
        if len(listraw) >= rest_edge:
            break
        if int(i) not in listraw:
            listraw.append(int(i))
            listcol.append(int(col[index]))
            index_list.append(index)

    index_list = np.array(index_list)
    index_list = torch.from_numpy(index_list)

    row = torch.index_select(row, dim=0, index=index_list)

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    r, c = row[:n_t], col[:n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row[n_t:], col[n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)

    data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))[:num_edges]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    row, col = neg_row[:n_t], neg_col[:n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_t:], neg_col[n_t:]
    data.train_neg_edge_index = torch.stack([row, col], dim=0)

    data.train_neg_edge_index = to_undirected(data.train_neg_edge_index)
    yield data


def train_test_split_edges_cv2s(data, tpos_edge_index, tneg_edge_index, test_ratio=0.1):
    num_nodes = data.num_nodes
    row, col = data.edge_index

    data.edge_index = None
    num_edges = row.size(0)

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    n_t = int(math.floor(test_ratio * num_edges))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    r, c = row[:n_t], col[:n_t]
    data.val_neg_edge_index = torch.stack([r, c], dim=0)

    r, c = row[n_t:], col[n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)

    data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))[:num_edges]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    row, col = neg_row[:n_t], neg_col[:n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_t:], neg_col[n_t:]
    data.train_neg_edge_index = torch.stack([row, col], dim=0)

    data.train_neg_edge_index = to_undirected(data.train_neg_edge_index)
    data.te_pos_edge_index = to_undirected(tpos_edge_index)
    data.te_neg_edge_index = to_undirected(tneg_edge_index)
    yield data
