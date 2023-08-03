import torch
import pandas as pd
import numpy as np
import os
import sklearn
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from .data_prepare import (
    SynlethDB,
    get_k_fold_data_random_neg,
    train_test_split_edges_cv2,
)
import math
from scipy import stats
from scipy.stats import t
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import pandas as pd
import numpy as np
from .metrics import get_metric_func


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
        num_nodes = 12918
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


def train_test_split_edges_omic_for_weight(data, test_ratio=0.1):
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


def construct_multi_omic(data, isme_data, lowexp_data, par_data, fold_data):

    synlethdb_isme = SynlethDB_omic(isme_data, 'label')
    synlethdb_isme = train_test_split_edges_omic(synlethdb_isme, test_ratio=0)

    synlethdb_low = SynlethDB_omic(lowexp_data, 'label')
    synlethdb_low = train_test_split_edges_omic(synlethdb_low, test_ratio=0)

    synlethdb_par = SynlethDB_omic(par_data, 'label')
    synlethdb_par = train_test_split_edges_omic(synlethdb_par, test_ratio=0)
    synlethdb_go_F = SynlethDB_omic(fold_data, 'weight')
    synlethdb_go_F = train_test_split_edges_omic(synlethdb_go_F, test_ratio=0)

    return synlethdb_isme, synlethdb_low, synlethdb_par, synlethdb_go_F


def cal_confidence_interval(data, confidence=0.95):
    data = 1.0 * np.array(data)
    n = len(data)
    sample_mean = np.mean(data)
    se = stats.sem(data)
    t_ci = t.ppf((1 + confidence) / 2.0, n - 1)  # T value of Confidence Interval
    bound = se * t_ci
    return sample_mean, bound


def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float)
    link_labels[: pos_edge_index.size(1)] = 1.0
    return link_labels


def evaluate(y_true, y_score, pos_threshold=0.5):
    # from sklearn.metrics import precision_score
    # from sklearn.metrics import recall_score
    # from sklearn.metrics import matthews_corrcoef
    metrics_func = ['roc_auc', 'prc_auc', 'matthews_corrcoef', 'recall', 'precision']
    auc_test = roc_auc_score(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    aupr_test = auc(recall, precision)
    f1_test = f1_score(y_true, y_score > pos_threshold)

    auc_test = get_metric_func(metrics_func[0])(y_true, y_score)
    aupr_test = get_metric_func(metrics_func[1])(y_true, y_score)
    mcc_test = get_metric_func(metrics_func[2])(y_true, y_score)
    precision = get_metric_func(metrics_func[3])(y_true, y_score)
    recall = get_metric_func(metrics_func[4])(y_true, y_score)
    # s1=[1 if data >=pos_threshold else 0 for data in y_score.numpy().tolist()]
    # precision=precision_score(y_true.numpy().tolist(),s1 ,average=None)[0]
    # recall=recall_score(y_true.numpy().tolist(),s1 , average=None)[0]
    return auc_test, aupr_test, mcc_test, precision, recall


def train(
    model,
    optimizer,
    synlethdb_sl,
    synlethdb_isme,
    synlethdb_low,
    synlethdb_par,
    synlethdb_go_F,
):
    model.train()
    optimizer.zero_grad()

    pos_edge_index = synlethdb_sl.train_pos_edge_index
    neg_edge_index = synlethdb_sl.train_neg_edge_index

    link_pred = model(
        synlethdb_sl.x,
        pos_edge_index,
        neg_edge_index,
        synlethdb_isme,
        synlethdb_low,
        synlethdb_par,
        synlethdb_go_F,
    )

    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
    loss = F.binary_cross_entropy(link_pred, link_labels)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def valid(
    model, synlethdb_sl, synlethdb_isme, synlethdb_low, synlethdb_par, synlethdb_go_F
):
    model.eval()

    pos_edge_index = synlethdb_sl.val_pos_edge_index
    neg_edge_index = synlethdb_sl.val_neg_edge_index

    perfs = []
    link_pred = model(
        synlethdb_sl.x,
        pos_edge_index,
        neg_edge_index,
        synlethdb_isme,
        synlethdb_low,
        synlethdb_par,
        synlethdb_go_F,
    )

    link_labels = get_link_labels(pos_edge_index, neg_edge_index)

    auc_test, aupr_test, f1_test, precision, recall = evaluate(
        link_labels.cpu(), link_pred.cpu()
    )
    perfs.extend([auc_test, aupr_test, f1_test, precision, recall])

    return perfs


def change(redic, data):
    data['gene1'] = data['gene1'].apply(lambda x: redic[x])
    data['gene2'] = data['gene2'].apply(lambda x: redic[x])
    data.insert(1, 'label', np.ones(len(data)))
    return data


def process(redic):
    '''
    description: isme_data represents EM data
    lowexp_data represents ES data
    par_data represents paralog data
    fold_data represents L1000 data
    return {processed multi-omics data}
    '''
    isme_data = pd.read_csv(
        '/home/testenv/SyntheticLethal/final_code/Data/A549/is_me_or_not_cutoff_0.05_symmetric_matrix.csv'
    )
    lowexp_data = pd.read_csv(
        '/home/testenv/SyntheticLethal/final_code/Data/A549/low_expr_low_geneEffect_symmetric_matrix.csv'
    )
    par_data = pd.read_csv(
        '/home/testenv/SyntheticLethal/final_code/Data/A549/paralogs_symmetric_matrix.csv'
    )
    fold_data = pd.read_csv(
        '/home/testenv/SyntheticLethal/final_code/Data/A549/fold.csv'
    )
    isme_data = change(redic, isme_data)
    lowexp_data = change(redic, lowexp_data)
    par_data = change(redic, par_data)
    fold_data = change(redic, fold_data)
    # fold_change=fold_change[fold_change['weight']>3]
    # go_F_data=change(redic,fold_change)

    return isme_data, lowexp_data, par_data, fold_data


@torch.no_grad()
def test(
    model, synlethdb_sl, synlethdb_isme, synlethdb_low, synlethdb_par, fold_change
):
    model.eval()

    pos_edge_index = synlethdb_sl.tpos_edge_index
    neg_edge_index = synlethdb_sl.tneg_edge_index

    perfs = []
    link_pred = model(
        synlethdb_sl.x,
        pos_edge_index,
        neg_edge_index,
        synlethdb_isme,
        synlethdb_low,
        synlethdb_par,
        fold_change,
    )

    link_labels = get_link_labels(pos_edge_index, neg_edge_index)

    auc_test, aupr_test, f1_test, precision, recall = evaluate(
        link_labels.cpu(), link_pred.cpu()
    )
    perfs.extend([auc_test, aupr_test, f1_test, precision, recall])

    return perfs


def main(args, data_path, model, epochs, lr=0.01):

    data = pd.read_csv(args.A549_data_path)

    sl_data = data[data['label'] == 1]
    nosl_data = data[data['label'] != 1]

    pos_set_IDa = []
    pos_set_IDb = []
    neg_set_IDb = []
    neg_set_IDa = []

    f = open(args.A549_gene, "r")
    lines = f.readlines()  # 读取全部内容 ，并以列表方式返回
    num_nodes = len(lines)
    all_gene = []
    for line in lines:
        all_gene.append(line)
    f.close()
    dit = {}
    redic = {}
    for i, lie in enumerate(all_gene):
        dit[i] = lie[:-1]
        redic[lie[:-1]] = i
    # 是否进行外部测试，构建外部测试集
    if args.out_test:
        df = pd.read_csv(args.HT29_data_path, delimiter=',')
        # df=sklearn.utils.shuffle(df)
        gene_left = df.loc[:, 'gene1'].tolist()
        gene_right = df.loc[:, 'gene2'].tolist()
        indexa = []
        # indexb=[]
        for inde, (a, b) in enumerate(zip(gene_left, gene_right)):
            if a in redic.keys() and b in redic.keys():
                indexa.append(inde)
        df = df.loc[indexa, :]

        testsl_data = df[df['label'] == 1]
        testsl_data['n_A'] = testsl_data['gene1'].apply(lambda x: redic[x])
        testsl_data['n_b'] = testsl_data['gene2'].apply(lambda x: redic[x])
        testnosl_data = df[df['label'] != 1]
        testnosl_data['n_A'] = testnosl_data['gene1'].apply(lambda x: redic[x])
        testnosl_data['n_b'] = testnosl_data['gene2'].apply(lambda x: redic[x])
        tpos_edge_index = torch.tensor(
            testsl_data[['n_A', 'n_b']].T.values, dtype=torch.long
        )
        tneg_edge_index = torch.tensor(
            testnosl_data[['n_A', 'n_b']].T.values, dtype=torch.long
        )
    else:
        tneg_edge_index = None
        tpos_edge_index = None

    sl_data['gene1'] = sl_data['gene1'].apply(lambda x: redic[x])
    sl_data['gene2'] = sl_data['gene2'].apply(lambda x: redic[x])
    nosl_data['gene1'] = nosl_data['gene1'].apply(lambda x: redic[x])
    nosl_data['gene2'] = nosl_data['gene2'].apply(lambda x: redic[x])
    synlethdb = SynlethDB(num_nodes, sl_data, nosl_data)
    # k_fold=train_test_split_edges_cv2(synlethdb,tpos_edge_index,tneg_edge_index, test_ratio=0.1)
    # 进行五折交叉验证划分数据集
    k_fold = get_k_fold_data_random_neg(
        synlethdb, tpos_edge_index, tneg_edge_index, k=5
    )

    isme_data, lowexp_data, par_data, fold_data = process(redic)

    synlethdb_isme, synlethdb_low, synlethdb_par, synlethdb_go_F = construct_multi_omic(
        data, isme_data, lowexp_data, par_data, fold_data
    )
    print("data prepare finished!")
    k_val_best_auc = []
    k_val_best_aupr = []
    k_val_best_MCC = []
    k_val_best_recall = []
    k_val_best_precision = []
    k_test_best_auc = []
    k_test_best_aupr = []
    k_test_best_MCC = []
    k_test_best_recall = []
    k_test_best_precision = []
    k = 0

    model_name = 'A549'
    for k_data in k_fold:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        explr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        best_loss = torch.tensor(10000)
        print('start one fold:')
        best_score_sum = 0

        for epoch in range(0, epochs):
            train_loss = train(
                model,
                optimizer,
                k_data,
                synlethdb_isme,
                synlethdb_low,
                synlethdb_par,
                synlethdb_go_F,
            )
            val_perf = valid(
                model,
                k_data,
                synlethdb_isme,
                synlethdb_low,
                synlethdb_par,
                synlethdb_go_F,
            )

            explr_scheduler.step()

            score_sum = np.array(val_perf).sum()

            counst = 0
            if train_loss < best_loss:
                best_loss = train_loss
                counst = 0
            else:
                counst += 1
                if counst == 30:
                    if args.out_test:

                        break

            if best_score_sum < score_sum:
                jilu = 0
                best_score_sum = score_sum
                torch.save(model, '%s/%s_%d' % (args.save_path, model_name, k) + '.pkl')

                best_val_perf_auc = val_perf[0]
                best_val_perf_aupr = val_perf[1]
                best_val_perf_MCC = val_perf[2]
                best_val_perf_pre = val_perf[3]
                best_val_perf_recall = val_perf[4]

            log = 'Epoch: {:03d}, Loss: {:.4f}, \
                Val_AUC: {:.4f}, Val_AUPR:{:.4f}, Val_MCC:{:.4f}, Val_precision:{:.4f}, Val_recall:{:.4f}'
            print(
                log.format(
                    epoch,
                    train_loss,
                    val_perf[0],
                    val_perf[1],
                    val_perf[2],
                    val_perf[3],
                    val_perf[4],
                )
            )

        k += 1
        k_val_best_auc.append(best_val_perf_auc)
        k_val_best_aupr.append(best_val_perf_aupr)
        k_val_best_MCC.append(best_val_perf_MCC)
        k_val_best_precision.append(best_val_perf_pre)
        k_val_best_recall.append(best_val_perf_recall)
        if args.out_test:

            test_perf = test(
                model,
                k_data,
                synlethdb_isme,
                synlethdb_low,
                synlethdb_par,
                fold_data,
            )
            k_test_best_auc.append(test_perf[0])
            k_test_best_aupr.append(test_perf[1])
            k_test_best_MCC.append(test_perf[2])
            k_test_best_recall.append(test_perf[3])
            k_test_best_precision.append(test_perf[4])
    if args.out_test:

        print('auc:', cal_confidence_interval(k_test_best_auc))
        print('aupr:', cal_confidence_interval(k_test_best_aupr))
        print('MCC:', cal_confidence_interval(k_test_best_MCC))
        print('precision:', cal_confidence_interval(k_val_best_precision))
        print('recall:', cal_confidence_interval(k_val_best_recall))
    else:
        print('auc:', cal_confidence_interval(k_val_best_auc))
        print('aupr:', cal_confidence_interval(k_val_best_aupr))
        print('MCC:', cal_confidence_interval(k_val_best_MCC))
        print('precision:', cal_confidence_interval(k_val_best_precision))
        print('recall:', cal_confidence_interval(k_val_best_recall))
