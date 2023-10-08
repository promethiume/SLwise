import torch
import pandas as pd
import numpy as np
import os
import sklearn
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from .data_prepare import (
    SynlethDB,
    get_k_fold_data_random_neg,
    # train_test_split_edges_cv2,
    train_test_split_edges_omic,
    get_k_fold_data_random_negcv2,
    get_k_fold_data_random_negcv3,
    SynlethDB_omic,
)
from .metrics import get_metric_func
from scipy import stats
from scipy.stats import t
import torch.nn.functional as F
import pickle
from sklearn.model_selection import ShuffleSplit, KFold, StratifiedKFold

# from pytorchtools import EarlyStopping
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


def cal_precision_recall(targets, predictions):

    # targets是真实值，predictions是预测值
    # pred = predictions[:k]
    num_hit = len(set(predictions).intersection(set(targets)))
    precision = float(num_hit / predictions)
    recall = float(num_hit / targets)
    return precision, recall


def evaluate(y_true, y_score, pos_threshold=0.5):
    # from sklearn.metrics import precision_score
    # from sklearn.metrics import recall_score
    # from sklearn.metrics import matthews_corrcoef
    metrics_func = ['roc_auc', 'prc_auc', 'matthews_corrcoef', 'recall', 'precision']
    if type(y_true) == torch.Tensor:
        ys = y_true
        yr = y_score
        y_true = y_true.detach().cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
    auc_test = roc_auc_score(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # aupr_test = auc(recall, precision)
    # f1_test = f1_score(y_true, y_score > pos_threshold)

    auc_test = get_metric_func(metrics_func[0])(y_true, y_score)
    aupr_test = get_metric_func(metrics_func[1])(y_true, y_score)
    matthews_corrcoef = get_metric_func(metrics_func[2])(y_true, y_score)
    # y_true=np.array(list(map(np.int_, y_true)))
    # y_score=np.array(list(map(np.int_, y_score)))
    # precision,recall=cal_precision_recall(y_true, y_score)
    # from torchmetrics import MetricCollection, Accuracy, Precision, Recall

    # metric_collection = MetricCollection([
    #     Accuracy(),
    #     Precision(num_classes=2, average='macro'),
    #     Recall(num_classes=2, average='macro')
    # ])
    # print(metric_collection(ys,yr))
    # precision=int(metric_collection(ys,yr)['Precision'])
    # recall=int(metric_collection(ys,yr)['Recall'])
    # y_true = list(y_true)
    # y_pred = list(y_pred)
    # from sklearn.metrics import precision_score, recall_score

    # recall = recall_score(y_true, y_pred, average='macro')
    # precision = precision_score(y_true, y_pred, average='macro')
    precision = get_metric_func(metrics_func[3])(y_true, y_score)
    recall = get_metric_func(metrics_func[4])(y_true, y_score)
    # s1=[1 if data >=pos_threshold else 0 for data in y_score.numpy().tolist()]
    # precision=precision_score(y_true.numpy().tolist(),s1 ,average=None)[0]
    # recall=recall_score(y_true.numpy().tolist(),s1 , average=None)[0]
    return auc_test, aupr_test, matthews_corrcoef, precision, recall


def train(
    model,
    optimizer,
    synlethdb_sl,
    synlethdb_isme,
    synlethdb_low,
    synlethdb_par,
    fold_change,
):
    model.train()
    optimizer.zero_grad()

    pos_edge_index = synlethdb_sl.train_pos_edge_index
    pos_edge_index_pre = synlethdb_sl.train_pos_edge_index[
        :, : int(1 / 2 * synlethdb_sl.train_pos_edge_index.shape[1])
    ]
    pos_edge_index_back = synlethdb_sl.train_pos_edge_index[
        :, int(1 / 2 * synlethdb_sl.train_pos_edge_index.shape[1]) :
    ]
    neg_edge_index = synlethdb_sl.train_neg_edge_index

    link_pred = model(
        synlethdb_sl.x,
        synlethdb_sl.train_pos_edge_index,
        pos_edge_index_pre,
        pos_edge_index_back,
        neg_edge_index,
        synlethdb_isme,
        synlethdb_low,
        synlethdb_par,
        fold_change,
        'train',
    )

    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
    loss = F.binary_cross_entropy(link_pred, link_labels)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def validcommon(
    model,
    num_nodes,
    tpos_edge_index,
    tneg_edge_index,
    synlethdb_isme,
    synlethdb_low,
    synlethdb_par,
    fold_change,
):
    model.eval()

    pos_edge_index = tpos_edge_index
    neg_edge_index = tneg_edge_index

    pos_edge_index_pre = tpos_edge_index[:, : int(1 / 2 * tpos_edge_index.shape[1])]
    pos_edge_index_back = tpos_edge_index[:, int(1 / 2 * tpos_edge_index.shape[1]) :]
    perfs = []
    link_pred = model(
        torch.ones(num_nodes, 1),
        # pos_edge_index,
        tpos_edge_index,
        pos_edge_index_pre,
        pos_edge_index_back,
        neg_edge_index,
        synlethdb_isme,
        synlethdb_low,
        synlethdb_par,
        fold_change,
        'test',
    )

    link_labels = get_link_labels(pos_edge_index_back, neg_edge_index)

    auc_test, aupr_test, mcc_test, precision, recall = evaluate(
        link_labels.cpu(), link_pred.cpu()
    )
    perfs.extend([auc_test, aupr_test, mcc_test, precision, recall])

    return perfs


@torch.no_grad()
def valid(
    model,
    num_nodes,
    tpos_edge_index,
    tneg_edge_index,
    synlethdb_isme,
    synlethdb_low,
    synlethdb_par,
    fold_change,
):
    model.eval()

    pos_edge_index = tpos_edge_index
    neg_edge_index = tneg_edge_index

    pos_edge_index_pre = tpos_edge_index[:, : int(1 / 2 * tpos_edge_index.shape[1])]
    pos_edge_index_back = tpos_edge_index[:, int(1 / 2 * tpos_edge_index.shape[1]) :]
    perfs = []
    link_pred = model(
        torch.ones(num_nodes, 1),
        # pos_edge_index,
        tpos_edge_index,
        pos_edge_index_pre,
        pos_edge_index_back,
        neg_edge_index,
        synlethdb_isme,
        synlethdb_low,
        synlethdb_par,
        fold_change,
        'test',
    )

    link_labels = get_link_labels(pos_edge_index_back, neg_edge_index)

    auc_test, aupr_test, mcc_test, precision, recall = evaluate(
        link_labels.cpu(), link_pred.cpu()
    )
    perfs.extend([auc_test, aupr_test, mcc_test, precision, recall])

    # lista = [auc_test, aupr_test, mcc_test, precision, recall, 2]
    # datass = pd.DataFrame([lista])

    # datass.to_csv('/home/intern/SyntheticLethal/final_code/Res/HT29/test.csv')
    return perfs, auc_test, aupr_test, mcc_test, precision, recall


@torch.no_grad()
def test(
    model, synlethdb_sl, synlethdb_isme, synlethdb_low, synlethdb_par, fold_change
):
    model.eval()

    pos_edge_index = synlethdb_sl.test_pos_edge_index
    neg_edge_index = synlethdb_sl.test_neg_edge_index
    pos_edge_index_pre = synlethdb_sl.test_pos_edge_index[
        :, : int(1 / 2 * synlethdb_sl.test_pos_edge_index.shape[1])
    ]
    pos_edge_index_back = synlethdb_sl.test_pos_edge_index[
        :, int(1 / 2 * synlethdb_sl.test_pos_edge_index.shape[1]) :
    ]
    perfs = []
    link_pred = model(
        synlethdb_sl.x,
        # pos_edge_index,
        synlethdb_sl.test_pos_edge_index,
        pos_edge_index_pre,
        pos_edge_index_back,
        neg_edge_index,
        synlethdb_isme,
        synlethdb_low,
        synlethdb_par,
        fold_change,
        'test',
    )

    link_labels = get_link_labels(pos_edge_index_back, neg_edge_index)

    auc_test, aupr_test, mcc_test, precision, recall = evaluate(
        link_labels.cpu(), link_pred.cpu()
    )
    perfs.extend([auc_test, aupr_test, mcc_test, precision, recall])

    return perfs


def change(redic, data):
    data['gene1'] = data['gene1'].apply(lambda x: redic[x])
    data['gene2'] = data['gene2'].apply(lambda x: redic[x])
    data.insert(1, 'label', np.ones(len(data)))
    return data


def remove_invalid(isme_data, redic, node):
    # isme_data = isme_data.rename(
    #     columns={"rowGene": "gene1", "colGene": "gene2"}, errors="raise"
    # )
    isme_data = isme_data[
        isme_data['gene1'].isin(redic.keys()) & isme_data['gene2'].isin(redic.keys())
    ]
    isme_data = isme_data[isme_data['gene1'].isin(node) | isme_data['gene2'].isin(node)]
    return isme_data


def process_data(redic, node_dict):
    '''
    description: combined_score_data represents EM data
    low_data represents ES data
    par_data represents paralog data
    fold_data represents L1000 data
    return {processed multi-omics data}
    '''

    isme_data = pd.read_table(
        '/home/pumengchen/DataSource/SLdata/temp/temp_mutational_exclu_list/HT29_me_list.txt',
        sep='\t',
    )
    isme_data = isme_data[isme_data['is_me_p0.05'] == 1]

    lowexp_data = pd.read_table(
        '/home/pumengchen/DataSource/SLdata/temp/temp_low_expr_low_geneEffect_list/HT29_low_expr_low_geneEffect_list.txt',
        sep=',',
    )
    lowexp_data.rename(columns={"rowGene": "gene1", "colGene": "gene2"}, errors="raise")
    par_data = pd.read_csv('/home/intern/SyntheticLethal/final_code/Data/HT29/p.csv')
    fold_data = pd.read_table(
        '/home/pumengchen/DataSource/SLdata/temp/temp_pair_gene_fold_change_value_list/HT29_pair_gene_fold_change_value_list.txt',
        sep=',',
    )
    fold_data = fold_data[fold_data['value'] > 3]
    lowexp_data = lowexp_data.rename(
        columns={"rowGene": "gene1", "colGene": "gene2"}, errors="raise"
    )
    fold_data = fold_data.rename(
        columns={"rowGene": "gene1", "colGene": "gene2"}, errors="raise"
    )
    # combined_score_data=pd.read_csv('/home/chengkaiyang/git_code/xiugai/new/HT29_process/is_me_or_not_cutoff_0.05_symmetric_matrix.csv')
    # low_data=pd.read_csv('/home/chengkaiyang/git_code/xiugai/new/HT29_process/low_expr_low_geneEffect_symmetric_matrix.csv')
    # par_data=pd.read_csv('/home/chengkaiyang/git_code/xiugai/new/HT29_process/paralogs_symmetric_matrix.csv')
    isme_data = remove_invalid(isme_data, redic, node_dict)
    low_data = remove_invalid(lowexp_data, redic, node_dict)
    par_data = remove_invalid(par_data, redic, node_dict)
    fold_data = remove_invalid(fold_data, redic, node_dict)
    # combined_score_data=pd.read_csv('/home/chengkaiyang/MGE4SL/update_data/is_me_or_not_p0.05_symmetric_matrix.csv')
    # low_data=pd.read_csv('/home/chengkaiyang/MGE4SL/update_data/low_expr_low_geneEffect_symmetric_matrix.csv')
    # par_data=pd.read_csv('/home/chengkaiyang/MGE4SL/update_data/paralogs.csv')
    # fold_change=pd.read_csv('/home/chengkaiyang/MGE4SL/A549_process/haha.csv')
    isme_data = change(redic, isme_data)
    low_data = change(redic, low_data)
    par_data = change(redic, par_data)
    fold_data = change(redic, fold_data)
    # fold_change=fold_change[fold_change['weight']>3]
    # go_F_data=change(redic,fold_change)

    return isme_data, low_data, par_data, fold_data


def processtest(cell, redic, node):
    '''
    description: isme_data represents EM data
    lowexp_data represents ES data
    par_data represents paralog data
    fold_data represents L1000 data
    return {processed multi-omics data}
    '''
    # cell = 'A375'
    if cell == 'A549':

        isme_data = pd.read_table(
            '/home/pumengchen/DataSource/SLdata/temp/temp_mutational_exclu_list/A549_me_list.txt',
            sep='\t',
        )
        isme_data = isme_data[isme_data['is_me_p0.05'] == 1]

        lowexp_data = pd.read_table(
            '/home/pumengchen/DataSource/SLdata/temp/temp_low_expr_low_geneEffect_list/A549_low_expr_low_geneEffect_list.txt',
            sep=',',
        )
        lowexp_data.rename(
            columns={"rowGene": "gene1", "colGene": "gene2"}, errors="raise"
        )
        par_data = pd.read_csv(
            '/home/intern/SyntheticLethal/final_code/Data/A549/p.csv'
        )
        fold_data = pd.read_table(
            '/home/pumengchen/DataSource/SLdata/temp/temp_pair_gene_fold_change_value_list/A549_pair_gene_fold_change_value_list.txt',
            sep=',',
        )
        fold_data = fold_data[fold_data['value'] > 3]
        lowexp_data = lowexp_data.rename(
            columns={"rowGene": "gene1", "colGene": "gene2"}, errors="raise"
        )
        fold_data = fold_data.rename(
            columns={"rowGene": "gene1", "colGene": "gene2"}, errors="raise"
        )
        # isme_data = isme_data[
        #     isme_data['gene1'].isin(redic.keys())
        #     & isme_data['gene2'].isin(redic.keys())
        # ]
        # lowexp_data=lowexp_data[lowexp_data['gene1'].isin(redic.keys()) & lowexp_data['gene2'].isin(redic.keys())]
        # lowexp_data[lowexp_data['gene1'].isin(redic.keys()) & lowexp_data['gene2'].isin(redic.keys())]
        # fold_data[isme_data['gene1'].isin(redic.keys()) & fold_data['gene2'].isin(redic.keys())]
        isme_data = remove_invalid(isme_data, redic, node)
        lowexp_data = remove_invalid(lowexp_data, redic, node)
        par_data = remove_invalid(par_data, redic, node)
        fold_data = remove_invalid(fold_data, redic, node)
        isme_data = change(redic, isme_data)
        lowexp_data = change(redic, lowexp_data)
        par_data = change(redic, par_data)
        fold_data = change(redic, fold_data)
    elif cell == 'A375':
        isme_data = pd.read_table(
            '/home/pumengchen/DataSource/SLdata/temp/temp_mutational_exclu_list/A375_me_list.txt',
            sep='\t',
        )
        isme_data = isme_data[isme_data['is_me_p0.05'] == 1]

        lowexp_data = pd.read_table(
            '/home/pumengchen/DataSource/SLdata/temp/temp_low_expr_low_geneEffect_list/A375_low_expr_low_geneEffect_list.txt',
            sep=',',
        )
        lowexp_data.rename(
            columns={"rowGene": "gene1", "colGene": "gene2"}, errors="raise"
        )
        par_data = pd.read_csv(
            '/home/intern/SyntheticLethal/final_code/Data/A375/p.csv'
        )
        fold_data = pd.read_table(
            '/home/pumengchen/DataSource/SLdata/temp/temp_pair_gene_fold_change_value_list/A375_pair_gene_fold_change_value_list.txt',
            sep=',',
        )
        fold_data = fold_data[fold_data['value'] > 3]
        lowexp_data = lowexp_data.rename(
            columns={"rowGene": "gene1", "colGene": "gene2"}, errors="raise"
        )
        fold_data = fold_data.rename(
            columns={"rowGene": "gene1", "colGene": "gene2"}, errors="raise"
        )
        # isme_data = isme_data[
        #     isme_data['gene1'].isin(redic.keys())
        #     & isme_data['gene2'].isin(redic.keys())
        # ]
        # lowexp_data=lowexp_data[lowexp_data['gene1'].isin(redic.keys()) & lowexp_data['gene2'].isin(redic.keys())]
        # lowexp_data[lowexp_data['gene1'].isin(redic.keys()) & lowexp_data['gene2'].isin(redic.keys())]
        # fold_data[isme_data['gene1'].isin(redic.keys()) & fold_data['gene2'].isin(redic.keys())]
        isme_data = remove_invalid(isme_data, redic, node)
        lowexp_data = remove_invalid(lowexp_data, redic, node)
        par_data = remove_invalid(par_data, redic, node)
        fold_data = remove_invalid(fold_data, redic, node)
        isme_data = change(redic, isme_data)
        lowexp_data = change(redic, lowexp_data)
        par_data = change(redic, par_data)
        fold_data = change(redic, fold_data)
    else:

        isme_data = pd.read_table(
            '/home/pumengchen/DataSource/SLdata/temp/temp_mutational_exclu_list/HT29_me_list.txt',
            sep='\t',
        )
        isme_data = isme_data[isme_data['is_me_p0.05'] == 1]

        lowexp_data = pd.read_table(
            '/home/pumengchen/DataSource/SLdata/temp/temp_low_expr_low_geneEffect_list/HT29_low_expr_low_geneEffect_list.txt',
            sep=',',
        )
        lowexp_data.rename(
            columns={"rowGene": "gene1", "colGene": "gene2"}, errors="raise"
        )
        par_data = pd.read_csv(
            '/home/intern/SyntheticLethal/final_code/Data/HT29/p.csv'
        )
        fold_data = pd.read_table(
            '/home/pumengchen/DataSource/SLdata/temp/temp_pair_gene_fold_change_value_list/HT29_pair_gene_fold_change_value_list.txt',
            sep=',',
        )
        fold_data = fold_data[fold_data['value'] > 3]
        lowexp_data = lowexp_data.rename(
            columns={"rowGene": "gene1", "colGene": "gene2"}, errors="raise"
        )
        fold_data = fold_data.rename(
            columns={"rowGene": "gene1", "colGene": "gene2"}, errors="raise"
        )
        # isme_data = isme_data[
        #     isme_data['gene1'].isin(redic.keys())
        #     & isme_data['gene2'].isin(redic.keys())
        # ]
        # lowexp_data=lowexp_data[lowexp_data['gene1'].isin(redic.keys()) & lowexp_data['gene2'].isin(redic.keys())]
        # lowexp_data[lowexp_data['gene1'].isin(redic.keys()) & lowexp_data['gene2'].isin(redic.keys())]
        # fold_data[isme_data['gene1'].isin(redic.keys()) & fold_data['gene2'].isin(redic.keys())]
        isme_data = remove_invalid(isme_data, redic, node)
        lowexp_data = remove_invalid(lowexp_data, redic, node)
        par_data = remove_invalid(par_data, redic, node)
        fold_data = remove_invalid(fold_data, redic, node)
        isme_data = change(redic, isme_data)
        lowexp_data = change(redic, lowexp_data)
        par_data = change(redic, par_data)
        fold_data = change(redic, fold_data)

    return isme_data, lowexp_data, par_data, fold_data


def construct_omic(isme_data, low_data, par_data, go_F_data, redic):

    synlethdb_isme = SynlethDB_omic(isme_data, 'label', redic)
    synlethdb_isme = train_test_split_edges_omic(synlethdb_isme, test_ratio=0)

    synlethdb_low = SynlethDB_omic(low_data, 'label', redic)
    synlethdb_low = train_test_split_edges_omic(synlethdb_low, test_ratio=0)

    par_data = SynlethDB_omic(par_data, 'label', redic)
    synlethdb_par = train_test_split_edges_omic(par_data, test_ratio=0)
    synlethdb_go_F = SynlethDB_omic(go_F_data, 'value', redic)
    synlethdb_go_F = train_test_split_edges_omic(synlethdb_go_F, test_ratio=0)

    return synlethdb_isme, synlethdb_low, synlethdb_par, synlethdb_go_F


def get_node(df_sl):
    ge1 = []

    jiyina = df_sl.loc[:, 'gene1'].tolist()
    jiyinc = df_sl.loc[:, 'gene2'].tolist()
    labels = df_sl.loc[:, 'label'].tolist()
    for a, b, j in zip(jiyina, jiyinc, labels):
        # if j==1:

        ge1.append(a)
        ge1.append(b)
    ge1 = list(set(ge1))

    return ge1


def start(args, index, data_path, model, epochs, lr=0.01):

    if args.a375test == 1:
        f = open(args.A375_gene, "r")
    if args.HT29test == 1:
        f = open(args.HT29_gene, "r")
    if args.a549test == 1:
        f = open(args.A549_gene, "r")

    lines = f.readlines()  # 读取全部内容 ，并以列表方式返回
    num_nodes = len(lines)
    gene_all = []
    for line in lines:
        gene_all.append(line)
    f.close()
    dit = {}
    redic = {}
    for i, lie in enumerate(gene_all):
        dit[i] = lie[:-1]
        redic[lie[:-1]] = i
    flag = False
    if args.a549test == 1 and args.test_cell == 'A549':
        flag = True
    if args.HT29test == 1 and args.test_cell == 'HT29':
        flag = True
    if args.a375test == 1 and args.test_cell == 'A375':
        flag = True

    # 构建外部测试集
    if args.out_test:

        if args.test_cell == 'A549':
            df = pd.read_csv(args.A549_data_path, delimiter=',')

        elif args.test_cell == 'A375':
            df = pd.read_csv(args.A375_data_path, delimiter=',')
        else:
            df = pd.read_csv(args.HT29_data_path, delimiter=',')

        test_node_dict = get_node(df)
        # df=sklearn.utils.shuffle(df)
        gene_lefts = df.loc[:, 'gene1'].tolist()
        gene_rights = df.loc[:, 'gene2'].tolist()
        indexa = []
        # indexb=[]
        for inde, (a, b) in enumerate(zip(gene_lefts, gene_rights)):
            if a in redic.keys() and b in redic.keys():
                indexa.append(inde)
        df = df.loc[indexa, :]

        testsl_data = df[df['label'] == 1]
        testsl_data['gene1'] = testsl_data['gene1'].apply(lambda x: redic[x])
        testsl_data['gene2'] = testsl_data['gene2'].apply(lambda x: redic[x])

        tnosl_data = df[df['label'] != 1]
        tnosl_data['gene1'] = tnosl_data['gene1'].apply(lambda x: redic[x])
        tnosl_data['gene2'] = tnosl_data['gene2'].apply(lambda x: redic[x])

        tpos_edge_index = torch.tensor(
            testsl_data[['gene1', 'gene2']].T.values, dtype=torch.long
        )

        tneg_edge_index = torch.tensor(
            tnosl_data[['gene1', 'gene2']].T.values, dtype=torch.long
        )
        synlethdb = SynlethDB(num_nodes, testsl_data, tnosl_data)
        genes = np.array([redic[i[:-1]] for i in gene_all])
        if flag == True:
            if args.cross == 'cv1':

                k_fold = get_k_fold_data_random_neg(
                    synlethdb, tpos_edge_index, tneg_edge_index, k=5
                )
            if args.cross == 'cv2':

                k_fold = get_k_fold_data_random_negcv2(
                    genes, synlethdb, tpos_edge_index, tneg_edge_index, k=5
                )
            if args.cross == 'cv3':

                k_fold = get_k_fold_data_random_negcv3(
                    genes, synlethdb, tpos_edge_index, tneg_edge_index, k=5
                )

        testisme_data, testlowexp_data, testpar_data, testfold_data = processtest(
            args.test_cell, redic, test_node_dict
        )

        (
            testsynlethdb_isme,
            testsynlethdb_low,
            testsynlethdb_par,
            testsynlethdb_go_F,
        ) = construct_omic(
            testisme_data, testlowexp_data, testpar_data, testfold_data, redic
        )
    else:
        tpos_edge_index = None
        tneg_edge_index = None
    if flag:
        counts = 0
        for k in k_fold:
            if index == 0:

                val_perf, auc_test, aupr_test, mcc_test, precision, recall = valid(
                    model,
                    num_nodes,
                    k.val_pos_edge_index,
                    k.val_neg_edge_index,
                    testsynlethdb_isme,
                    testsynlethdb_low,
                    testsynlethdb_par,
                    testsynlethdb_go_F,
                )
                break
            else:
                counts += 1
                if counts == index:
                    val_perf, auc_test, aupr_test, mcc_test, precision, recall = valid(
                        model,
                        num_nodes,
                        k.val_pos_edge_index,
                        k.val_neg_edge_index,
                        testsynlethdb_isme,
                        testsynlethdb_low,
                        testsynlethdb_par,
                        testsynlethdb_go_F,
                    )
                    break

    else:

        val_perf, auc_test, aupr_test, mcc_test, precision, recall = valid(
            model,
            num_nodes,
            tpos_edge_index,
            tneg_edge_index,
            testsynlethdb_isme,
            testsynlethdb_low,
            testsynlethdb_par,
            testsynlethdb_go_F,
        )

    return auc_test, aupr_test, mcc_test, precision, recall
