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
    train_test_split_edges_omic,
    SynlethDB_omic,
)
from .metrics import get_metric_func
from scipy import stats
from scipy.stats import t
import torch.nn.functional as F

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
    matthews_corrcoef = get_metric_func(metrics_func[2])(y_true, y_score)
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
    neg_edge_index = synlethdb_sl.train_neg_edge_index

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
    loss = F.binary_cross_entropy(link_pred, link_labels)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def valid(
    model, synlethdb_sl, synlethdb_isme, synlethdb_low, synlethdb_par, fold_change
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
        fold_change,
    )

    link_labels = get_link_labels(pos_edge_index, neg_edge_index)

    auc_test, aupr_test, mcc_test, precision, recall = evaluate(
        link_labels.cpu(), link_pred.cpu()
    )
    perfs.extend([auc_test, aupr_test, mcc_test, precision, recall])

    return perfs


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


def remove_invalid(sl, redic):
    jiyina = sl.loc[:, 'gene1'].tolist()
    jiyinc = sl.loc[:, 'gene2'].tolist()
    indexa = []
    indexb = []
    for inde, (a, b) in enumerate(zip(jiyina, jiyinc)):
        if a in redic.keys() and b in redic.keys():
            indexa.append(inde)
    sl = sl.loc[indexa, :]
    return sl


def pro(redic):
    '''
    description: isme_data represents EM data
    lowexp_data represents ES data
    par_data represents paralog data
    fold_data represents L1000 data
    return {processed multi-omics data}
    '''
    isme_data = pd.read_csv('/home/testenv/SyntheticLethal/mymodel/HT29_process/is.csv')
    lowexp_data = pd.read_csv(
        '/home/testenv/SyntheticLethal/mymodel/HT29_process/low.csv'
    )
    par_data = pd.read_csv('/home/testenv/SyntheticLethal/mymodel/HT29_process/p.csv')
    fold_data = pd.read_csv(
        '/home/testenv/SyntheticLethal/mymodel/HT29_process/folds.csv'
    )

    isme_data = remove_invalid(isme_data, redic)
    lowexp_data = remove_invalid(lowexp_data, redic)
    par_data = remove_invalid(par_data, redic)
    fold_data = remove_invalid(fold_data, redic)

    isme_data = change(redic, isme_data)
    lowexp_data = change(redic, lowexp_data)
    par_data = change(redic, par_data)
    fold_data = change(redic, fold_data)
    # fold_change=fold_change[fold_change['weight']>3]
    # go_F_data=change(redic,fold_change)

    return isme_data, lowexp_data, par_data, fold_data


def construct_omic(data, isme_data, low_data, par_data, go_F_data):

    synlethdb_isme = SynlethDB_omic(isme_data, 'label')
    synlethdb_isme = train_test_split_edges_omic(synlethdb_isme, test_ratio=0)

    synlethdb_low = SynlethDB_omic(low_data, 'label')
    synlethdb_low = train_test_split_edges_omic(synlethdb_low, test_ratio=0)

    par_data = SynlethDB_omic(par_data, 'label')
    synlethdb_par = train_test_split_edges_omic(par_data, test_ratio=0)
    synlethdb_go_F = SynlethDB_omic(go_F_data, 'weight')
    synlethdb_go_F = train_test_split_edges_omic(synlethdb_go_F, test_ratio=0)

    return synlethdb_isme, synlethdb_low, synlethdb_par, synlethdb_go_F


def main(args, data_path, model, epochs, lr=0.01):

    # data = pd.read_csv("/home/testenv/MVGCNiSL-main/code/sl.csv")
    data = pd.read_csv(args.HT29_data_path)

    pos_set_IDa = []
    pos_set_IDb = []
    neg_set_IDb = []
    neg_set_IDa = []
    jiyin = []
    id_jiyin = {}
    f = open(args.HT29_gene, "r")
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
    gene_left = data.loc[:, 'gene1'].tolist()
    gene_right = data.loc[:, 'gene2'].tolist()
    indexas = []
    # indexb=[]
    for inde, (a, b) in enumerate(zip(gene_left, gene_right)):
        if a in redic.keys() and b in redic.keys():
            indexas.append(inde)
    data = data.loc[indexas, :]
    sl_data = data[data['label'] == 1]
    nosl_data = data[data['label'] != 1]
    # 构建外部测试集
    if args.out_test:
        df = pd.read_csv(args.A375_data_path, sep=',')
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
        testsl_data['n_A'] = testsl_data['gene1'].apply(lambda x: redic[x])
        testsl_data['n_b'] = testsl_data['gene2'].apply(lambda x: redic[x])
        tnosl_data = df[df['label'] != 1]
        tnosl_data['n_A'] = tnosl_data['gene1'].apply(lambda x: redic[x])
        tnosl_data['n_b'] = tnosl_data['gene2'].apply(lambda x: redic[x])
        tpos_edge_index = torch.tensor(
            testsl_data[['n_A', 'n_b']].T.values, dtype=torch.long
        )
        tneg_edge_index = torch.tensor(
            tnosl_data[['n_A', 'n_b']].T.values, dtype=torch.long
        )
    else:
        tpos_edge_index = None
        tneg_edge_index = None

    sl_data['gene1'] = sl_data['gene1'].apply(lambda x: redic[x])
    sl_data['gene2'] = sl_data['gene2'].apply(lambda x: redic[x])
    nosl_data['gene1'] = nosl_data['gene1'].apply(lambda x: redic[x])
    nosl_data['gene2'] = nosl_data['gene2'].apply(lambda x: redic[x])
    synlethdb = SynlethDB(num_nodes, sl_data, nosl_data)
    # k_fold=train_test_split_edges_cv2(synlethdb,tpos_edge_index,tneg_edge_index, test_ratio=0.1)
    k_fold = get_k_fold_data_random_neg(
        synlethdb, tpos_edge_index, tneg_edge_index, k=5
    )
    # for k_data in k_fold:
    #     print(k_data)
    isme_data, lowexp_data, par_data, fold_data = pro(redic)

    synlethdb_isme, synlethdb_low, synlethdb_par, fold_data = construct_omic(
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
    for k_data in k_fold:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        # optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
        explr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

        print('start one fold:')
        best_score_sum = 0
        best_loss = torch.tensor(10000)
        for epoch in range(0, epochs):
            train_loss = train(
                model,
                optimizer,
                k_data,
                synlethdb_isme,
                synlethdb_low,
                synlethdb_par,
                fold_data,
            )
            val_perf = valid(
                model, k_data, synlethdb_isme, synlethdb_low, synlethdb_par, fold_data
            )

            explr_scheduler.step()

            score_sum = np.array(val_perf).sum()
            # score_sum=100
            model_name = 'HT29'
            counst = 0
            if train_loss < best_loss:
                best_loss = train_loss
                counst = 0
            else:
                counst += 1
                if counst == 30:

                    break
            if best_score_sum < score_sum:
                jilu = 0
                best_score_sum = score_sum
                torch.save(model, '%s/%s_%d' % (args.save_path, model_name, k) + '.pkl')
                # torch.save(model, '/home/testenv/git_code/xiugai/new/transfercp_sigedouyong/A375/%s_%d'%(model_name, k) + '.pkl')
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
