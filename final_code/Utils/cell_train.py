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
    get_k_fold_data_random_negcv2,
    train_test_split_edges_omic,
    get_k_fold_data_random_negcv3,
    SynlethDB_omic,
)
from .metrics import get_metric_func, EarlyStopping
from scipy import stats
from scipy.stats import t
import torch.nn.functional as F


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
    if type(y_true) == torch.Tensor:
        y_true = y_true.detach().cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
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
def valid(
    model, synlethdb_sl, synlethdb_isme, synlethdb_low, synlethdb_par, fold_change
):
    model.eval()

    pos_edge_index = synlethdb_sl.val_pos_edge_index
    neg_edge_index = synlethdb_sl.val_neg_edge_index
    pos_edge_index_pre = synlethdb_sl.val_pos_edge_index[
        :, : int(1 / 2 * synlethdb_sl.val_pos_edge_index.shape[1])
    ]
    pos_edge_index_back = synlethdb_sl.val_pos_edge_index[
        :, int(1 / 2 * synlethdb_sl.val_pos_edge_index.shape[1]) :
    ]
    perfs = []
    link_pred = model(
        synlethdb_sl.x,
        # pos_edge_index,
        synlethdb_sl.val_pos_edge_index,
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

    auc_test, aupr_test, f1_test, precision, recall = evaluate(
        link_labels.cpu(), link_pred.cpu()
    )
    perfs.extend([auc_test, aupr_test, f1_test, precision, recall])

    return perfs


@torch.no_grad()
def test(
    model, synlethdb_sl, synlethdb_isme, synlethdb_low, synlethdb_par, synlethdb_go_F
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
        synlethdb_sl.test_pos_edge_index,
        pos_edge_index_pre,
        pos_edge_index_back,
        neg_edge_index,
        synlethdb_isme,
        synlethdb_low,
        synlethdb_par,
        synlethdb_go_F,
        'test',
    )

    link_labels = get_link_labels(pos_edge_index_back, neg_edge_index)

    auc_test, aupr_test, mcc_test, precision, recall = evaluate(link_labels, link_pred)
    perfs.extend([auc_test, aupr_test, mcc_test, precision, recall])

    return perfs


def change(redic, data):
    data['gene1'] = data['gene1'].apply(lambda x: redic[x])
    data['gene2'] = data['gene2'].apply(lambda x: redic[x])
    data.insert(1, 'label', np.ones(len(data)))
    return data


# def remove_invalid(sl, redic):
#     jiyina = sl.loc[:, 'gene1'].tolist()
#     jiyinc = sl.loc[:, 'gene2'].tolist()
#     indexa = []
#     indexb = []
#     for inde, (a, b) in enumerate(zip(jiyina, jiyinc)):
#         if a in redic.keys() and b in redic.keys():
#             indexa.append(inde)
#     sl = sl.loc[indexa, :]
#     return sl


def process_data(cell, redic, node_dict):
    '''
    description: combined_score_data represents EM data
    low_data represents ES data
    par_data represents paralog data
    fold_data represents L1000 data
    return {processed multi-omics data}
    '''

    isme_data = pd.read_table(
        '/home/pumengchen/DataSource/SLdata/temp/temp_mutational_exclu_list/{}_me_list.txt'.format(
            cell
        ),
        sep='\t',
    )
    isme_data = isme_data[isme_data['is_me_p0.05'] == 1]

    lowexp_data = pd.read_table(
        '/home/pumengchen/DataSource/SLdata/temp/temp_low_expr_low_geneEffect_list/{}_low_expr_low_geneEffect_list.txt'.format(
            cell
        ),
        sep=',',
    )
    lowexp_data.rename(columns={"rowGene": "gene1", "colGene": "gene2"}, errors="raise")
    par_data = pd.read_csv(
        '/home/intern/SyntheticLethal/final_code/Data/{}/p.csv'.format(cell)
    )
    fold_data = pd.read_table(
        '/home/pumengchen/DataSource/SLdata/temp/temp_pair_gene_fold_change_value_list/{}_pair_gene_fold_change_value_list.txt'.format(
            cell
        ),
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
  
    isme_data = change(redic, isme_data)
    low_data = change(redic, low_data)
    par_data = change(redic, par_data)
    fold_data = change(redic, fold_data)
    # fold_change=fold_change[fold_change['weight']>3]
    # go_F_data=change(redic,fold_change)

    return isme_data, low_data, par_data, fold_data


# def process_data(sl, nonsl, dic):
#     jiyina = sl.loc[:, 'A'].tolist()
#     jiyinc = sl.loc[:, 'c'].tolist()
#     indexa = []
# indexb=[]
# for inde,(a,b) in enumerate(zip(jiyina,jiyinc)):
#     if a  in redic.keys() and b in redic.keys() :
#         indexa.append(inde)
# df=df.loc[indexa, :]


def construct_omic(redic, isme_data, low_data, par_data, go_F_data):

    synlethdb_isme = SynlethDB_omic(isme_data, 'label', redic)
    synlethdb_isme = train_test_split_edges_omic(synlethdb_isme, test_ratio=0)
   
    synlethdb_low = SynlethDB_omic(low_data, 'label', redic)
    synlethdb_low = train_test_split_edges_omic(synlethdb_low, test_ratio=0)
  
    par_data = SynlethDB_omic(par_data, 'label', redic)
    synlethdb_par = train_test_split_edges_omic(par_data, test_ratio=0)
    synlethdb_go_F = SynlethDB_omic(go_F_data, 'value', redic)
    synlethdb_go_F = train_test_split_edges_omic(synlethdb_go_F, test_ratio=0)

    return synlethdb_isme, synlethdb_low, synlethdb_par, synlethdb_go_F


def remove_invalid(isme_data, redic, node):
    # isme_data = isme_data.rename(
    #     columns={"rowGene": "gene1", "colGene": "gene2"}, errors="raise"
    # )
    isme_data = isme_data[
        isme_data['gene1'].isin(redic.keys()) & isme_data['gene2'].isin(redic.keys())
    ]
    # isme_data = isme_data[isme_data['gene1'].isin(node) | isme_data['gene2'].isin(node)]
    return isme_data


def get_node(df_sl):
    ge1 = []

    jiyina = df_sl.loc[:, 'gene1'].tolist()
    jiyinc = df_sl.loc[:, 'gene2'].tolist()
    labels = df_sl.loc[:, 'label'].tolist()
    for a, b, j in zip(jiyina, jiyinc, labels):

        ge1.append(a)
        ge1.append(b)
    ge1 = list(set(ge1))

    return ge1


def main(args, data_path, model, epochs, lr=0.01):
    if args.a375test == 1:
        cell_train = 'A375'

        data = pd.read_csv(args.A375_data_path)

        node_dict = get_node(data)
        f = open(args.A375_gene, "r")
    elif args.HT29test == 1:
        cell_train = 'HT29'
        data = pd.read_csv(args.HT29_data_path)

        node_dict = get_node(data)
        f = open(args.HT29_gene, "r")
    else:
        cell_train = 'A549'

        data = pd.read_csv(args.A549_data_path)

        node_dict = get_node(data)
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
    train_gene1 = data.loc[:, 'gene1'].tolist()
    train_gene2 = data.loc[:, 'gene2'].tolist()
    indexas = []
    # indexb=[]
    for inde, (a, b) in enumerate(zip(train_gene1, train_gene2)):
        if a in redic.keys() and b in redic.keys():
            indexas.append(inde)
    data = data.loc[indexas, :]
    sl_data = data[data['label'] == 1]
    nosl_data = data[data['label'] != 1]

    if args.out_test:
        if args.test_cell == 'HT29':
            df = pd.read_csv(args.HT29_data_path, delimiter=',')
            test_node_dict = get_node(df)
        elif args.test_cell == 'A549':
            df = pd.read_csv(args.A549_data_path, delimiter=',')
            test_node_dict = get_node(df)
        else:
            df = pd.read_csv(args.A375_data_path, delimiter=',')
            test_node_dict = get_node(df)

        # df=sklearn.utils.shuffle(df)
        gene_left = df.loc[:, 'gene1'].tolist()
        gene_right = df.loc[:, 'gene2'].tolist()
        indexa = []
        # indexb=[]
        for inde, (a, b) in enumerate(zip(gene_left, gene_right)):
            if a in redic.keys() and b in redic.keys():
                indexa.append(inde)
        df = df.loc[indexa, :]
        # 构建测试集合，构建外部测试集
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
        testisme_data, testlowexp_data, testpar_data, testfold_data = process_data(
            args.test_cell, redic, test_node_dict
        )

        (
            testsynlethdb_isme,
            testsynlethdb_low,
            testsynlethdb_par,
            testsynlethdb_go_F,
        ) = construct_omic(
            data, testisme_data, testlowexp_data, testpar_data, testfold_data
        )
    else:
        tpos_edge_index = None
        tneg_edge_index = None

    sl_data['gene1'] = sl_data['gene1'].apply(lambda x: redic[x])
    sl_data['gene2'] = sl_data['gene2'].apply(lambda x: redic[x])
    nosl_data['gene1'] = nosl_data['gene1'].apply(lambda x: redic[x])
    nosl_data['gene2'] = nosl_data['gene2'].apply(lambda x: redic[x])
    synlethdb = SynlethDB(num_nodes, sl_data, nosl_data)
    # genes = np.array(
    #     list(set(sl_data['gene1'].tolist()).union(set(sl_data['gene2'].tolist())))
    # )
    genes = np.array([redic[i[:-1]] for i in gene_all])
    train_edge_index = torch.tensor(
        sl_data[['gene1', 'gene2']].T.values, dtype=torch.long
    )
    train_neg_edge_index = torch.tensor(
        nosl_data[['gene1', 'gene2']].T.values, dtype=torch.long
    )
    # row, col = train_edge_index
    # perm = torch.randperm(train_edge_index.shape[1])
    # row, col = row[perm], col[perm]
    # train_edge_index = torch.stack([row, col], dim=0)
    # k_fold=train_test_split_edges_cv2(synlethdb,tpos_edge_index,tneg_edge_index, test_ratio=0.1)
    # k_fold = get_k_fold_data_random_neg(
    #     synlethdb, train_edge_index, train_neg_edge_index, k=5
    # )
    if args.cross == 'cv1':

        k_fold = get_k_fold_data_random_neg(
            synlethdb, train_edge_index, train_neg_edge_index, k=5
        )
    if args.cross == 'cv2':

        k_fold = get_k_fold_data_random_negcv2(
            genes, synlethdb, train_edge_index, train_neg_edge_index, k=5
        )
    if args.cross == 'cv3':

        k_fold = get_k_fold_data_random_negcv3(
            genes, synlethdb, train_edge_index, train_neg_edge_index, k=5
        )
    # for k_data in k_fold:
    #     print(k_data)
    isme_data, low_data, par_data, fold_data = process_data(
        cell_train, redic, node_dict
    )

    synlethdb_isme, synlethdb_low, synlethdb_par, fold_data = construct_omic(
        redic, isme_data, low_data, par_data, fold_data
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
    df_test = pd.DataFrame(
        columns=['AUC', 'AUPR', 'MCC', 'Precision', 'Recall', 'fold_num']
    )
    # df_test.to_csv(args.save_log)
    # df_testa = pd.DataFrame(
    #     columns=['AUC', 'AUPR', 'MCC', 'Precision', 'Recall', 'fold_num']
    # )
    # df_testa.to_csv(
    #     '/home/intern/SyntheticLethal/final_code/Result/Train_Res/A375/train_crossss.csv'
    # )
    early_stopping = EarlyStopping()

    for k_data in k_fold:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        # # optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
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
            # score_sum = val_perf[3] + val_perf[4]
            # score_sum=100

            early_stopping(train_loss)
            if early_stopping.early_stop:
                break

            if best_score_sum < score_sum:
                jilu = 0
                best_score_sum = score_sum

                # torch.save(model, '%s/%s_%d' % (args.save_path, cell_train, k) + '.pkl')
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
            # lists = val_perf

            # data = pd.DataFrame([lists])
            # data.to_csv(
            #     '/home/intern/SyntheticLethal/final_code/Result/A375.csv',
            #     mode='a',
            #     header=False,
            #     index=False,
            # )  #

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
                testsynlethdb_isme,
                testsynlethdb_low,
                testsynlethdb_par,
                testsynlethdb_go_F,
            )
            k_test_best_auc.append(test_perf[0])
            k_test_best_aupr.append(test_perf[1])
            k_test_best_MCC.append(test_perf[2])
            k_test_best_recall.append(test_perf[3])
            k_test_best_precision.append(test_perf[4])
            lista = [
                test_perf[0],
                test_perf[1],
                test_perf[2],
                test_perf[3],
                test_perf[4],
                k,
            ]
            datass = pd.DataFrame([lista])
            return test_perf[0], test_perf[1], test_perf[2], test_perf[3], test_perf[4]

        # else:
        lista = [val_perf[0], val_perf[1], val_perf[2], val_perf[3], val_perf[4], k]
        datass = pd.DataFrame([lista])

        datass.to_csv(
            args.save_paths,
            mode='a',
            header=False,
            index=True,
        )
