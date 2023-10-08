import torch
import pandas as pd
import numpy as np
import sys, os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


import math
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import pandas as pd
import numpy as np


# related knowledge graph
class SynlethDB_omic(Data):
    def __init__(self, omic_data, types, num_nodes):
        self.type = types

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
    # perm = torch.randperm(row.size(0))
    # row, col = row[perm], col[perm]

    r, c = row[:n_t], col[:n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row[n_t:], col[n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    # perm = torch.randperm(neg_row.size(0))[:num_edges]
    # neg_row, neg_col = neg_row[perm], neg_col[perm]

    row, col = neg_row[:n_t], neg_col[:n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_t:], neg_col[n_t:]
    data.train_neg_edge_index = torch.stack([row, col], dim=0)

    return data


def construct_multi_omic(num_nodes, isme_data, lowexp_data, par_data, fold_data):

    synlethdb_isme = SynlethDB_omic(isme_data, 'label', num_nodes)
    synlethdb_isme = train_test_split_edges_omic(synlethdb_isme, test_ratio=0)

    synlethdb_low = SynlethDB_omic(lowexp_data, 'label', num_nodes)
    synlethdb_low = train_test_split_edges_omic(synlethdb_low, test_ratio=0)

    synlethdb_par = SynlethDB_omic(par_data, 'label', num_nodes)
    synlethdb_par = train_test_split_edges_omic(synlethdb_par, test_ratio=0)
    synlethdb_go_F = SynlethDB_omic(fold_data, 'value', num_nodes)
    synlethdb_go_F = train_test_split_edges_omic(synlethdb_go_F, test_ratio=0)

    return synlethdb_isme, synlethdb_low, synlethdb_par, synlethdb_go_F


def change(redic, data):
    data['gene1'] = data['gene1'].apply(lambda x: redic[x])
    data['gene2'] = data['gene2'].apply(lambda x: redic[x])
    data.insert(1, 'label', np.ones(len(data)))
    return data


def processtest(cell, redic):
    '''
    description: isme_data represents EM data
    lowexp_data represents ES data
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
    # count_frequency(data)
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

    isme_data = remove_invalid(isme_data, redic)
    lowexp_data = remove_invalid(lowexp_data, redic)
    par_data = remove_invalid(par_data, redic)
    fold_data = remove_invalid(fold_data, redic)
    isme_data = change(redic, isme_data)
    lowexp_data = change(redic, lowexp_data)
    par_data = change(redic, par_data)
    fold_data = change(redic, fold_data)

    return isme_data, lowexp_data, par_data, fold_data


def remove_invalid(isme_data, redic):
    # isme_data = isme_data.rename(
    #     columns={"rowGene": "gene1", "colGene": "gene2"}, errors="raise"
    # )
    isme_data = isme_data[
        isme_data['gene1'].isin(redic.keys()) & isme_data['gene2'].isin(redic.keys())
    ]
    # isme_data = isme_data[isme_data['gene1'].isin(node) | isme_data['gene2'].isin(node)]
    return isme_data


def judge(dit, shang, yushu, tmp_input):
    flag = False
    if dit[shang] in tmp_input and dit[yushu] in tmp_input:
        flag = True
    if [yushu, shang] in tmp_input:
        flag = True
    return flag


def judges(shang, yushu, tmp_input):
    flag = False
    if [shang, yushu] in tmp_input:
        flag = True
    if [yushu, shang] in tmp_input:
        flag = True
    return flag


@torch.no_grad()
def tests(
    outgene,
    args,
    model,
    num_nodes,
    tpos_edge_index,
    tneg_edge_index,
    testsynlethdb_isme,
    testsynlethdb_low,
    testsynlethdb_par,
    testsynlethdb_go_F,
    dit,
    tmp_input,
    all_input,
):
    model.eval()

    pos_edge_index_pre = tpos_edge_index[:, : int(1 / 2 * tpos_edge_index.shape[1])]
    pos_edge_index_back = tpos_edge_index[:, int(1 / 2 * tpos_edge_index.shape[1]) :]

    neg_edge_index = tneg_edge_index

    pred_score = get_emb(
        model,
        num_nodes,
        pos_edge_index_pre,
        # pos_edge_index,
        neg_edge_index,
        testsynlethdb_isme,
        testsynlethdb_low,
        testsynlethdb_par,
        testsynlethdb_go_F,
    )

    shangsanjiao = torch.triu(torch.ones(num_nodes, num_nodes), diagonal=1)

    li = [pred_score]
    for pred_score in li:
        pred_score = pred_score * shangsanjiao
        aaas = pred_score.cpu().numpy()
        pred_score = torch.flatten(pred_score).cpu().numpy()
        link_pred = np.argsort(-pred_score)
        lieb = []
        count = 0
        jishh = 0
        for index, i in enumerate(link_pred):
            shang = i // num_nodes
            yushu = i - num_nodes * shang
            # if judges(shang, yushu, all_input):
            #     print(dit[shang], dit[yushu])

            if judge(dit, shang, yushu, outgene):

                # if judges(shang, yushu, all_input):
                #     print(dit[shang], dit[yushu])

                # jishu = [dit[shang], dit[yushu], aaas[shang][yushu]]

                # lieb.append([shang, yushu])

                lists = [dit[shang], dit[yushu], aaas[shang][yushu]]
                data = pd.DataFrame([lists])
                data.to_csv(
                    args.save_paths,
                    mode='a',
                    header=False,
                    index=False,
                )  #
                count += 1
                if args.Topk == count:
                    print(jishh)
                    break
            else:
                jishh += 1

    return pred_score


def get_emb(
    model,
    num_nodes,
    sl_pos,
    sl_neg,
    isme,
    low_exp,
    para,
    emb_fold_change,
    # dit,
    # rank_id,
):

    sl_pos = sl_pos
    node_tensor = torch.ones(num_nodes, 1)
    emb_sl = model.encode_sl(node_tensor, sl_pos)

    emb_isme = model.encode_isme(node_tensor, isme.train_pos_edge_index)
    emb_low = model.encode_low(node_tensor, low_exp.train_pos_edge_index)
    emb_par = model.encode_par(node_tensor, para.train_pos_edge_index)
    emb_fold_change = model.fold_change(
        node_tensor, emb_fold_change.train_pos_edge_index
    )

    emb_all = [emb_sl, emb_isme, emb_low, emb_par, emb_fold_change]
    # emb_all = [emb_sl, emb_isme, emb_fold_change]
    features_all = torch.zeros((node_tensor.shape[0], 1))
    new_feature = []
    for emb in emb_all:

        new_feature.append(emb)
        features_all = torch.cat((features_all, emb), dim=1)

    features_all = torch.sum(features_all, 1)
    features_all = torch.unsqueeze(features_all, 1)
    # _,features_all3=torch.sort(torch.sum(features_all3,1), descending=True)
    # aaabig=torch.matmul(features_key,features_key.transpose(0, 1))
    output = torch.matmul(features_all, features_all.transpose(0, 1))

    return output


def get_index(gene_left, gene_right, gene1, gene2):
    index = []
    for inde, (a, b) in enumerate(zip(gene_left, gene_right)):
        if a == gene1 and b == gene2:
            index.append(inde)
        if a == gene2 and b == gene1:
            index.append(inde)
    return index


def infer(args, gene, data_path, model):

    f = open(gene, "r")
    lines = f.readlines()  # 读取全部内容 ，并以列表方式返回
    num_nodes = len(lines)
    all_gene = []
    for line in lines:
        all_gene.append(line)
    f.close()
    dit = {}
    redic = {}
    # all_genes = [i[:-1] for i in all_genes]
    for i, lie in enumerate(all_gene):
        dit[i] = lie[:-1]
        redic[lie[:-1]] = i

    # 是否进行外部测试，构建外部测试集

    df = pd.read_csv(data_path, delimiter=',')
    if data_path.split('/')[-2] == 'HT29':

        query_gene1 = 'MAP2K1'
        query_gene2 = 'MTOR'

        # node_dict = get_node(df)

    else:

        query_gene1 = 'BRAF'
        query_gene2 = 'MAP2K1'

    gene_left = df.loc[:, 'gene1'].tolist()
    gene_right = df.loc[:, 'gene2'].tolist()
    indexa = []

    for inde, (a, b) in enumerate(zip(gene_left, gene_right)):
        # if inde in id_location:
        #     continue

        # if not flags(a,b,query_gene1,query_gene2):
        if a in redic.keys() and b in redic.keys():
            indexa.append(inde)
    df = df.loc[indexa, :]

    out_gene = pd.read_csv(
        '/home/intern/SyntheticLethal/final_codecp/Data/gene_enzyme_enriched_reannot.txt',
        sep=',',
    ).symbol.tolist()
    testsl_data = df[df['label'] == 1]
    testsl_data['n_A'] = testsl_data['gene1'].apply(lambda x: redic[x])
    testsl_data['n_b'] = testsl_data['gene2'].apply(lambda x: redic[x])
    testnosl_data = df[df['label'] != 1]
    testnosl_data['n_A'] = testnosl_data['gene1'].apply(lambda x: redic[x])
    testnosl_data['n_b'] = testnosl_data['gene2'].apply(lambda x: redic[x])
    tpos_edge_index = torch.tensor(
        testsl_data[['n_A', 'n_b']].T.values, dtype=torch.long
    )
    all_input = [
        [int(i), int(j)] for i, j in zip(tpos_edge_index[0], tpos_edge_index[1])
    ]

    tneg_edge_index = torch.tensor(
        testnosl_data[['n_A', 'n_b']].T.values, dtype=torch.long
    )
    x = tpos_edge_index[:, : int(1 / 2 * tpos_edge_index.shape[1])]
    tmp_input = [[int(i), int(j)] for i, j in zip(x[0], x[1])]

    testisme_data, testlowexp_data, testpar_data, testfold_data = processtest(
        data_path.split('/')[-2], redic
    )

    (
        testsynlethdb_isme,
        testsynlethdb_low,
        testsynlethdb_par,
        testsynlethdb_go_F,
    ) = construct_multi_omic(
        num_nodes, testisme_data, testlowexp_data, testpar_data, testfold_data
    )

    print("data prepare finished!")

    num_nodes = len(all_gene)

    res = tests(
        out_gene,
        args,
        model,
        num_nodes,
        tpos_edge_index,
        tneg_edge_index,
        testsynlethdb_isme,
        testsynlethdb_low,
        testsynlethdb_par,
        testsynlethdb_go_F,
        dit,
        tmp_input,
        all_input,
    )
