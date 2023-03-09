import numpy as np
import torch
import pandas as pd
import argparse
import os


def change_to_adj(df1, sl_ind2, lens):
    raw = np.zeros((lens, lens))
    df1 = pd.read_csv(df1)
    # df1=df1[df1['weight']>3]
    geneleft = df1.loc[:, 'gene1'].tolist()
    generight = df1.loc[:, 'gene2'].tolist()

    for inde, (a, b) in enumerate(zip(geneleft, generight)):
        if a in sl_ind2.keys() and b in sl_ind2.keys():
            raw[sl_ind2[a]][sl_ind2[b]] = 1
            raw[sl_ind2[b]][sl_ind2[a]] = 1

    return raw


def jarrd(raw1, raw2):
    listA = [(i, j) for i, j in zip(np.nonzero(raw1)[0], np.nonzero(raw1)[1])]
    listB = [(i, j) for i, j in zip(np.nonzero(raw2)[0], np.nonzero(raw2)[1])]
    retB = list(set(listA).intersection(set(listB)))

    # union
    retC = list(set(listA).union(set(listB)))
    return len(retB) / len(retC)


def process(sl_data, redic):

    sl_data['gene1'] = sl_data['gene1'].apply(lambda x: redic[x])
    sl_data['gene2'] = sl_data['gene2'].apply(lambda x: redic[x])
    return sl_data


def get_cos_similar(v1: list, v2: list):
    num = float(np.dot(v1, v2))  # dot
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0


def get_res(file):
    f = open(file, "r")
    lines = f.readlines() # return a list
    num_nodes = len(lines)
    gene_list = []
    for line in lines:
        gene_list.append(line)
    f.close()
    dit = {}
    redic = {}
    for i, lie in enumerate(gene_list):
        dit[i] = lie[:-1]
        redic[lie[:-1]] = i
    return redic, dit, gene_list


def EuclideanDistance(x, y):
    d = 0
    for a, b in zip(x, y):  #
        d += (a - b) ** 2
    return d**0.5


def AdjCosine_np(x, y):
    a = np.array(x)
    b = np.array(y)
    avr = (x[0] + y[0]) / 2
    d = np.linalg.norm(a - avr) * np.linalg.norm(b - avr)
    return 0.5 + 0.5 * (np.dot(a - avr, b - avr) / d)


def Manhattan(x, y):
    d = 0
    for a, b in zip(x, y):
        d += a - b
    return abs(d)


def Cosine(x, y):
    sum_xy = 0
    num_x = 0
    num_y = 0
    for a, b in zip(x, y):
        sum_xy += a * b
        num_x += a**2
        num_y += b**2
    if num_x == 0 or num_y == 0:  # denominator ==0 ？
        return None
    else:
        return sum_xy / (num_y * num_x) ** 0.5


def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    sim = num / denom
    return sim


def add_optimizer_args(parser):
    group = parser.add_argument_group("optimizering options")

    group.add_argument("--a549test", help="whether to test", type=int, default=0)
    group.add_argument(
        "--A549_data_path",
        help="a549 train sl path",
        type=str,
        default='/home/intern/SyntheticLethal/new/self_A549/sl.csv',
    )
    group.add_argument("--flag_test", help="flag to test rank id", type=int, default=0)
    group.add_argument(
        "--A549_gene",
        help="A549_gene",
        type=str,
        default='~/SyntheticLethal/data/self_A549/gene.txt',
    )
    group.add_argument(
        "--A375_data_path",
        help="a375 train sl path",
        type=str,
        default='~/SyntheticLethal/data/self_A375/1.csv',
    )
    group.add_argument("--a375test", help="flag to test rank id", type=int, default=0)
    group.add_argument(
        "--A375_gene",
        help="A375_gene",
        type=str,
        default='~/SyntheticLethal/data/A375_process/gene.txt',
    )
    group.add_argument(
        "--HT29_data_path",
        help="HT29 train sl path",
        type=str,
        default='~/SyntheticLethal/data/self_HT29/sl.csv',
    )
    group.add_argument("--HT29test", help="flag to test rank id", type=int, default=1)
    group.add_argument(
        "--HT29_gene",
        help="HT29_gene",
        type=str,
        default='~/SyntheticLethal/data/HT29_process/gene.txt',
    )
    return group


if __name__ == '__main__':

    # EPOCHS = 200
    # data_path = '/home/chengkaiyang/git_code/MGE4SL/data/human_sl_encoder.csv'
    # flag_test=True
    parser = argparse.ArgumentParser("synthetic lethal prediction")
    group = add_optimizer_args(parser)

    # parsing.add_train_args(parser)
    # parser.add_argument_group
    pathA549 = '/home/intern/SyntheticLethal/final_code/Data/A549'
    pathHT29 = '/home/intern/SyntheticLethal/final_code/Data/HT29'
    pathA375 = '/home/intern/SyntheticLethal/final_code/Data/A375'
    all_list = [
        'fold.csv',
        'is_me_or_not_cutoff_0.05_symmetric_matrix.csv',
        'paralogs_symmetric_matrix.csv',
        'low_expr_low_geneEffect_symmetric_matrix.csv',
    ]
    args = parser.parse_args()
    if args.HT29test:
        redic1, dic1, gene_list1 = get_res(args.A549_gene)
        redic2, dic2, gene_list2 = get_res(args.A375_gene)

        common1 = [dic2[i] for i in dic1 if i in dic2.keys()]
        common2 = [dic1[i] for i in dic2 if i in dic1.keys()]
        commongene = list(set(gene_list1).intersection(set(gene_list2)))
        commongene = [i[:-1] for i in commongene]
        local_ind1 = {}  # local cell dict: global cell dict
        sl_ind1 = {}  # global cell dict: local cell dict
        local_ind2 = {}  # local cell dict: global cell dict
        sl_ind2 = {}  # global cell dict: local cell dict
        for i in range(len(commongene)):
            local_ind1[i] = redic1[commongene[i]]
            sl_ind1[dic1[redic1[commongene[i]]]] = i
        for i in range(len(commongene)):
            local_ind2[i] = redic2[commongene[i]]
            sl_ind2[dic2[redic2[commongene[i]]]] = i

        for ida, inde1 in enumerate(all_list):
            raw1 = change_to_adj(
                os.path.join(pathA549, inde1), sl_ind2, len(commongene)
            )

            raw2 = change_to_adj(
                os.path.join(pathA375, inde1), sl_ind2, len(commongene)
            )
            import scipy.stats

            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            from sklearn.metrics.pairwise import pairwise_distances
            from scipy.stats import pearsonr

            vector_a = np.sum(raw1, 0).tolist()
            vector_b = np.sum(raw2, 0).tolist()
            temp = np.sum(raw1, 0) - np.sum(raw2, 0)
            dist = np.sqrt(np.dot(temp.T, temp))
            dist1 = Cosine(vector_a, vector_b)
            dist3 = jarrd(raw1, raw2)
            from scipy.spatial.distance import euclidean
            from scipy.spatial.distance import cosine
            from sklearn.preprocessing import StandardScaler

            print(f'person:' + str(pearsonr(np.sum(raw1, 0), np.sum(raw2, 0))))
            print(f'cos:' + str(dist1))
            print(
                f'EuclideanDistance:' + str(euclidean(np.sum(raw1, 0), np.sum(raw2, 0)))
            )
            print(
                f'AdjCosine_np:' + str(AdjCosine_np(np.sum(raw1, 0), np.sum(raw2, 0)))
            )
            print(f'Manhattan:' + str(Manhattan(np.sum(raw1, 0), np.sum(raw2, 0))))
            print(f'jarrd:' + str(dist3))

    if args.a549test:
        redic1, dic1, liebiao1 = get_res(args.HT29_gene)
        redic2, dic2, liebiao2 = get_res(args.A375_gene)

        common1 = [dic2[i] for i in dic1 if i in dic2.keys()]
        common2 = [dic1[i] for i in dic2 if i in dic1.keys()]
        commongene = list(set(liebiao1).intersection(set(liebiao2)))
        commongene = [i[:-1] for i in commongene]
        local_ind1 = {}  # local cell dict: global cell dict
        sl_ind1 = {}  # global cell dict: local cell dict
        local_ind2 = {}  # local cell dict: global cell dict
        sl_ind2 = {}  # global cell dict: local cell dict
        for i in range(len(commongene)):
            local_ind1[i] = redic1[commongene[i]]
            sl_ind1[dic1[redic1[commongene[i]]]] = i
        for i in range(len(commongene)):
            local_ind2[i] = redic2[commongene[i]]
            sl_ind2[dic2[redic2[commongene[i]]]] = i

        for ida, inde1 in enumerate(all_list):
            raw1 = change_to_adj(
                os.path.join(pathHT29, inde1), sl_ind2, len(commongene)
            )

            raw2 = change_to_adj(
                os.path.join(pathA375, inde1), sl_ind2, len(commongene)
            )
            import scipy.stats

            # print(get_cos_similar(np.sum(raw1,1).tolist(),np.sum(raw2,1).tolist()))
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            from sklearn.metrics.pairwise import pairwise_distances
            from scipy.stats import pearsonr

            vector_a = np.sum(raw1, 0).tolist()
            vector_b = np.sum(raw2, 0).tolist()
            temp = np.sum(raw1, 0) - np.sum(raw2, 0)
            dist = np.sqrt(np.dot(temp.T, temp))
            dist1 = Cosine(vector_a, vector_b)
            dist3 = jarrd(raw1, raw2)
            from scipy.spatial.distance import euclidean
            from scipy.spatial.distance import cosine
            from sklearn.preprocessing import StandardScaler

            print(f'person:' + str(pearsonr(np.sum(raw1, 0), np.sum(raw2, 0))))
            print(f'cos:' + str(dist1))
            print(
                f'EuclideanDistance:' + str(euclidean(np.sum(raw1, 0), np.sum(raw2, 0)))
            )
            print(
                f'AdjCosine_np:' + str(AdjCosine_np(np.sum(raw1, 0), np.sum(raw2, 0)))
            )
            print(f'Manhattan:' + str(Manhattan(np.sum(raw1, 0), np.sum(raw2, 0))))
            print(f'jarrd:' + str(dist3))

    if args.a375test:
        redic1, dic1, liebiao1 = get_res(args.HT29_gene)
        redic2, dic2, liebiao2 = get_res(args.A549_gene)

        common1 = [dic2[i] for i in dic1 if i in dic2.keys()]
        common2 = [dic1[i] for i in dic2 if i in dic1.keys()]
        commongene = list(set(liebiao1).intersection(set(liebiao2)))
        commongene = [i[:-1] for i in commongene]
        local_ind1 = {}  # local cell dict: global cell dict
        sl_ind1 = {}  # global cell dict: local cell dict
        local_ind2 = {}  # local cell dict: global cell dict
        sl_ind2 = {}  # global cell dict: local cell dict
        for i in range(len(commongene)):
            local_ind1[i] = redic1[commongene[i]]
            sl_ind1[dic1[redic1[commongene[i]]]] = i
        for i in range(len(commongene)):
            local_ind2[i] = redic2[commongene[i]]
            sl_ind2[dic2[redic2[commongene[i]]]] = i

        for ida, inde1 in enumerate(all_list):
            raw1 = change_to_adj(
                os.path.join(pathHT29, inde1), sl_ind2, len(commongene)
            )

            raw2 = change_to_adj(
                os.path.join(pathA549, inde1), sl_ind2, len(commongene)
            )
            import scipy.stats

            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            from sklearn.metrics.pairwise import pairwise_distances
            from scipy.stats import pearsonr

            vector_a = np.sum(raw1, 0).tolist()
            vector_b = np.sum(raw2, 0).tolist()
            temp = np.sum(raw1, 0) - np.sum(raw2, 0)
            dist = np.sqrt(np.dot(temp.T, temp))
            dist1 = Cosine(vector_a, vector_b)
            dist3 = jarrd(raw1, raw2)
            from scipy.spatial.distance import euclidean
            from scipy.spatial.distance import cosine
            from sklearn.preprocessing import StandardScaler

            print(f'person:' + str(pearsonr(np.sum(raw1, 0), np.sum(raw2, 0))))
            print(f'cos:' + str(dist1))
            print(
                f'EuclideanDistance:' + str(euclidean(np.sum(raw1, 0), np.sum(raw2, 0)))
            )
            print(
                f'AdjCosine_np:' + str(AdjCosine_np(np.sum(raw1, 0), np.sum(raw2, 0)))
            )
            print(f'Manhattan:' + str(Manhattan(np.sum(raw1, 0), np.sum(raw2, 0))))
            print(f'jarrd:' + str(dist3))
