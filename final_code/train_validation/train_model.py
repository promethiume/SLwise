'''
Descripttion: 
version: 
Author: 成凯阳
Date: 2022-10-25 00:57:50
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-07-18 16:08:23
'''
import argparse
import torch
import os
import sys, os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import pandas as pd
import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, TransformerConv
import copy
import collections
from final_code.Model.modules import MFFSL
from final_code.Utils.finetine import start
from final_code.Utils.cell_train import main
from final_code.Utils.inference import infer
import random
import os
from typing import Optional

from torch import Tensor

from torch_geometric.nn import global_max_pool, global_add_pool


def add_optimizer_args(parser):
    group = parser.add_argument_group("synthetic lethal prediction")

    group.add_argument("--lr", help="learning rate", type=float, default=0.002)
    group.add_argument("--a549test", help="whether to test", type=int, default=0)
    group.add_argument(
        "--A549_data_path",
        help="a549 train sl path",
        type=str,
        default='/home/intern/SyntheticLethal/final_code/Data/A549/sl.csv',
    )

    group.add_argument("--flag_test", help="flag to test rank id", type=int, default=0)
    group.add_argument(
        "--A549_gene",
        help="A549_gene",
        type=str,
        default='/home/intern/SyntheticLethal/final_code/Data/A549/gene.txt',
    )
    group.add_argument(
        "--A375_data_path",
        help="a375 train sl path",
        type=str,
        default='/home/intern/SyntheticLethal/final_code/Data/A375/sl.csv',
    )
    group.add_argument("--a375test", help="flag to test rank id", type=int, default=0)
    group.add_argument(
        "--A375_gene",
        help="A375_gene",
        type=str,
        default='/home/intern/SyntheticLethal/final_code/Data/A375/gene.txt',
    )
    group.add_argument(
        "--HT29_data_path",
        help="HT29 train sl path",
        type=str,
        default='/home/intern/SyntheticLethal/final_code/Data/HT29/sl.csv',
    )
    group.add_argument("--HT29test", help="flag to test rank id", type=int, default=0)
    group.add_argument(
        "--HT29_gene",
        help="HT29_gene",
        type=str,
        default='/home/intern/SyntheticLethal/final_code/Data/HT29/gene.txt',
    )
    group.add_argument("--model", help="encoder", type=str, default='GraphSage')
    group.add_argument("--cross", help="cross validation", type=str, default='cv1')
    # group.add_argument("--save", help="save path", type=str, )
    group.add_argument(
        "--save_paths",
        help="log save_paths",
        type=str,
        default='/home/intern/SyntheticLethal/SL_project/final_code/train_validation/log.csv',
    )
    group.add_argument("--test_cell", help="testcell", type=str, default='A549')

    group.add_argument("--data_aba", help="data_aba", type=int, default=0)
    group.add_argument("--delete_EM", help="data_aba", type=int, default=0)
    group.add_argument("--delete_Es", help="data_aba", type=int, default=0)
    group.add_argument("--delete_par", help="data_aba", type=int, default=0)
    group.add_argument("--delete_L1000", help="data_aba", type=int, default=0)
    group.add_argument("--max_epochs", help="max_epoch", type=int, default=100)
    group.add_argument(
        "--out_test", help="test on the cross cell line", type=int, default=0
    )
    group.add_argument(
        "--inference", help="inference on cell line", type=int, default=0
    )
    group.add_argument("--finetine", help="fintine on cell line", type=int, default=0)
    group.add_argument(
        "--pre",
        help="checkpoint",
        type=str,
        default='/home/intern/SyntheticLethal/SL_project/final_code/Result/checkpoint/',
    )
    group.add_argument(
        "--Topk",
        help="Topk result",  # 排名前k个基因对
        type=int,
        default='1',
    )
    return group


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    parser = argparse.ArgumentParser("synthetic lethal prediction")
    group = add_optimizer_args(parser)

    args = parser.parse_args()

    if args.a549test:
        print('start A549')
        if args.inference:
            setup_seed(20)
            pkl = os.path.join(args.pre, 'A549cv1_0.pkl')
            model = torch.load(
                pkl,
                map_location=torch.device('cpu'),
            )
            gene = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'Data/A549/gene.txt',
            )
            if args.test_cell == 'A549':
                datapath = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'Data/A549/sl.csv',
                )
            if args.test_cell == 'HT29':
                datapath = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'Data/HT29/sl.csv',
                )
            if args.test_cell == 'A375':
                datapath = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'Data/A375/sl.csv',
                )
            infer(args, gene, datapath, model)

        if args.finetine:

            for index in range(5):
                model = MFFSL(
                    ablation=args.data_aba,
                    model_type=args.model,
                    args=args,
                    n_graph=4,
                    node_emb_dim=16,
                    sl_input_dim=1,
                )

                pkl = os.path.join(args.pre, 'A549{}_{}.pkl').format(args.cross, index)
                model = torch.load(
                    pkl,
                    map_location=torch.device('cpu'),
                )
                auc_test, aupr_test, mcc_test, precision, recall = start(
                    args,
                    index,
                    args.A549_data_path,
                    model,
                    epochs=args.max_epochs,
                    lr=args.lr,
                )
                lista = [auc_test, aupr_test, mcc_test, precision, recall, index]
                datass = pd.DataFrame([lista])

                datass.to_csv(
                    args.save_paths,
                    mode='a',
                    header=False,
                    index=True,
                )

        elif args.inference == 0:

            model = MFFSL(
                ablation=args.data_aba,
                model_type=args.model,
                args=args,
                n_graph=4,
                node_emb_dim=16,
                sl_input_dim=1,
            )
            main(args, args.A549_data_path, model, epochs=args.max_epochs, lr=args.lr)

    if args.a375test:
        if args.inference:
            setup_seed(20)

            pkl = os.path.join(args.pre, 'A375cv1_0.pkl')
            model = torch.load(
                pkl,
                map_location=torch.device('cpu'),
            )
            gene = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'Data/A375/gene.txt',
            )
            if args.test_cell == 'A549':
                datapath = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'Data/A549/sl.csv',
                )
            if args.test_cell == 'HT29':
                datapath = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'Data/HT29/sl.csv',
                )
            if args.test_cell == 'A375':
                datapath = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'Data/A375/sl.csv',
                )
            infer(args, gene, datapath, model)

        if args.finetine:
            # from final_code.Utils.A375_fin import main
            # from final_code.Model.modules import MFFSL

            for index in range(5):
                model = MFFSL(
                    ablation=args.data_aba,
                    model_type=args.model,
                    args=args,
                    n_graph=4,
                    node_emb_dim=16,
                    sl_input_dim=1,
                )

                pkl = os.path.join(args.pre, 'A375{}_{}.pkl').format(args.cross, index)

                model = torch.load(
                    pkl,
                    map_location=torch.device('cpu'),
                )
                auc_test, aupr_test, mcc_test, precision, recall = start(
                    args,
                    index,
                    args.A375_data_path,
                    model,
                    epochs=args.max_epochs,
                    lr=args.lr,
                )
                lista = [auc_test, aupr_test, mcc_test, precision, recall, index]
                datass = pd.DataFrame([lista])

                datass.to_csv(
                    args.save_paths,
                    mode='a',
                    header=False,
                    index=True,
                )

        elif args.inference == 0:

            model = MFFSL(
                ablation=args.data_aba,
                model_type=args.model,
                args=args,
                n_graph=4,
                node_emb_dim=16,
                sl_input_dim=1,
            )
            main(args, args.A375_data_path, model, epochs=args.max_epochs, lr=args.lr)

    if args.HT29test:
        if args.inference:
            setup_seed(20)

            pkl = os.path.join(args.pre, 'HT29cv1_0.pkl')
            model = torch.load(
                pkl,
                map_location=torch.device('cpu'),
            )
            gene = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'Data/HT29/gene.txt',
            )
            if args.test_cell == 'A549':
                datapath = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'Data/A549/sl.csv',
                )
            if args.test_cell == 'HT29':
                datapath = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'Data/HT29/sl.csv',
                )
            if args.test_cell == 'A375':
                datapath = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'Data/A375/sl.csv',
                )
            infer(args, gene, datapath, model)
        if args.finetine:

            # sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

            for index in range(5):
                model = MFFSL(
                    ablation=args.data_aba,
                    model_type=args.model,
                    args=args,
                    n_graph=4,
                    node_emb_dim=16,
                    sl_input_dim=1,
                )

                # pre = '/home/intern/SyntheticLethal/final_code/Res/checkpoint'
                pkl = os.path.join(args.pre, 'HT29{}_{}.pkl').format(args.cross, index)

                model = torch.load(
                    pkl,
                    map_location=torch.device('cpu'),
                )
                auc_test, aupr_test, mcc_test, precision, recall = start(
                    args,
                    index,
                    args.HT29_data_path,
                    model,
                    epochs=args.max_epochs,
                    lr=args.lr,
                )
                lista = [auc_test, aupr_test, mcc_test, precision, recall, index]
                datass = pd.DataFrame([lista])

                datass.to_csv(
                    args.save_paths,
                    mode='a',
                    header=False,
                    index=True,
                )

            # pre = os.path.dirname((os.path.dirname(os.path.abspath(__file__))))
            # pkl_path = os.path.join(pre, 'check_point/{}.pkl').format(args.cell)

        elif args.inference == 0:

            model = MFFSL(
                ablation=args.data_aba,
                model_type=args.model,
                args=args,
                n_graph=4,
                node_emb_dim=16,
                sl_input_dim=1,
            )
            main(args, args.HT29_data_path, model, epochs=args.max_epochs, lr=args.lr)

    else:
        pass
