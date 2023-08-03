import argparse
import torch
import os
import sys, os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

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
from final_code.Model.module import MFFSL


def add_optimizer_args(parser):
    group = parser.add_argument_group("synthetic lethal prediction")

    group.add_argument("--lr", help="learning rate", type=float, default=0.002)
    group.add_argument("--a549test", help="whether to test", type=int, default=1)
    group.add_argument(
        "--A549_data_path",
        help="a549 train sl path",
        type=str,
        default='/home/testenv/SyntheticLethal/new/self_A549/sl.csv',
    )
    group.add_argument(
        "--save_path",
        help="a549 model save path",
        type=str,
        default='/home/testenv/SyntheticLethal/final_code/Result',
    )
    group.add_argument("--flag_test", help="flag to test rank id", type=int, default=0)
    group.add_argument(
        "--A549_gene",
        help="A549_gene",
        type=str,
        default='/home/testenv/SyntheticLethal/final_code/Data/A549/gene.txt',
    )
    group.add_argument(
        "--A375_data_path",
        help="a375 train sl path",
        type=str,
        default='/home/testenv/SyntheticLethal/final_code/Data/A375/sl.csv',
    )
    group.add_argument("--a375test", help="flag to test rank id", type=int, default=1)
    group.add_argument(
        "--A375_gene",
        help="A375_gene",
        type=str,
        default='/home/testenv/SyntheticLethal/final_code/Data/A375/gene.txt',
    )
    group.add_argument(
        "--HT29_data_path",
        help="HT29 train sl path",
        type=str,
        default='/home/testenv/SyntheticLethal/final_code/Data/HT29/sl.csv',
    )
    group.add_argument("--HT29test", help="flag to test rank id", type=int, default=0)
    group.add_argument(
        "--HT29_gene",
        help="HT29_gene",
        type=str,
        default='/home/testenv/SyntheticLethal/final_code/Data/HT29/gene.txt',
    )
    group.add_argument("--max_epochs", help="max_epoch", type=int, default=200)
    group.add_argument(
        "--out_test", help="test on the cross cell line", type=int, default=1
    )
    return group


if __name__ == '__main__':

    parser = argparse.ArgumentParser("synthetic lethal prediction")
    group = add_optimizer_args(parser)

    # parsing.add_train_args(parser)
    # parser.add_argument_group

    args = parser.parse_args()

    if args.a549test:

        # sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from final_code.Utils.A549_utils import main

        model = MFFSL(n_graph=4, node_emb_dim=16, sl_input_dim=1)
        main(args, args.A549_data_path, model, epochs=args.max_epochs, lr=args.lr)

    if args.a375test:

        from final_code.Utils.A375_utils import main

        model = MFFSL(n_graph=4, node_emb_dim=16, sl_input_dim=1)
        main(args, args.A375_data_path, model, epochs=args.max_epochs, lr=args.lr)

    if args.HT29test:

        from final_code.Utils.HT29_utils import main

        model = MFFSL(n_graph=4, node_emb_dim=16, sl_input_dim=1)
        main(args, args.HT29_data_path, model, epochs=args.max_epochs, lr=args.lr)
