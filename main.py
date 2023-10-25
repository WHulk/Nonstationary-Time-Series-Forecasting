# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 16:25:36 2023

@author: hwang147
"""

import argparse
import os
import torch

from trainable import Transformer_TimeSeries
from utils.tools import string_split

parser = argparse.ArgumentParser(description='CrossFormer')

parser.add_argument('--data', type=str, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default=r'C:\Users\hwang147\Desktop\Crossformer-master\datasets', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')  
parser.add_argument('--data_split', type=list, default=[0.7,0.2,0.1],help='train/val/test split, can be ratio or number')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location to store model checkpoints')

parser.add_argument('--in_len', type=int, default=40, help='input MTS length (T)')
parser.add_argument('--out_len', type=int, default=1, help='output MTS length (\tau)')
# parser.add_argument('--seg_len', type=int, default=6, help='segment length (L_seg)')
# parser.add_argument('--win_size', type=int, default=2, help='window size for segment merge')
# parser.add_argument('--factor', type=int, default=10, help='num of routers in Cross-Dimension Stage of TSA (c)')

# parser.add_argument('--data_dim', type=int, default=7, help='Number of dimensions of the MTS data (D)')
parser.add_argument('--d_input', type=int, default=1, help='input_dimension')
parser.add_argument('--d_output', type=int, default=1, help='output_dimension')
parser.add_argument('--d_model', type=int, default=32, help='dimension of hidden states (d_model)')
# parser.add_argument('--d_ff', type=int, default=512, help='dimension of MLP in transformer')
parser.add_argument('--n_heads', type=int, default=3, help='num of heads')
parser.add_argument('--q_dimension', type=int, default=32, help='q_dimension')
parser.add_argument('--k_dimension', type=int, default=32, help='k_dimension')
parser.add_argument('--v_dimension', type=int, default=32, help='v_dimension')
parser.add_argument('--num_layers', type=int, default=2, help='num of encoder layers (N)')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')

#parser.add_argument('--baseline', action='store_true', help='whether to use mean of past series as baseline for prediction', default=False)

#parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--train_epochs', type=int, default=55, help='train epochs')
parser.add_argument('--patience', type=int, default=7, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=2e-3, help='optimizer initial learning rate')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--itr', type=int, default=1, help='experiments times')

parser.add_argument('--save_pred', action='store_true', help='whether to save the predicted future MTS', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
    print(args.gpu)

    
exp = Transformer_TimeSeries(args)

setting = 'Transformer_{}_il{}_ol{}_sl{}_win{}_fa{}_dm{}_nh{}'.format(args.data, args.in_len, 
                                                                      args.out_len, args.d_input, args.d_output, 
                                                                      args.d_model, args.n_heads, args.num_layers)
exp.train(setting)
exp.eval(setting, save_pred=True,inverse=False)









