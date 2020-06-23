import argparse
import os

import torch
from torch.utils.data import Subset

from data import load_dataset
from utils import str2bool
from disentanglement_class import Disentangler
# from disentanglement import Disentangler
from utils import build_inout_feature_sets
import utils
import numpy as np
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CIFAR-10', choices=['CIFAR-10'])
parser.add_argument('--setsize', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--base_path', type=str, default='/mnt/disk1/heonseok/MPMLD')
parser.add_argument('--resume', type=str2bool, default='0')
parser.add_argument('--train_batch_size', type=int, default=100)
parser.add_argument('--valid_batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=100)

parser.add_argument('--disentanglement_model', type=str, default='AE', choices=['AE', 'VAE'])
parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--early_stop', type=str2bool, default='1')
parser.add_argument('--early_stop_observation_period', type=int, default=20)
parser.add_argument('--repeat_idx', type=int, default=0)
parser.add_argument('--gpu_id', type=int, default=0)

parser.add_argument('--z_dim', type=int, default=16)
parser.add_argument('--disentanglement_type', type=str, default='base', choices=['base', 'type1', 'type2'])
parser.add_argument('--reconstruction_type', type=str, default='partial_z', choices=['full_z', 'partial_z'])

parser.add_argument('--train_disentangler', type=str2bool, default='0')
parser.add_argument('--reconstruct_datasets', type=str2bool, default='0')

args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

# -- Directory -- #
args.output_path = os.path.join(args.base_path, 'output', args.dataset)
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

args.data_path = os.path.join(args.base_path, 'data', args.dataset)
if not os.path.exists(args.data_path):
    os.makedirs(args.data_path)

args.disentanglement_name = os.path.join('disentangler_z{}'.format(args.z_dim), args.disentanglement_type)
args.disentanglement_path = os.path.join(args.output_path, args.disentanglement_name,
                                         'repeat{}'.format(args.repeat_idx))
# args.reconstruction_name = os.path.join(args.disentanglement_name, args.reconstruction_type)
args.reconstruction_path = os.path.join(args.disentanglement_path,
                                        'recon_{}.pt'.format(args.reconstruction_type))

train_set, test_set = load_dataset(args.dataset, args.data_path)

subset0 = Subset(train_set, range(0, args.setsize))
subset1 = Subset(train_set, range(args.setsize, int(1.2 * args.setsize)))
subset2 = Subset(train_set, range(int(1.2 * args.setsize), int(1.4 * args.setsize)))
subset3 = Subset(test_set, range(0, args.setsize))

class_datasets = {
    'train': subset0,
    'valid': subset1,
    'test': subset2,
}
# inout_datasets should be transformed to inout_feature_sets for training attacker
inout_datasets = {
    'in': subset0,
    'out': subset3,
}

for dataset_type, dataset in class_datasets.items():
    print('Class {:<5} dataset: {}'.format(dataset_type, len(dataset)))
print()
for dataset_type, dataset in inout_datasets.items():
    print('Inout {:<3} dataset: {}'.format(dataset_type, len(dataset)))
print()

if args.train_disentangler or args.reconstruct_datasets:
    if args.use_reconstructed_datasets:
        print('You use disentangler with reconstructed datasets; set use_reconstructed_datasets as 0')
        sys.exit(1)
    else:
        disentangler = Disentangler(args)
        if args.train_disentangler:
            disentangler.train(class_datasets['train'])
        if args.reconstruct_datasets:
            disentangler.reconstruct(class_datasets)
