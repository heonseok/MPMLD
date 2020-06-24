import argparse
import os

import torch
from torch.utils.data import Subset

from data import load_dataset
from utils import str2bool
from reconstruction_AE import ReconstructorAE
from reconstruction_VAE import ReconstructorVAE
from torch.utils.data import ConcatDataset
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

parser.add_argument('--reconstruction_model', type=str, default='AE', choices=['AE', 'VAE'])
parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--early_stop', type=str2bool, default='1')
parser.add_argument('--early_stop_observation_period', type=int, default=20)
parser.add_argument('--repeat_idx', type=int, default=0)
parser.add_argument('--gpu_id', type=int, default=0)

parser.add_argument('--z_dim', type=int, default=16)
parser.add_argument('--disentanglement_type', type=str, default='base', choices=['base', 'type1', 'type2'])

parser.add_argument('--train_reconstructor', type=str2bool, default='0')
parser.add_argument('--reconstruct_datasets', type=str2bool, default='1')

args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

# -- Directory -- #
args.output_path = os.path.join(args.base_path, 'output', args.dataset)
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

args.data_path = os.path.join(args.base_path, 'data', args.dataset)
if not os.path.exists(args.data_path):
    os.makedirs(args.data_path)

args.reconstruction_name = os.path.join(
    '{}_z{}_{}_setsize{}'.format(args.reconstruction_model, args.z_dim, args.disentanglement_type, args.setsize))
args.reconstruction_path = os.path.join(args.output_path, 'reconstructor', args.reconstruction_name,
                                         'repeat{}'.format(args.repeat_idx))

train_set, test_set = load_dataset(args.dataset, args.data_path)
concat_set = ConcatDataset((train_set, test_set))

if args.setsize * 2.4 > len(concat_set):
    print('Setsize * 2.4 > len(concatset); Terminate program')
    sys.exit(1)

subset0 = Subset(concat_set, range(0, args.setsize))
subset1 = Subset(concat_set, range(args.setsize, int(1.2 * args.setsize)))
subset2 = Subset(concat_set, range(int(1.2 * args.setsize), int(1.4 * args.setsize)))
subset3 = Subset(concat_set, range(int(1.4 * args.setsize), int(2.4 * args.setsize)))

class_datasets = {
    'train': subset0,
    'valid': subset1,
    'test': subset2,
}

for dataset_type, dataset in class_datasets.items():
    print('Class {:<5} dataset: {}'.format(dataset_type, len(dataset)))
print()

if args.reconstruction_model == 'AE':
    reconstructor = ReconstructorAE(args)
elif args.reconstruction_model == 'VAE':
    reconstructor = ReconstructorVAE(args)

if args.train_reconstructor:
    reconstructor.train(class_datasets['train'], class_datasets['valid'])

if args.reconstruct_datasets:
    reconstructor.reconstruct(class_datasets, 'full_z')
    reconstructor.reconstruct(class_datasets, 'partial_z')
