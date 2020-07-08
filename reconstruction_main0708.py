import argparse
import os

import torch
from torch.utils.data import Subset

from data import load_dataset
from utils import str2bool
from reconstruction_AE0708 import ReconstructorAE
from reconstruction_VAE import ReconstructorVAE
from torch.utils.data import ConcatDataset
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='location',
                    choices=['MNIST', 'Fashion-MNIST', 'CIFAR-10', 'adult', 'location'])
parser.add_argument('--setsize', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--base_path', type=str, default='/mnt/disk1/heonseok/MPMLD')
parser.add_argument('--resume', type=str2bool, default='0')
parser.add_argument('--train_batch_size', type=int, default=100)
parser.add_argument('--valid_batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=100)

parser.add_argument('--reconstruction_model', type=str, default='AE', choices=['AE', 'VAE'])
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--early_stop', type=str2bool, default='1')
parser.add_argument('--early_stop_observation_period', type=int, default=20)
parser.add_argument('--repeat_idx', type=int, default=0)
parser.add_argument('--gpu_id', type=int, default=3)

parser.add_argument('--z_dim', type=int, default=64)
parser.add_argument('--disentanglement_type', type=str, default='type2', choices=['base', 'type1', 'type2', 'type3'])

parser.add_argument('--train_reconstructor', type=str2bool, default='1')
parser.add_argument('--reconstruct_datasets', type=str2bool, default='1')

parser.add_argument('--ref_ratio', type=float, default=0.1)

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

merged_dataset = load_dataset(args.dataset, args.data_path)
print(merged_dataset.__len__())

if args.dataset in ['adult', 'location']:
    args.encoder_input_dim = merged_dataset.__getitem__(0)[0].numpy().shape[0]
    if args.dataset == 'adult':
        args.class_num = 2
    elif args.dataset == 'location':
        args.class_num = 30

if args.setsize * 2.4 > len(merged_dataset):
    print('Setsize * 2.4 > len(concatset); Terminate program')
    sys.exit(1)

subset0 = Subset(merged_dataset, range(0, args.setsize))
subset1 = Subset(merged_dataset, range(args.setsize, int(1.2 * args.setsize)))
subset2 = Subset(merged_dataset, range(int(1.2 * args.setsize), int(1.4 * args.setsize)))
subset3 = Subset(merged_dataset, range(int(1.4 * args.setsize), int(2.4 * args.setsize)))
subset4 = Subset(merged_dataset, range(int(2.4 * args.setsize), int((2.4 + args.ref_ratio) * args.setsize)))

class_datasets = {
    'train': subset0,
    'valid': subset1,
    'test': subset2,
}

for dataset_type, dataset in class_datasets.items():
    print('Class {:<5} dataset: {}'.format(dataset_type, len(dataset)))
print()

ref_dataset = subset4
for _ in range(int(1 / args.ref_ratio) - 1):
    ref_dataset = ConcatDataset((ref_dataset, subset4))

if args.reconstruction_model == 'AE':
    reconstructor = ReconstructorAE(args)
elif args.reconstruction_model == 'VAE':
    reconstructor = ReconstructorVAE(args)

if args.train_reconstructor:
    # if args.disentanglement_type in ['base', 'type1', 'type2']:
    reconstructor.train(class_datasets['train'], class_datasets['valid'], ref_dataset)
    # elif args.disentanglement_type in ['type3']:
    #     reconstructor.train(class_datasets['train'], class_datasets['valid'], ref_dataset)

if args.reconstruct_datasets:
    reconstructor.reconstruct(class_datasets, 'full_z')
    reconstructor.reconstruct(class_datasets, 'content_z')
    reconstructor.reconstruct(class_datasets, 'style_z')
