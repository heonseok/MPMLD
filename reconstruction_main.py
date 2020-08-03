import argparse
import os

import torch
from torch.utils.data import Subset

from data import load_dataset
from utils import str2bool
from reconstruction import Reconstructor
from torch.utils.data import ConcatDataset
import datetime
import shutil
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CIFAR-10',
                    choices=['MNIST', 'Fashion-MNIST', 'CIFAR-10', 'adult', 'location', 'SVHN'])
parser.add_argument('--output_dir', type=str, default='output0803')
parser.add_argument('--setsize', type=int, default=10000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--base_path', type=str, default='/mnt/disk1/heonseok/MPMLD')
parser.add_argument('--resume', type=str2bool, default='0')
parser.add_argument('--train_batch_size', type=int, default=100)
parser.add_argument('--valid_batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=100)

parser.add_argument('--reconstruction_model', type=str, default='VAE', choices=['AE', 'VAE'])
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--early_stop', type=str2bool, default='1')
parser.add_argument('--early_stop_observation_period', type=int, default=20)
parser.add_argument('--repeat_idx', type=int, default=0)
parser.add_argument('--gpu_id', type=int, default=3)

parser.add_argument('--beta', type=float, default=0.000001)
parser.add_argument('--z_dim', type=int, default=64)

parser.add_argument('--recon_weight', type=float, default='1')
parser.add_argument('--class_cz_weight', type=float, default='0')
parser.add_argument('--class_mz_weight', type=float, default='1')
parser.add_argument('--membership_cz_weight', type=float, default='1')
parser.add_argument('--membership_mz_weight', type=float, default='0')

parser.add_argument('--train_reconstructor', type=str2bool, default='1')
parser.add_argument('--reconstruct_datasets', type=str2bool, default='1')

parser.add_argument('--ref_ratio', type=float, default=0.1)
parser.add_argument('--print_training', type=str2bool, default='True')

args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

# -- Directory -- #
args.output_path = os.path.join(args.base_path, args.output_dir, args.dataset)
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

args.data_path = os.path.join(args.base_path, 'data', args.dataset)
if not os.path.exists(args.data_path):
    os.makedirs(args.data_path)

if args.reconstruction_model == 'VAE':
    args.reconstruction_model += str(args.beta)

args.reconstruction_name = os.path.join(
    '{}_z{}_setsize{}_lr{}_ref{}_rw{}_cc{}_cm{}_mc{}_mm{}'.format(args.reconstruction_model, args.z_dim, args.setsize,
                                                                  args.lr, args.ref_ratio, args.recon_weight,
                                                                  args.class_cz_weight, args.class_mz_weight,
                                                                  args.membership_cz_weight, args.membership_mz_weight,
                                                                  ))

date = str(datetime.datetime.now())[:-16]
time = str(datetime.datetime.now())[-15:-7]
backup_path = os.path.join('backup', date, time + ' ' + args.reconstruction_name)
os.makedirs(backup_path)
for file in os.listdir(os.getcwd()):
    if file.endswith('.py'):
        shutil.copy2(file, backup_path)

args.reconstruction_path = os.path.join(args.output_path, 'reconstructor', args.reconstruction_name,
                                        'repeat{}'.format(args.repeat_idx))

print(args.reconstruction_path)

merged_dataset = load_dataset(args.dataset, args.data_path)
print(merged_dataset.__len__())

if args.dataset in ['adult', 'location']:
    args.encoder_input_dim = merged_dataset.__getitem__(0)[0].numpy().shape[0]
    if args.dataset == 'adult':
        args.class_num = 2
    elif args.dataset == 'location':
        args.class_num = 30

if args.dataset in ['MNIST', 'SVHN', 'CIFAR-10']:
    args.class_num = 10

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

reconstructor = Reconstructor(args)

if args.train_reconstructor:
    reconstructor.train(class_datasets['train'], class_datasets['valid'], ref_dataset)

class_datasets = {
    'train': subset0,
    # 'valid': subset1,
    # 'test': subset2,
    'out': subset3,
}
if args.reconstruct_datasets:
    reconstructor.reconstruct(class_datasets)
