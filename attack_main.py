import argparse
import os

import torch

from utils import str2bool
from attack import Attacker
from utils import build_inout_feature_sets
import utils
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='adult', choices=['MNIST', 'Fashion-MNIST', 'CIFAR-10', 'adult', 'location'])
parser.add_argument('--setsize', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--base_path', type=str, default='/mnt/disk1/heonseok/MPMLD')
parser.add_argument('--resume', type=str2bool, default='0')
parser.add_argument('--train_batch_size', type=int, default=100)
parser.add_argument('--valid_batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=100)
parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--early_stop', type=str2bool, default='1')
parser.add_argument('--early_stop_observation_period', type=int, default=20)
parser.add_argument('--repeat_idx', type=int, default=0)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--attack_type', type=str, default='black', choices=['black', 'white'])
parser.add_argument('--recon_type', type=str, default='full_z', choices=['full_z', 'content_z', 'style_z'])

parser.add_argument('--target_classifier', type=str, default='FCN_setsize100_AE_z8_base')
# parser.add_argument('--target_classifier', type=str, default='FCN_setsize100_original')
# parser.add_argument('--target_classifier', type=str, default='ResNet18_setsize10000_original')

# parser.add_argument('--dataset_type', type=str, default='original', choices=['original', 'reconstructed'])
parser.add_argument('--reconstruction_path', type=str, default='todo...')

parser.add_argument('--train_attacker', type=str2bool, default='1')
parser.add_argument('--test_attacker', type=str2bool, default='1')

parser.add_argument('--statistical_attack', type=str2bool, default='1')

args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

args.output_path = os.path.join(args.base_path, 'output', args.dataset)
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

args.target_classifier = os.path.join(args.target_classifier, args.recon_type)

args.classification_name = os.path.join(args.target_classifier, 'repeat{}'.format(args.repeat_idx))
# if 'original' in args.target_classifier:
# # if args.dataset_type == 'original':
# else:
#     args.classification_name = os.path.join(
#         '{}_setsize{}_{}'.format(args.classification_model, args.setsize, args.reconstruction_path),
#         'repeat{}'.format(args.repeat_idx))

args.classification_path = os.path.join(args.output_path, 'classifier', args.classification_name)
print(args.classification_path)

# -- Run -- #

if args.statistical_attack:
    args.attack_path = os.path.join(args.output_path, 'attacker', args.classification_name, 'stat')
    if not os.path.exists(args.attack_path):
        os.makedirs(args.attack_path)
    utils.statistical_attack(args.classification_path, args.attack_path)

if args.train_attacker or args.test_attacker:
    args.attack_path = os.path.join(args.output_path, 'attacker', args.classification_name, args.attack_type)
    if not os.path.exists(args.attack_path):
        os.makedirs(args.attack_path)

    inout_feature_sets = build_inout_feature_sets(args.classification_path, args.attack_type)
    for dataset_type, dataset in inout_feature_sets.items():
        print('Inout {:<3} feature set: {}'.format(dataset_type, len(dataset)))

    attacker = Attacker(args)
    if args.train_attacker:
        attacker.train(inout_feature_sets['train'], inout_feature_sets['valid'])
    if args.test_attacker:
        attacker.test(inout_feature_sets['test'])
