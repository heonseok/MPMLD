import argparse
import os

import torch
from torch.utils.data import Subset

from data import load_dataset
from utils import str2bool
from classification import Classifier
from attack import Attacker
from disentanglement_class import Disentangler
# from disentanglement import Disentangler
from utils import build_inout_feature_sets
import utils
import numpy as np
import sys

parser = argparse.ArgumentParser(description='Membership Privacy-preserving Machine Learning models by Disentanglement')
parser.add_argument('--dataset', type=str, default='CIFAR-10', choices=['CIFAR-10'])
parser.add_argument('--setsize', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--base_path', type=str, default='/mnt/disk1/heonseok/MPMLD')
parser.add_argument('--resume', type=str2bool, default='0')
parser.add_argument('--train_batch_size', type=int, default=100)
parser.add_argument('--valid_batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=100)
parser.add_argument('--classification_model', type=str, default='ResNet18', choices=['VGG19', 'ResNet18'])
parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--early_stop', type=str2bool, default='1')
parser.add_argument('--early_stop_observation_period', type=int, default=20)
parser.add_argument('--repeat_idx', type=int, default=0)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--attack_type', type=str, default='black', choices=['black', 'white'])
parser.add_argument('--z_dim', type=int, default=16)
parser.add_argument('--disentanglement_type', type=str, default='base', choices=['base', 'type1', 'type2'])
# parser.add_argument('--reconstruction_type', type=str, default='full_z', choices=['full_z', 'partial_z'])
parser.add_argument('--reconstruction_type', type=str, default='partial_z', choices=['full_z', 'partial_z'])

parser.add_argument('--train_classifier', type=str2bool, default='0')
parser.add_argument('--test_classifier', type=str2bool, default='0')
parser.add_argument('--extract_classifier_features', type=str2bool, default='0')
# parser.add_argument('--use_reconstructed_datasets', type=str2bool, default='0')

parser.add_argument('--statistical_attack', type=str2bool, default='0')
parser.add_argument('--train_attacker', type=str2bool, default='0')
parser.add_argument('--test_attacker', type=str2bool, default='0')

parser.add_argument('--train_disentangler', type=str2bool, default='0')
parser.add_argument('--reconstruct_datasets', type=str2bool, default='0')

args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

# -- Directory -- #
if not os.path.exists(args.base_path):
    os.mkdir(args.base_path)

args.data_path = os.path.join(args.base_path, 'data', args.dataset)
if not os.path.exists(args.data_path):
    os.makedirs(args.data_path)

args.disentanglement_name = os.path.join('disentangler_z{}'.format(args.z_dim), args.disentanglement_type)
args.disentanglement_path = os.path.join(args.base_path, args.disentanglement_name, 'repeat{}'.format(args.repeat_idx))
# args.reconstruction_name = os.path.join(args.disentanglement_name, args.reconstruction_type)
args.reconstruction_path = os.path.join(args.disentanglement_path,
                                        'recon_{}.pt'.format(args.reconstruction_type))

if args.use_reconstructed_datasets:
    try:
        class_datasets = utils.build_reconstructed_datasets(args.reconstruction_path)
    except FileNotFoundError:
        print('There is no reconstructed data')
        sys.exit(1)
    args.classification_name = os.path.join(
        '{}_setsize{}_{}'.format(args.classification_model, args.setsize, args.reconstruction_path),
        'repeat{}'.format(args.repeat_idx))
else:
    trainset, testset = load_dataset(args.dataset, args.data_path)

    subset0 = Subset(trainset, range(0, args.setsize))
    subset1 = Subset(trainset, range(args.setsize, int(1.2 * args.setsize)))
    subset2 = Subset(trainset, range(int(1.2 * args.setsize), int(1.4 * args.setsize)))
    subset3 = Subset(testset, range(0, args.setsize))

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
    args.classification_name = os.path.join('{}_setsize{}'.format(args.classification_model, args.setsize),
                                            'repeat{}'.format(args.repeat_idx))

for dataset_type, dataset in class_datasets.items():
    print('Class {:<5} dataset: {}'.format(dataset_type, len(dataset)))
print()
for dataset_type, dataset in inout_datasets.items():
    print('Inout {:<3} dataset: {}'.format(dataset_type, len(dataset)))
print()

args.classification_path = os.path.join(args.base_path, 'classifier', args.classification_name)
if args.statistical_attack:
    args.attack_path = os.path.join(args.base_path, 'attacker', args.classification_name, 'stat')
else:
    args.attack_path = os.path.join(args.base_path, 'attacker', args.classification_name, args.attack_type)
if not os.path.exists(args.attack_path):
    os.makedirs(args.attack_path)

# -- Run -- #
if args.train_classifier or args.test_classifier or args.extract_classifier_features:

    classifier = Classifier(args)

    if args.train_classifier:
        classifier.train(class_datasets['train'], class_datasets['valid'])
    if args.test_classifier:
        classifier.test(class_datasets['test'])
    if args.extract_classifier_features:
        classifier.extract_features(inout_datasets)

if args.train_attacker or args.test_attacker:
    inout_feature_sets = build_inout_feature_sets(args.classification_path, args.attack_type)
    for dataset_type, dataset in inout_feature_sets.items():
        print('Inout {:<3} feature set: {}'.format(dataset_type, len(dataset)))
    attacker = Attacker(args)
    if args.train_attacker:
        attacker.train(inout_feature_sets['train'], inout_feature_sets['valid'])
    if args.test_attacker:
        attacker.test(inout_feature_sets['test'])

if args.statistical_attack:
    utils.statistical_attack(args.classification_path, args.attack_path)

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
