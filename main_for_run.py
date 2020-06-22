import argparse
import os

import torch
from torch.utils.data import Subset

from data import load_dataset
from utils import str2bool
from classification import Classifier
from attack import Attacker
from disentanglement import Disentangler
from utils import build_inout_dataset
import utils

parser = argparse.ArgumentParser(description='Membership Privacy-preserving Machine Learning models by Disentanglement')
parser.add_argument('--dataset', type=str, default='CIFAR-10', choices=['CIFAR-10'])
parser.add_argument('--setsize', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--base_path', type=str, default='/mnt/disk1/heonseok/MPMLD')
parser.add_argument('--resume', type=str2bool, default='0')
parser.add_argument('--train_batch_size', type=int, default=100)
parser.add_argument('--valid_batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=100)
parser.add_argument('--classification_model', type=str, default='VGG19', choices=['VGG19', 'ResNet18'])
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--early_stop', type=str2bool, default='1')
parser.add_argument('--early_stop_observation_period', type=int, default=20)
parser.add_argument('--repeat_idx', type=int, default=0)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--attack_type', type=str, default='black', choices=['black', 'white'])

parser.add_argument('--train_classifier', type=str2bool, default='0')
parser.add_argument('--test_classifier', type=str2bool, default='0')
parser.add_argument('--extract_classifier_features', type=str2bool, default='0')

parser.add_argument('--train_attacker', type=str2bool, default='0')
parser.add_argument('--test_attacker', type=str2bool, default='0')
parser.add_argument('--statistical_attack', type=str2bool, default='0')

parser.add_argument('--train_disentangler', type=str2bool, default='0')
parser.add_argument('--z_dim', type=int, default=64)

args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

# -- Directory -- #
if not os.path.exists(args.base_path):
    os.mkdir(args.base_path)

args.data_path = os.path.join(args.base_path, 'data', args.dataset)
if not os.path.exists(args.data_path):
    os.makedirs(args.data_path)

args.classification_name = os.path.join('{}_setsize{}'.format(args.classification_model, args.setsize),
                                        'repeat{}'.format(args.repeat_idx))
args.classification_path = os.path.join(args.base_path, 'classifier', args.classification_name)

if args.statistical_attack:
    args.attack_path = os.path.join(args.base_path, 'attacker', args.classification_name, 'stat')
else:
    args.attack_path = os.path.join(args.base_path, 'attacker', args.classification_name, args.attack_type)

args.disentanglement_path = os.path.join(args.base_path, 'disentangler')

# -- Dataset -- #
trainset, testset = load_dataset(args.dataset, args.data_path)

subset0 = Subset(trainset, range(args.setsize))
subset1 = Subset(trainset, range(args.setsize, 2 * args.setsize))
subset2 = Subset(trainset, range(2 * args.setsize, 3 * args.setsize))

class_datasets = {
    'train': subset0,
    'valid': subset1,
    'test': subset2,
}
for dataset_type, dataset in class_datasets.items():
    print('Cls {:<5} : {}'.format(dataset_type, len(dataset)))

# -- Run -- #
if args.train_classifier or args.test_classifier or args.extract_classifier_features:

    classifier = Classifier(args)

    if args.train_classifier:
        classifier.train(class_datasets['train'], class_datasets['valid'])
    if args.test_classifier:
        classifier.test(class_datasets['test'])
    if args.extract_classifier_features:
        classifier.extract_features(class_datasets)

if args.train_attacker or args.test_attacker:
    inout_dataset = build_inout_dataset(args.classification_path, args.attack_type)
    attacker = Attacker(args)
    if args.train_attacker:
        attacker.train(inout_dataset['train'], inout_dataset['valid'])
    if args.test_attacker:
        attacker.test(inout_dataset['test'])

if args.statistical_attack:
    utils.statistical_attack(args.classification_path, args.attack_path)

if args.train_disentangler:
    disentangler = Disentangler(args)
    disentangler.train(class_datasets['train'])
