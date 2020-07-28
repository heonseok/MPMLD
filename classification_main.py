import argparse
import os

import torch
from torch.utils.data import Subset

from data import load_dataset
from utils import str2bool
from classification import Classifier
import utils
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='SVHN',
                    choices=['MNIST', 'Fashion-MNIST', 'CIFAR-10', 'CIFAR-100', 'adult', 'location', 'SVHN'])
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--output_dir', type=str, default='output0727')
parser.add_argument('--base_path', type=str, default='/mnt/disk1/heonseok/MPMLD')
parser.add_argument('--resume', type=str2bool, default='0')
parser.add_argument('--train_batch_size', type=int, default=100)
parser.add_argument('--valid_batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=100)
parser.add_argument('--classification_model', type=str, default='ResNet18',
                    choices=['FCClassifier', 'ConvClassifier', 'VGG19', 'ResNet18', 'ResNet50', 'ResNet101',
                             'DenseNet121'])
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--early_stop', type=str2bool, default='1')
parser.add_argument('--early_stop_observation_period', type=int, default=20)
parser.add_argument('--repeat_idx', type=int, default=0)
parser.add_argument('--gpu_id', type=int, default=0)

parser.add_argument('--target_data', type=str, default='original_setsize10000')
parser.add_argument('--recon_type', type=str, default='cb_mb')

parser.add_argument('--train_classifier', type=str2bool, default='1')
parser.add_argument('--test_classifier', type=str2bool, default='1')
parser.add_argument('--extract_classifier_features', type=str2bool, default='1')

parser.add_argument('--print_training', type=str2bool, default='1')

args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

# -- Directory -- #
args.output_path = os.path.join(args.base_path, args.output_dir, args.dataset)
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

args.data_path = os.path.join(args.base_path, 'data', args.dataset)
if not os.path.exists(args.data_path):
    os.makedirs(args.data_path)

merged_dataset = load_dataset(args.dataset, args.data_path)
print(merged_dataset.__len__())
print('Number of features: ' + str(merged_dataset.__getitem__(0)[0].numpy().shape[0]))

if 'original' in args.target_data:
    args.setsize = int(args.target_data.split('_')[1][7:])
else:
    args.setsize = int(args.target_data.split('_')[2][7:])

if args.setsize * 2.4 > len(merged_dataset):
    print('Setsize * 2.4 > len(concatset); Terminate program')
    sys.exit(1)

subset0 = Subset(merged_dataset, range(0, args.setsize))
subset1 = Subset(merged_dataset, range(args.setsize, int(1.2 * args.setsize)))
subset2 = Subset(merged_dataset, range(int(1.2 * args.setsize), int(1.4 * args.setsize)))
subset3 = Subset(merged_dataset, range(int(1.4 * args.setsize), int(2.4 * args.setsize)))

class_datasets = {
    'train': subset0,
    'valid': subset1,
    'test': subset2,
}

if 'original' in args.target_data:
    args.classification_name = os.path.join('{}_{}'.format(args.target_data, args.classification_model),
                                            'repeat{}'.format(args.repeat_idx))
else:
    args.classification_name = os.path.join('{}_{}'.format(args.target_data, args.classification_model),
                                            args.recon_type, 'repeat{}'.format(args.repeat_idx))

    args.reconstruction_path = os.path.join(args.output_path, 'reconstructor', args.target_data,
                                            'repeat{}'.format(args.repeat_idx), 'recon_{}.pt'.format(args.recon_type))
    try:
        recon_datasets = utils.build_reconstructed_datasets(args.reconstruction_path)
        class_datasets['train'] = recon_datasets['train']
    except FileNotFoundError:
        print('There is no reconstructed data: ', args.reconstruction_path)
        sys.exit(1)

for dataset_type, dataset in class_datasets.items():
    print('Class {:<5} dataset: {}'.format(dataset_type, len(dataset)))
print()

args.classification_path = os.path.join(args.output_path, 'classifier', args.classification_name)
print(args.classification_path)

# -- Run -- #
classifier = Classifier(args)

if args.train_classifier:
    classifier.train(class_datasets['train'], class_datasets['valid'])

if args.test_classifier:
    classifier.test(class_datasets['test'])

if args.extract_classifier_features:
    # inout_datasets should be transformed to inout_feature_sets for training attacker
    inout_datasets = {
        'in': subset0,
        'out': subset3,
    }

    if 'original' not in args.target_data:
        inout_datasets['in'] = recon_datasets['train']
        inout_datasets['out'] = recon_datasets['out']

    for dataset_type, dataset in inout_datasets.items():
        print('Inout {:<3} dataset: {}'.format(dataset_type, len(dataset)))
    print()
    classifier.extract_features(inout_datasets)
