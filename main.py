import argparse
import os

import torch
from torch.utils.data import Subset

from classification import Classifier
from data import load_dataset
from utils import str2bool

parser = argparse.ArgumentParser(description='Membership Privacy-preserving Machine Learning models by Disentanglement')
parser.add_argument('--dataset', type=str, default='CIFAR-10', choices=['CIFAR-10'])
parser.add_argument('--setsize', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--base_path', type=str, default='/mnt/disk1/heonseok/MPMLD')
parser.add_argument('--resume', type=str2bool, default='0')
parser.add_argument('--train_batch_size', type=int, default=100)
parser.add_argument('--valid_batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=100)
parser.add_argument('--model_type', type=str, default='ResNet18', choices=['VGG19', 'ResNet18'])
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--early_stop', type=str2bool, default='t')
parser.add_argument('--early_stop_observation_period', type=int, default=20)
parser.add_argument('--repeat_idx', type=int, default=0)
parser.add_argument('--gpu_id', type=int, default=0)
args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

if not os.path.exists(args.base_path):
    os.mkdir(args.base_path)

args.data_path = os.path.join(args.base_path, 'data', args.dataset)
if not os.path.exists(args.data_path):
    os.makedirs(args.data_path)

trainset, testset = load_dataset(args.dataset, args.data_path)

subset0 = Subset(trainset, range(args.setsize))
subset1 = Subset(trainset, range(args.setsize, 2 * args.setsize))
subset2 = Subset(trainset, range(args.setsize))

cls_datasets = {
    'train': subset0,
    'valid': subset1,
    'test': subset2,
}

for dataset_type, dataset in cls_datasets.items():
    print('Cls {:<5} : {}'.format(dataset_type, len(dataset)))

cls_model = Classifier(args)
# cls_model.train(cls_datasets['train'], cls_datasets['valid'])
# cls_model.test(cls_datasets['test'])

cls_model.log_prediction_score(cls_datasets)
