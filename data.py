import torchvision
import torchvision.transforms as transforms
import sys
import numpy as np
import os
from utils import CustomDataset
from torch.utils.data import ConcatDataset
import torch


def load_dataset(dataset, data_path):
    print('==> Preparing data..')
    if dataset == 'CIFAR-10':
        transform_train = transforms.Compose([
            # transforms.RandomCrop(64, padding=4),
            # transforms.Resize(64),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=transform_train)

        testset = torchvision.datasets.CIFAR10(
            root=data_path, train=False, download=True, transform=transform_test)

        concat_set = ConcatDataset((trainset, testset))

    elif dataset == 'adult':
        # print(data_path)
        dataset = np.load(os.path.join(data_path, 'preprocessed.npy'), allow_pickle=True).item()
        print(dataset['data'].shape)
        # print(dataset)

        # concat_set = CustomDataset(torch.FloatTensor(dataset['data']), torch.FloatTensor(dataset['label']))
        concat_set = CustomDataset(torch.FloatTensor(dataset['data']), dataset['label'])
        # sys.exit(1)
        # pass

    return concat_set
    # return trainset, testset

