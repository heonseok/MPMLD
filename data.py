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
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=transform_train)

        testset = torchvision.datasets.CIFAR10(
            root=data_path, train=False, download=True, transform=transform_test)

        total_set = ConcatDataset((trainset, testset))

    elif dataset == 'CIFAR-100':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR100(
            root=data_path, train=True, download=True, transform=transform_train)

        testset = torchvision.datasets.CIFAR100(
            root=data_path, train=False, download=True, transform=transform_test)

        total_set = ConcatDataset((trainset, testset))

    elif dataset == 'MNIST':
        transform_train = transforms.Compose([
            # transforms.Resize(32),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        transform_test = transforms.Compose([
            # transforms.Resize(32),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        trainset = torchvision.datasets.MNIST(
            root=data_path, train=True, download=True, transform=transform_train)

        testset = torchvision.datasets.MNIST(
            root=data_path, train=False, download=True, transform=transform_test)

        total_set = ConcatDataset((trainset, testset))

    elif dataset == 'Fashion-MNIST':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        trainset = torchvision.datasets.FashionMNIST(
            root=data_path, train=True, download=True, transform=transform_train)

        testset = torchvision.datasets.FashionMNIST(
            root=data_path, train=False, download=True, transform=transform_test)

        total_set = ConcatDataset((trainset, testset))

    elif dataset == 'adult':
        # todo : import pre processing code
        dataset = np.load(os.path.join(data_path, 'preprocessed.npy'), allow_pickle=True).item()
        total_set = CustomDataset(torch.FloatTensor(dataset['data']), dataset['label'])

    elif dataset == 'location':
        dataset = np.load(os.path.join(data_path, 'data_complete.npz'))
        # print(np.unique(dataset['y']-1))
        total_set = CustomDataset(torch.FloatTensor(dataset['x']), dataset['y']-1)

    return total_set
