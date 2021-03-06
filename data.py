import torchvision
import torchvision.transforms as transforms
import sys
import numpy as np
import os
from utils import CustomDataset
from torch.utils.data import Dataset, Subset, ConcatDataset
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.utils as vutils
from utils import CustomDataset, NonIIDCustomDataset

THIRD_QUANTILE = np.int64(130560) * 1.5
MEDIAN = np.int64(130560)

RED = 0
GREEN = 1
BLUE = 2

def load_non_iid_dataset(dataset, data_path, scenario):
    print('==> Preparing non-iid data..')
    target_channel = BLUE

    if dataset == 'SVHN':

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        trainset = torchvision.datasets.SVHN(root=data_path, split='train', download=True, transform=transform_train)
        testset = torchvision.datasets.SVHN(root=data_path, split='test', download=True, transform=transform_test)

        # totalset = ConcatDataset((trainset, testset))
        totalset = trainset

    else:
        pass

    if scenario == 'color_abs':
       # raw = trainset.data[0:100]
                # vutils.save_image(torch.FloatTensor(raw), 'raw.png', nrow=10, normalize=True)

        # for i in range(3):
        #     test = raw.copy()
        #     test[:, i, :, :]  = 0
        #     vutils.save_image(torch.FloatTensor(test), 'test{}.png'.format(i), nrow=10, normalize=True)

        # for i in range(3):
        #     test = raw.copy()
        #     test[:, (i+1)%3, :, :]  = 0
        #     test[:, (i+2)%3, :, :]  = 0
        #     vutils.save_image(torch.FloatTensor(test), 'test-{}.png'.format(i), nrow=10, normalize=True)

        # test = raw.copy()

        # blue_sum_list = []
        # for i in range(test.shape[0]):
        #     blue_sum_list.append(np.sum(test[i, 2, :, :]))

        # idx = blue_sum_list > np.average(blue_sum_list)
        # vutils.save_image(torch.FloatTensor(test[idx]), 'blud_filtered.png', nrow=10, normalize=True)

        # idx = blue_sum_list > np.int64(130560)
        # vutils.save_image(torch.FloatTensor(test[idx]), 'blud_filtered2.png', nrow=10, normalize=True)

        # print(trainset.__len__())
        # print(trainset_blue.__len__())
        # print(trainset_non_blue.__len__())

        # vutils.save_image(torch.FloatTensor(total_set.datasets[0:100]), 'raw.png', nrow=10, normalize=True)

        # trainset_blue_idx = trainset.data
        # total_set = ConcatDataset((trainset, testset))
        # print(total_set.datasets[0:100])
        # vutils.save_image(torch.FloatTensor(total_set.datasets[0:100]), 'raw.png', nrow=10, normalize=True)

        # print(dataset_target_color[0:100][0].shape)
        # vutils.save_image(torch.FloatTensor(dataset_target_color[0:100][0]), 'blue.png', nrow=10, normalize=True)
        # vutils.save_image(torch.FloatTensor(dataset_non_target_color[0:100][0]), 'non_blue.png', nrow=10, normalize=True)
        # sys.exit(1)

        in_dataset, out_dataset = split_imgs_by_color_abs(totalset, target_channel)

    elif scenario == 'color_zero':
        # print(totalset)
        # print(totalset.data)
        # sys.exit(1)

        in_dataset = totalset
        out_dataset = torchvision.datasets.SVHN(root=data_path, split='train', download=True, transform=transform_train)
        out_dataset.data[:, target_channel, :, :]  = 0.0

        # in_dataset = totalset
        # transform_op = transforms.Compose([
        #     # transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        # ])
        # in_dataset, out_dataset = extract_zero_color_imgs(totalset, target_channel, transform_op)
        # print(in_dataset)
        # print(out_dataset)

    vutils.save_image(torch.FloatTensor(in_dataset.data[0:100]), '{}_{}_in.png'.format(scenario, target_channel), nrow=10, normalize=True)
    vutils.save_image(torch.FloatTensor(out_dataset.data[0:100]), '{}_{}_out.png'.format(scenario, target_channel), nrow=10, normalize=True)

    return in_dataset, out_dataset  

def split_imgs_by_color_abs(totalset, target_color_channel):
    target_color_sum_list = []
    for i in range(totalset.data.shape[0]):
        target_color_sum_list.append(np.sum(totalset.data[i, target_color_channel, :, :]))

    target_color_bool = (target_color_sum_list > THIRD_QUANTILE)
    non_target_color_bool = (target_color_sum_list < MEDIAN)

    target_color_idx = [i for i, elem in enumerate(target_color_bool) if elem]
    non_target_color_idx = [i for i, elem in enumerate(non_target_color_bool) if elem]

    dataset_target_color = Subset(totalset, target_color_idx)
    dataset_non_target_color = Subset(totalset, non_target_color_idx)

    return dataset_target_color, dataset_non_target_color

def extract_zero_color_imgs(totalset, target_color_channel, transform_op):    
    in_dataset = NonIIDCustomDataset(totalset.data, totalset.labels, transform_op)
    out_dataset = NonIIDCustomDataset(totalset.data, totalset.labels, transform_op)
    # print(out_dataset.data[:, target_color_channel, :, :])
    # out_dataset.data[:, target_color_channel, :, :]  = out_dataset.data[:, target_color_channel, :, :] / 2
    # out_dataset.data[:, target_color_channel, :, :]  = 0.0out_dataset.data[:, target_color_channel, :, :]
    # print(out_dataset.data[:, target_color_channel, :, :])

    return in_dataset, out_dataset 

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

        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)

        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)

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

        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)

        testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)

        total_set = ConcatDataset((trainset, testset))

    elif dataset == 'MNIST':
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(32),
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
            transforms.Resize(32),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        trainset = torchvision.datasets.FashionMNIST(
            root=data_path, train=True, download=True, transform=transform_train)

        testset = torchvision.datasets.FashionMNIST(
            root=data_path, train=False, download=True, transform=transform_test)

        total_set = ConcatDataset((trainset, testset))

    elif dataset == 'SVHN':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        trainset = torchvision.datasets.SVHN( root=data_path, split='train', download=True, transform=transform_train)

        testset = torchvision.datasets.SVHN( root=data_path, split='test', download=True, transform=transform_test)

        total_set = ConcatDataset((trainset, testset))

        

    elif dataset == 'adult':
        # todo : import pre processing code
        dataset = np.load(os.path.join(data_path, 'preprocessed.npy'), allow_pickle=True).item()
        total_set = CustomDataset(torch.FloatTensor(dataset['data']), dataset['label'])

    elif dataset == 'location':
        dataset = np.load(os.path.join(data_path, 'data_complete.npz'))
        # print(np.unique(dataset['y']-1))
        total_set = CustomDataset(torch.FloatTensor(dataset['x']), dataset['y'] - 1)

    return total_set


class DoubleDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        return self.dataset1[idx][0], self.dataset1[idx][1], self.dataset2[idx][0], self.dataset2[idx][1]
        # return self.dataset1[idx], self.dataset2[idx]

# class ConcatDataset(Dataset):
#     def __init__(self, dataset1, dataset2):
#         self.da
