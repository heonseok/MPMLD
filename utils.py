import os
from torch.utils.data import ConcatDataset
from scipy.stats import entropy
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
import sys
import time

import numpy as np
import torch.nn as nn
import torch.nn.init as init
from sklearn import metrics


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def str2bool(s):
    if s.lower() in ('yes', 'y', '1', 'true', 't'):
        return True
    elif s.lower() in ('no', 'n', '0', 'false', 'f'):
        return False


def classify_membership(data_in, data_out):
    data_concat = np.concatenate((data_in, data_out))
    sorted_idx = np.argsort(data_concat)
    inout_truth = np.concatenate((np.ones_like(data_in), np.zeros_like(data_out)))
    inout_truth_sorted = inout_truth[sorted_idx]

    inout_pred = np.concatenate((np.ones_like(data_out), np.zeros_like(data_in)))

    acc = metrics.accuracy_score(inout_truth_sorted, inout_pred)
    auroc = metrics.roc_auc_score(inout_truth_sorted, inout_pred)

    return acc, auroc


class NonIIDCustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = torch.FloatTensor(data)
        self.targets = targets
        # self.data = data.copy()
        # self.targets = targets.copy()
        self.transform = transform 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        if self.transform is not None:
            return self.transform(self.data[idx]), self.targets[idx]
        else:
            return self.data[idx], self.targets[idx]

class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.FloatTensor(data)
        self.targets = targets
        # self.data = torch.FloatTensor(data).copy()
        # self.targets = targets.copy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        return self.data[idx], self.targets[idx]

class MIDataset(Dataset):
    def __init__(self, responses, class_labels, membership_labels):
        self.responses = responses
        self.class_labels = class_labels
        self.membership_labels = membership_labels

    def __len__(self):
        return len(self.responses)

    def __getitem__(self, idx):
        return self.responses[idx], self.class_labels[idx], self.membership_labels[idx]



def concat_datasets(in_dataset, out_dataset, start, end):
    return ConcatDataset([
        Subset(in_dataset, range(int(start * len(in_dataset)), int(end * len(in_dataset)))),
        Subset(out_dataset, range(int(start * len(out_dataset)), int(end * len(out_dataset))))
    ])


def statistical_attack(cls_path, attack_path):
    features = np.load(os.path.join(cls_path, 'features.npy'), allow_pickle=True).item()

    print('==> Statistical attack')
    in_entropy = entropy(features['in']['preds'], base=2, axis=1)
    out_entropy = entropy(features['out']['preds'], base=2, axis=1)
    acc, auroc = classify_membership(in_entropy, out_entropy)
    np.save(os.path.join(attack_path, 'acc.npy'), acc)
    np.save(os.path.join(attack_path, 'auroc.npy'), auroc)

    print('Statistical attack accuracy : {:.2f}'.format(acc))
    # todo: sort by confidence score


def build_reconstructed_datasets(reconstruction_path):
    recon_datasets_ = torch.load(reconstruction_path)
    recon_datasets = {}
    for dataset_type, dataset in recon_datasets_.items():
        recon_datasets[dataset_type] = CustomDataset(dataset['recons'], dataset['labels'])
    return recon_datasets


def build_inout_features(total_responses, attack_type):
    labels = total_responses['labels']
    labels_onehot = torch.zeros((len(labels), len(np.unique(labels)))).scatter_(1, torch.LongTensor(labels).reshape( (-1, 1)), 1)

    if attack_type == 'black':
        responses = torch.Tensor(total_responses['preds'])
    elif attack_type == 'white':
        activations = total_responses['activations']
        # for act in activations:
        #     print(act.shape)
        # sys.exit(1)
        responses = torch.Tensor(activations[1])

    return responses, labels_onehot

def build_inout_feature_sets(classification_path, attack_type):
    features = np.load(os.path.join(classification_path,
                                    'features.npy'), allow_pickle=True).item()
    
    in_features, in_class_labels = build_inout_features(features['in'], attack_type)
    out_features, out_class_labels = build_inout_features(features['out'], attack_type)

    in_feature_set = MIDataset(in_features, in_class_labels, torch.ones(in_features.shape[0]))
    out_feature_set = MIDataset(out_features, out_class_labels, torch.zeros(out_features.shape[0]))

    inout_feature_sets = {
        'train': concat_datasets(in_feature_set, out_feature_set, 0, 0.7),
        'valid': concat_datasets(in_feature_set, out_feature_set, 0.7, 0.85),
        'test': concat_datasets(in_feature_set, out_feature_set, 0.85, 1.0),
    }
    return inout_feature_sets
