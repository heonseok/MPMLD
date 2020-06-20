import os

import numpy as np
from scipy.stats import entropy
import torch
import sys
import torch.optim as optim
import torch.backends.cudnn as cudnn

from utils import classify_membership
from utils import progress_bar
from utils import CustomDataset
from torch.utils.data import ConcatDataset
import torch.nn as nn
from module import SimpleNet
from sklearn import metrics


class Attacker(object):
    def __init__(self, args):
        self.train_batch_size = args.train_batch_size
        self.valid_batch_size = args.valid_batch_size
        self.test_batch_size = args.test_batch_size
        self.epochs = args.epochs
        self.early_stop = args.early_stop
        self.early_stop_observation_period = args.early_stop_observation_period

        self.cls_name = args.cls_name
        self.cls_path = args.cls_path
        self.attack_path = args.attack_path
        print(self.attack_path)
        if not os.path.exists(self.attack_path):
            os.makedirs(self.attack_path)

        self.attack_type = args.attack_type

        # Model
        print('==> Building {}'.format(self.attack_path))
        if self.attack_type == 'black':
            net = SimpleNet(20)

        self.start_epoch = 0
        self.best_valid_acc = 0
        self.train_acc = 0
        self.train_auroc = 0
        self.valid_auroc = 0
        self.early_stop_count = 0

        self.device = torch.device("cuda:{}".format(args.gpu_id))
        self.net = net.to(self.device)
        if self.device == 'cuda':
            cudnn.benchmark = True

        if args.resume:
            print('==> Resuming from checkpoint..')
            try:
                self.load()
            except FileNotFoundError:
                print('There is no pre-trained model; Train model from scratch')

        self.train_flag = False

        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    #########################
    # -- Base operations -- #
    #########################
    def load(self):
        print('====> Loading checkpoint {}'.format(self.attack_path))
        checkpoint = torch.load(os.path.join(self.attack_path, 'ckpt.pth'))
        self.net.load_state_dict(checkpoint['net'])
        self.best_valid_acc = checkpoint['best_valid_acc']
        self.train_acc = checkpoint['train_acc']
        self.train_auroc = checkpoint['train_auroc']
        self.valid_auroc = checkpoint['valid_auroc']
        self.start_epoch = checkpoint['epoch']

    def train_epoch(self, trainloader, epoch):
        self.net.train()
        train_loss = 0
        predicted = []
        labels = []
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs.squeeze(), targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            predicted_batch = outputs.cpu().detach().squeeze().numpy()
            labels_batch = targets.cpu().detach().numpy()

            if batch_idx == 0:
                predicted = predicted_batch
                labels = labels_batch
            else:
                predicted = np.concatenate((predicted, predicted_batch))
                labels = np.concatenate((labels, labels_batch))

        self.train_auroc = metrics.roc_auc_score(labels, predicted)
        self.train_acc = metrics.accuracy_score(labels, np.round(predicted))

    def inference(self, loader, epoch, type='valid'):
        self.net.eval()
        test_loss = 0
        predicted = []
        labels = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs.squeeze(), targets)

                test_loss += loss.item()
                predicted_batch = outputs.cpu().detach().squeeze().numpy()
                labels_batch = targets.cpu().detach().numpy()

                if batch_idx == 0:
                    predicted = predicted_batch
                    labels = labels_batch
                else:
                    predicted = np.concatenate((predicted, predicted_batch))
                    labels = np.concatenate((labels, labels_batch))

        auroc = metrics.roc_auc_score(labels, predicted)
        acc = metrics.accuracy_score(labels, np.round(predicted))

        if type == 'valid':
            print('\nEpoch: {:>3}, Train Acc: {:.2f}, Valid Acc: {:.2f}'.format(epoch, self.train_acc, acc))
            if acc > self.best_valid_acc:
                print('Saving..')
                state = {
                    'net': self.net.state_dict(),
                    'best_valid_acc': acc,
                    'train_acc': self.train_acc,
                    'train_auroc': self.train_auroc,
                    'valid_auroc': auroc,
                    'epoch': epoch,
                }
                torch.save(state, os.path.join(self.attack_path, 'ckpt.pth'))
                self.best_valid_acc = acc
                self.early_stop_count = 0
            else:
                self.early_stop_count += 1
                print('Eearly stop count: {}'.format(self.early_stop_count))

            if self.early_stop_count == self.early_stop_observation_period:
                print('Early stop count == {}; Terminate training'.format(self.early_stop_observation_period))
                self.train_flag = False

        elif type == 'test':
            return acc, auroc

    def train(self, trainset, validset=None):
        print('==> Start training {}'.format(self.cls_name))
        self.train_flag = True
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.train_batch_size, shuffle=True,
                                                  num_workers=2)
        if self.early_stop:
            validloader = torch.utils.data.DataLoader(validset, batch_size=self.valid_batch_size, shuffle=True,
                                                      num_workers=2)
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            if self.train_flag:
                self.train_epoch(trainloader, epoch)
                if self.early_stop:
                    self.inference(validloader, epoch, type='valid')
            else:
                break

    def test(self, testset):
        print('==> Test {}'.format(self.cls_name))
        try:
            self.load()
        except FileNotFoundError:
            print('There is no pre-trained model; First, train the classifier.')
            sys.exit(1)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.test_batch_size, shuffle=False, num_workers=2)
        test_acc, test_auroc = self.inference(testloader, epoch=-1, type='test')
        acc_dict = {
            'train': self.train_acc,
            'valid': self.best_valid_acc,
            'test': test_acc,
        }
        print(acc_dict)
        np.save(os.path.join(self.attack_path, 'acc.npy'), acc_dict)
        auroc_dict = {
            'train': self.train_auroc,
            'valid': self.valid_auroc,
            'test': test_auroc,
        }
        print(auroc_dict)
        np.save(os.path.join(self.attack_path, 'auroc.npy'), auroc_dict)

