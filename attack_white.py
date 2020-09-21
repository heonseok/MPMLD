import os

import numpy as np
import torch
import sys
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.utils.data import ConcatDataset
import torch.nn as nn
from module import MIAttacker
from sklearn import metrics


class Attacker(object):
    def __init__(self, args):
        self.train_batch_size = args.attack_train_batch_size
        self.test_batch_size = args.test_batch_size
        self.epochs = args.epochs
        self.early_stop = args.early_stop
        self.early_stop_observation_period = args.early_stop_observation_period
        self.resume = args.resume

        self.attack_path = args.attack_path
        self.attack_type = args.attack_type

        # Model
        # print('==> Building {}'.format(self.attack_path))
        if self.attack_type == 'black':
            if args.dataset == 'adult':
                net = MIAttacker(4)
            elif args.dataset in ['MNIST', 'Fashion-MNIST', 'CIFAR-10', 'SVHN']:
                # net = MIAttacker(20)
                net = MIAttacker(10)
            elif args.dataset == 'location':
                net = MIAttacker(60)
        elif self.attack_type == 'white':
            if args.dataset in ['MNIST', 'Fashion-MNIST', 'CIFAR-10', 'SVHN']:
                net = ConvMIAttacker()

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


        self.train_flag = False

        self.criterion = nn.BCELoss()
        # self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(net.parameters(), lr=args.attack_lr, betas=(0.5, 0.999))
        # self.optimizer = optim.SGD(net.parameters(), lr=args.attack_lr, momentum=0.9, weight_decay=5e-4)

    #########################
    # -- Base operations -- #
    #########################
    def load(self):
        # print('====> Loading checkpoint {}'.format(self.attack_path))
        checkpoint = torch.load(os.path.join(
            self.attack_path, 'ckpt.pth'), map_location=self.device)
        self.net.load_state_dict(checkpoint['net'])
        self.best_valid_acc = checkpoint['best_valid_acc']
        self.train_acc = checkpoint['train_acc']
        self.train_auroc = checkpoint['train_auroc']
        self.valid_auroc = checkpoint['valid_auroc']
        self.start_epoch = checkpoint['epoch']

    def train_epoch(self, train_loader, epoch):

        self.net.train()
        train_loss = 0
        predicted = []
        labels = []
        for batch_idx, (inputs, targets) in enumerate(train_loader):
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
            print('Epoch: {:>3}, Train Acc: {:.2f}, Valid Acc: {:.2f}'.format(epoch, self.train_acc, acc))
            if acc > self.best_valid_acc:
                # print('Saving..')
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
                print('Early stop count: {}'.format(self.early_stop_count))

            if self.early_stop_count == self.early_stop_observation_period:
                print('Early stop count == {}; Terminate training'.format(self.early_stop_observation_period))
                self.train_flag = False

        elif type == 'test':
            return acc, auroc

    def train(self, train_set, valid_set=None):
        print('==> Start training {}'.format(self.attack_path))

        if self.resume:
            print('==> Resuming from checkpoint..')
            try:
                self.load()
            except FileNotFoundError:
                print('There is no pre-trained model; Train model from scratch')

        self.train_flag = True
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.train_batch_size, shuffle=True,
                                                   num_workers=2)
        if self.early_stop:
            valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=self.test_batch_size, shuffle=True,
                                                       num_workers=2)
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            print('Epoch: {}'.format(epoch))
            if self.train_flag:
                self.train_epoch(train_loader, epoch)
                if self.early_stop:
                    self.inference(valid_loader, epoch, type='valid')
            else:
                break

    def test(self, test_set):
        # print('==> Test {}'.format(self.attack_path))
        try:
            self.load()
        except FileNotFoundError:
            print('There is no pre-trained model; First, train the classifier.')
            sys.exit(1)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.test_batch_size, shuffle=False,
                                                  num_workers=2)
        test_acc, test_auroc = self.inference(
            test_loader, epoch=-1, type='test')
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
        # print(auroc_dict)
        np.save(os.path.join(self.attack_path, 'auroc.npy'), auroc_dict)
