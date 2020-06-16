import json
import os
import sys

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
from models import *

from utils import progress_bar


class Classifier(object):
    def __init__(self, args):
        self.train_batch_size = args.train_batch_size
        self.valid_batch_size = args.valid_batch_size
        self.test_batch_size = args.test_batch_size
        self.epochs = args.epochs
        self.early_stop = args.early_stop
        self.early_stop_observation_period = args.early_stop_observation_period

        self.model_name = os.path.join('{}_setsize{}'.format(args.model_type, args.setsize),
                                       'repeat{}'.format(args.repeat_idx))
        self.model_path = os.path.join(args.base_path, 'classifier', self.model_name)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        with open(os.path.join(self.model_path, 'descriptions.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        # Model
        print('==> Building {}'.format(self.model_name))
        if 'VGG' in args.model_type:
            print('VGG(\'' + args.model_type + '\')')
            net = eval('VGG(\'' + args.model_type + '\')')
        else:
            net = eval(args.model_type + '()')

        self.start_epoch = 0
        self.best_valid_acc = 0
        self.train_acc = 0
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

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    def load(self):
        print('==> Loading checkpoint {}'.format(self.model_name))
        checkpoint = torch.load(os.path.join(self.model_path, 'ckpt.pth'))
        self.net.load_state_dict(checkpoint['net'])
        self.best_valid_acc = checkpoint['best_valid_acc']
        self.train_acc = checkpoint['train_acc']
        self.start_epoch = checkpoint['epoch']

    def train_epoch(self, trainloader, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        self.train_acc = 100. * correct / total

    def inference(self, loader, epoch, type='valid'):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        acc = 100. * correct / total
        if type == 'valid':
            if acc > self.best_valid_acc:
                print('Saving..')
                state = {
                    'net': self.net.state_dict(),
                    'best_valid_acc': acc,
                    'train_acc': self.train_acc,
                    'epoch': epoch,
                }
                torch.save(state, os.path.join(self.model_path, 'ckpt.pth'))
                self.best_valid_acc = acc
                self.early_stop_count = 0
            else:
                self.early_stop_count += 1
                print('Eearly stop count: {}'.format(self.early_stop_count))

            if self.early_stop_count == self.early_stop_observation_period:
                print('Early stop count == {}; Terminate training'.format(self.early_stop_observation_period))
                self.train_flag = False

        elif type == 'test':
            # print('Test acc : {}'.format(acc))
            return acc

    def train(self, trainset, validset):
        print('==> Start training {}'.format(self.model_name))
        self.train_flag = True
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.train_batch_size, shuffle=True,
                                                  num_workers=2)
        validloader = torch.utils.data.DataLoader(validset, batch_size=self.valid_batch_size, shuffle=True,
                                                  num_workers=2)
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            if self.train_flag:
                self.train_epoch(trainloader, epoch)
                self.inference(validloader, epoch, type='valid')
            else:
                break

    def test(self, testset):
        print('==> Test {}'.format(self.model_name))
        try:
            self.load()
        except FileNotFoundError:
            print('There is no pre-trained model; First, train the classifier.')
            sys.exit(1)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.test_batch_size, shuffle=False, num_workers=2)
        test_acc = self.inference(testloader, epoch=-1, type='test')
        acc_dict = {
            'train': self.train_acc,
            'valid': self.best_valid_acc,
            'test': test_acc,
        }
        print(acc_dict)
        np.save(os.path.join(self.model_path, 'acc.npy'), acc_dict)
