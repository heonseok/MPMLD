import json
import os
import sys

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
from models import *
import module


class Classifier(object):
    def __init__(self, args):
        self.train_batch_size = args.train_batch_size
        self.valid_batch_size = args.valid_batch_size
        self.test_batch_size = args.test_batch_size
        self.epochs = args.epochs
        self.early_stop = args.early_stop
        self.early_stop_observation_period = args.early_stop_observation_period

        self.print_training = args.print_training
        # self.classification_name = args.classification_name
        self.classification_path = args.classification_path
        if not os.path.exists(self.classification_path):
            os.makedirs(self.classification_path)

        with open(os.path.join(self.classification_path, 'descriptions.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        # Model
        print('==> Building {}'.format(self.classification_path))

        if args.dataset in ['CIFAR-10', 'SVHN']:
            if 'VGG' in args.classification_model:
                print('VGG(\'' + args.classification_model + '\')')
                net = eval('VGG(\'' + args.classification_model + '\')')
            else:
                net = eval(args.classification_model + '()')
        elif args.dataset == 'CIFAR-100':
            # net = eval(args.classification_model + '(\'num_classes\'=100)')
            # net = DenseNet121(num_classes=100)
            net = DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32, num_classes=100)
        elif args.dataset == 'adult':
            net = module.FCClassifier(108, output_dim=2)
        elif args.dataset == 'location':
            if args.classifier_type == 'A':
                net = module.FCClassifier(446, output_dim=30)
            elif args.classifier_type == 'B':
                net = module.FCNClassifierB(446, output_dim=30)

        elif args.dataset in ['MNIST', 'Fashion-MNIST']:
            net = module.ConvClassifier()

        # print(net)

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

        # self.criterion = nn.Cro
        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.999, 0.999))
        self.optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.5, 0.999))
        # self.optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
        # self.optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)

    #########################
    # -- Base operations -- #
    #########################
    def load(self):
        print('====> Loading checkpoint {}'.format(self.classification_path))
        checkpoint = torch.load(os.path.join(self.classification_path, 'ckpt.pth'))
        self.net.load_state_dict(checkpoint['net'])
        self.best_valid_acc = checkpoint['best_valid_acc']
        self.train_acc = checkpoint['train_acc']
        self.start_epoch = checkpoint['epoch']

    def train_epoch(self, trainloader, epoch):
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
            # print(predicted)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        self.train_acc = correct / total

        if not self.early_stop:
            print('Epoch: {:>3}, Train Acc: {:.4f}'.format(epoch, self.train_acc))

            if epoch == self.epochs - 1:
                state = {
                    'net': self.net.state_dict(),
                    'best_valid_acc': -1,
                    'train_acc': self.train_acc,
                    'epoch': epoch,
                }
                torch.save(state, os.path.join(self.classification_path, 'ckpt.pth'))

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

        acc = correct / total
        if type == 'valid':
            if self.print_training:
                print('Epoch: {:>3}, Train Acc: {:.4f}, Valid Acc: {:.4f}'.format(epoch, self.train_acc, acc))
            if acc > self.best_valid_acc:
                if self.print_training:
                    print('Saving..')
                state = {
                    'net': self.net.state_dict(),
                    'best_valid_acc': acc,
                    'train_acc': self.train_acc,
                    'epoch': epoch,
                }
                torch.save(state, os.path.join(self.classification_path, 'ckpt.pth'))
                self.best_valid_acc = acc
                self.early_stop_count = 0
            else:
                self.early_stop_count += 1
                if self.print_training:
                    print('Early stop count: {}'.format(self.early_stop_count))

            if self.early_stop_count == self.early_stop_observation_period:
                if self.print_training:
                    print('Early stop count == {}; Terminate training\n'.format(self.early_stop_observation_period))
                self.train_flag = False

        elif type == 'test':
            return acc

    def train(self, trainset, validset=None):
        print('==> Start training {}'.format(self.classification_path))
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
                # self.scheduler.step()
            else:
                break

    def test(self, testset):
        print('==> Test {}'.format(self.classification_path))
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
        np.save(os.path.join(self.classification_path, 'acc.npy'), acc_dict)

    #####################
    # ---- For MIA ---- #
    #####################
    def extract_features(self, dataset_dict):
        print('==> Extract features')
        try:
            self.load()
        except FileNotFoundError:
            print('There is no pre-trained model; First, train the classifier.')
            sys.exit(1)
        self.net.eval()

        # print(self.net)

        features_dict = {}
        for dataset_type, dataset in dataset_dict.items():
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
            # loader = torch.utils.data.DataLoader(dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=2)
            logits = []
            prediction_scores = []
            labels = []
            print('====> Extract features from {} dataset'.format(dataset_type))
            # with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits_ = self.net(inputs)
                # prediction_scores_batch = outputs.cpu().numpy()
                # print(outputs[0])
                logits_batch = logits_.cpu().detach().numpy()
                prediction_scores_batch = torch.softmax(logits_, dim=1).cpu().detach().numpy()
                labels_batch = targets.cpu().detach().numpy()

                # x1, x2, x3 = self.net.extract_features(inputs)
                # print(x1.shape)
                # print(x2.shape)
                # print(x3.shape)
                # sys.exit(1)

                # loss = self.criterion(outputs, targets)
                # print(loss)
                # loss.backward()

                # print(self.net.fc2.weight)
                # print(self.net.fc2.weight.grad)
                # print(self.net.fc2.weight.grad.shape)

                # print('Prediction score:', prediction_scores_batch.shape)
                # print('Label:', labels_batch.shape)
                # print('Gradient:', self.net.fc2.weight.grad.shape)
                # sys.exit(1)

                if len(prediction_scores) == 0:
                    logits = logits_batch
                    prediction_scores = prediction_scores_batch
                    labels = labels_batch
                else:
                    logits = np.vstack((logits, logits_batch))
                    prediction_scores = np.vstack((prediction_scores, prediction_scores_batch))
                    labels = np.concatenate((labels, labels_batch))

            # print(prediction_scores.shape)
            # print(labels.shape)
            features_dict[dataset_type] = {
                'logits': logits,
                'preds': prediction_scores,
                'labels': labels,
            }
        np.save(os.path.join(self.classification_path, 'features.npy'), features_dict)
