from models import *
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os
from utils import progress_bar


class Classifier(object):
    def __init__(self, args):
        self.train_batch_size = args.train_batch_size
        self.valid_batch_size = args.valid_batch_size
        self.test_batch_size = args.test_batch_size
        self.epochs = args.epochs

        self.model_name = os.path.join('{}_setsize{}'.format(args.model_type, args.setsize),
                                       'repeat{}'.format(args.repeat_idx))
        self.model_path = os.path.join(args.base_path, 'classifier', self.model_name)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # Model
        print('==> Building {}'.format(self.model_name))
        if 'VGG' in args.model_type:
            print('VGG(\'' + args.model_type + '\')')
            net = eval('VGG(\'' + args.model_type + '\')')
        else:
            net = eval(args.model_type + '()')

        self.start_epoch = 0
        self.best_acc = 0

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = net.to(self.device)
        if self.device == 'cuda':
            cudnn.benchmark = True

        if args.resume:
            print('==> Resuming from checkpoint..')
            try:
                checkpoint = torch.load(os.path.join(self.model_path, 'ckpt.pth'))
                net.load_state_dict(checkpoint['net'])
                self.best_acc = checkpoint['acc']
                self.start_epoch = checkpoint['epoch']
            except FileNotFoundError:
                print('There is no pre-trained model; Train model from scratch')

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

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
            if acc > self.best_acc:
                print('Saving..')
                state = {
                    'net': self.net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
                torch.save(state, os.path.join(self.model_path, 'ckpt.pth'))
                self.best_acc = acc

        elif type == 'test':
            print('Test acc : {}'.format(acc))

    def train(self, trainset, validset):
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.train_batch_size, shuffle=True,
                                                  num_workers=2)
        validloader = torch.utils.data.DataLoader(validset, batch_size=self.valid_batch_size, shuffle=True,
                                                  num_workers=2)
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            self.train_epoch(trainloader, epoch)
            self.inference(validloader, epoch, type='valid')

    def test(self, testset):
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.test_batch_size, shuffle=False, num_workers=2)
        self.inference(testloader, epoch=-1, type='test')
