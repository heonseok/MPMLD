import module
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
import torchvision.utils as vutils
import os
import numpy as np


class Reconstructor(object):
    def __init__(self, args):
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.epochs = args.epochs
        self.early_stop = args.early_stop
        self.early_stop_observation_period = args.early_stop_observation_period

        self.z_dim = args.z_dim
        self.disc_input_dim = int(self.z_dim / 2)
        self.num_channels = 3
        self.image_size = 64

        self.disentanglement_type = args.disentanglement_type
        self.reconstruction_path = args.reconstruction_path

        if not os.path.exists(self.reconstruction_path):
            os.makedirs(self.reconstruction_path)

        self.encoder = module.ConvEncoder(self.z_dim, self.num_channels)
        self.decoder = module.ConvDecoder(self.z_dim, self.num_channels)
        self.classifier = module.SimpleClassifier(self.disc_input_dim, 10)

        self.optimizer_enc = optim.Adam(self.encoder.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.optimizer_dec = optim.Adam(self.decoder.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.optimizer_class = optim.Adam(self.classifier.parameters(), lr=args.lr, betas=(0.5, 0.999))

        # self.criterion = nn.BCEWithLogitsLoss()
        # self.criterion = nn.BCELoss()
        self.recon_loss = nn.MSELoss()
        self.class_loss = nn.CrossEntropyLoss()

        self.device = torch.device("cuda:{}".format(args.gpu_id))
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.classifier = self.classifier.to(self.device)

        self.start_epoch = 0
        self.best_valid_loss = float("inf")
        self.train_loss = 0
        self.train_acc = 0
        self.early_stop_count = 0

        # print(self.encoder)
        # print(self.decoder)

        if self.device == 'cuda':
            cudnn.benchmark = True

        self.fixed_inputs = None

        if args.resume:
            print('==> Resuming from checkpoint..')
            try:
                self.load()
            except FileNotFoundError:
                print('There is no pre-trained model; Train model from scratch')

    #########################
    # -- Base operations -- #
    #########################
    def load(self):
        print('====> Loading checkpoint {}'.format(self.reconstruction_path))
        checkpoint = torch.load(os.path.join(self.reconstruction_path, 'ckpt.pth'))
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.classifier.load_state_dict(checkpoint['classifier'])
        self.start_epoch = checkpoint['epoch']

    def train_epoch(self, trainloader, epoch):
        self.encoder.train()
        self.decoder.train()
        self.classifier.train()
        recon_train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # if batch_idx == 0 and epoch == 0:
            # if batch_idx == 0:
            #     self.fixed_inputs = inputs
            #     vutils.save_image(self.fixed_inputs, os.path.join(self.disentanglement_path, 'real_samples.png'),
            #                       normalize=True, nrow=10)

            # -------------------------------------------------------
            # ---- Reconstruction ---- #
            self.optimizer_enc.zero_grad()
            self.optimizer_dec.zero_grad()

            z = self.encoder(inputs)
            recons = self.decoder(z)
            recon_loss = self.recon_loss(recons, inputs)
            recon_loss.backward()

            self.optimizer_enc.step()
            self.optimizer_dec.step()

            # ---- Disentanglement ---- #
            if self.disentanglement_type == 'type1':
                self.optimizer_enc.zero_grad()
                z = self.encoder(inputs)
                pred_label = self.classifier(z[:, self.disc_input_dim:])
                class_loss = -self.class_loss(pred_label, targets)
                class_loss.backward()
                self.optimizer_enc.step()

                self.optimizer_class.zero_grad()
                z = self.encoder(inputs)
                pred_label = self.classifier(z[:, self.disc_input_dim:])
                class_loss = self.class_loss(pred_label, targets)
                class_loss.backward()
                self.optimizer_class.step()

            elif self.disentanglement_type == 'type2':
                self.optimizer_enc.zero_grad()
                z = self.encoder(inputs)
                pred_label = self.classifier(z[:, 0:self.disc_input_dim])
                class_loss = self.class_loss(pred_label, targets)
                class_loss.backward()
                self.optimizer_enc.step()

                self.optimizer_class.zero_grad()
                z = self.encoder(inputs)
                pred_label = self.classifier(z[:, 0:self.disc_input_dim])
                class_loss = self.class_loss(pred_label, targets)
                class_loss.backward()
                self.optimizer_class.step()

            recon_train_loss += recon_loss.item()
            if self.disentanglement_type != 'base':
                _, predicted = pred_label.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        self.train_loss = recon_train_loss
        if self.disentanglement_type != 'base':
            self.train_acc = correct / total
        # if self.disentanglement_type != 'base':
        #     # print(epoch, recon_train_loss, correct / total)
        #     print(
        #         'Epoch: {:>3}, Train Loss: {:.4f}, Train Acc: {:.4f}'.format(epoch, recon_train_loss, correct / total))
        #     # print('Epoch: {:>3}, Train Acc: {:.4f}, Valid Acc: {:.4f}'.format(epoch, self.train_acc, acc))
        # else:
        #     # print(epoch, recon_train_loss)
        #     print('Epoch: {:>3}, Train Loss: {:.4f}'.format(epoch, recon_train_loss))

        if (epoch + 1) % 50 == 0:
            print('saving the output')
            vutils.save_image(inputs, os.path.join(self.reconstruction_path, 'real_samples.png'),
                              normalize=True, nrow=10)
            vutils.save_image(recons,
                              os.path.join(self.reconstruction_path, 'recon_%03d.png' % (epoch + 1)), normalize=True,
                              nrow=10)

    def inference(self, loader, epoch, type='valid'):
        self.encoder.eval()
        self.decoder.eval()
        loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                z = self.encoder(inputs)
                recons = self.decoder(z)
                recon_loss = self.recon_loss(recons, inputs)
                loss += recon_loss.item()

                if self.disentanglement_type == 'type1':
                    pred_label = self.classifier(z[:, self.disc_input_dim:])
                elif self.disentanglement_type == 'type2':
                    pred_label = self.classifier(z[:, 0:self.disc_input_dim])

                if self.disentanglement_type != 'base':
                    _, predicted = pred_label.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

        if type == 'valid':
            if self.disentanglement_type == 'base':
                print('Epoch: {:>3}, Train Loss: {:.4f}, Valid Loss: {:.4f}'.format(epoch, self.train_loss, loss))
            else:
                print(
                    'Epoch: {:>3}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Train Acc : {:.4f}, Valid Acc : {:.4f}'.format(
                        epoch, self.train_loss, loss, self.train_acc, correct / total))

            if loss < self.best_valid_loss:
                print('Saving..')
                state = {
                    'encoder': self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                    'classifier': self.classifier.state_dict(),
                    'best_valid_loss': loss,
                    'train_loss': self.train_loss,
                    'epoch': epoch,
                }
                torch.save(state, os.path.join(self.reconstruction_path, 'ckpt.pth'))
                self.best_valid_loss = loss
                self.early_stop_count = 0
            else:
                self.early_stop_count += 1
                print('Early stop count: {}'.format(self.early_stop_count))

            if self.early_stop_count == self.early_stop_observation_period:
                print('Early stop count == {}; Terminate training\n'.format(self.early_stop_observation_period))
                self.train_flag = False

    def train(self, trainset, validset=None):
        print('==> Start training {}'.format(self.reconstruction_path))
        self.train_flag = True
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.train_batch_size, shuffle=True,
                                                  num_workers=2)
        if self.early_stop:
            validloader = torch.utils.data.DataLoader(validset, batch_size=self.train_batch_size, shuffle=True,
                                                      num_workers=2)
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            if self.train_flag:
                self.train_epoch(trainloader, epoch)
                if self.early_stop:
                    self.inference(validloader, epoch, type='valid')
            else:
                break

    def reconstruct(self, dataset_dict, reconstruction_type):
        print('==> Reconstruct datasets')
        try:
            self.load()
        except FileNotFoundError:
            print('There is no pre-trained model; First, train the disentangler.')
            sys.exit(1)
        self.encoder.eval()
        self.decoder.eval()

        recon_datasets_dict = {}
        for dataset_type, dataset in dataset_dict.items():
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=2)
            recons = []
            labels = []
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(loader):
                    inputs = inputs.to(self.device)
                    z = self.encoder(inputs)
                    if reconstruction_type == 'partial_z':
                        paritial_z = z[:, 0:self.disc_input_dim]
                        z = torch.cat((paritial_z, torch.zeros_like(paritial_z)), axis=1)
                    recons_batch = self.decoder(z).cpu()
                    labels_batch = targets
                    if len(recons) == 0:
                        recons = recons_batch
                        labels = labels_batch

                        # save images
                        vutils.save_image(recons, os.path.join(self.reconstruction_path,
                                                               'recon_{}_{}.png'.format(dataset_type,
                                                                                        reconstruction_type)),
                                          normalize=True, nrow=10)

                    else:
                        recons = torch.cat((recons, recons_batch), axis=0)
                        labels = torch.cat((labels, labels_batch), axis=0)

            recon_datasets_dict[dataset_type] = {
                'recons': recons,
                'labels': labels,
            }

        # todo : refactor dict to CustomDataset
        torch.save(recon_datasets_dict,
                   os.path.join(self.reconstruction_path, 'recon_{}.pt'.format(reconstruction_type)))
