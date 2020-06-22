import module
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
import torchvision.utils as vutils
import os
import numpy as np


class Disentangler(object):
    def __init__(self, args):
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.epochs = args.epochs
        self.start_epoch = 0
        self.z_dim = args.z_dim
        self.num_channels = 3
        self.image_size = 64
        self.disentanglement_path = args.disentanglement_path

        if not os.path.exists(self.disentanglement_path):
            os.makedirs(self.disentanglement_path)

        self.encoder = module.ConvEncoder(self.z_dim, self.num_channels)
        self.decoder = module.ConvDecoder(self.z_dim, self.num_channels)
        # self.discriminator = module.SimpleNet(10)

        self.optimizer_enc = optim.Adam(self.encoder.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.optimizer_dec = optim.Adam(self.decoder.parameters(), lr=args.lr, betas=(0.5, 0.999))

        # self.criterion = nn.BCEWithLogitsLoss()
        # self.criterion = nn.BCELoss()
        self.criterion = nn.MSELoss()

        self.device = torch.device("cuda:{}".format(args.gpu_id))
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

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
        print('====> Loading checkpoint {}'.format(self.disentanglement_path))
        checkpoint = torch.load(os.path.join(self.disentanglement_path, 'ckpt.pth'))
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.start_epoch = checkpoint['epoch']

    def train_epoch(self, trainloader, epoch):
        self.encoder.train()
        self.decoder.train()
        train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # if batch_idx == 0 and epoch == 0:
            # if batch_idx == 0:
            #     self.fixed_inputs = inputs
            #     vutils.save_image(self.fixed_inputs, os.path.join(self.disentanglement_path, 'real_samples.png'),
            #                       normalize=True, nrow=10)
            self.optimizer_enc.zero_grad()
            self.optimizer_dec.zero_grad()

            z = self.encoder(inputs)
            outputs = self.decoder(z)

            loss = self.criterion(outputs, inputs)
            loss.backward()

            self.optimizer_enc.step()
            self.optimizer_dec.step()

            train_loss += loss.item()
        print(epoch, train_loss)

        if (epoch + 1) % 50 == 0:
            print('saving the output')
            vutils.save_image(inputs, os.path.join(self.disentanglement_path, 'real_samples.png'),
                              normalize=True, nrow=10)
            vutils.save_image(outputs,
                              os.path.join(self.disentanglement_path, 'recon_%03d.png' % (epoch + 1)), normalize=True,
                              nrow=10)
            # try:
            #     outputs_from_fixed_inputs = self.decoder(self.encoder(self.fixed_inputs)).detach()
            # except TypeError:
            #     return
            # vutils.save_image(outputs_from_fixed_inputs,
            #                   os.path.join(self.disentanglement_path, 'recon_%03d.png' % (epoch + 1)), normalize=True,
            #                   nrow=10)

        state = {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'epoch': epoch,
        }
        torch.save(state, os.path.join(self.disentanglement_path, 'ckpt.pth'))

    def train(self, trainset):
        print('==> Start training {}'.format(self.disentanglement_path))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.train_batch_size, shuffle=True,
                                                  num_workers=2)
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            self.train_epoch(trainloader, epoch)

    def reconstruct(self, dataset_dict):
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
                    recons_batch = self.decoder(self.encoder(inputs)).cpu()
                    labels_batch = targets
                    # recons_batch = self.decoder(self.encoder(inputs)).cpu().numpy()
                    # labels_batch = targets.numpy()
                    if len(recons) == 0:
                        recons = recons_batch
                        labels = labels_batch
                    else:
                        recons = torch.cat((recons, recons_batch), axis=0)
                        labels = torch.cat((labels, labels_batch), axis=0)
                        # recons = np.vstack((recons, recons_batch))
                        # labels = np.concatenate((labels, labels_batch))

            recon_datasets_dict[dataset_type] = {
                'recons': recons,
                'labels': labels,
            }
        # np.save(os.path.join(self.disentanglement_path, 'recon_datasets.npy'), recon_datasets_dict)
        torch.save(recon_datasets_dict, os.path.join(self.disentanglement_path, 'recon_datasets.pt'))
