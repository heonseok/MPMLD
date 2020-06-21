import module
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
import torchvision.utils as vutils
import os


class Disentangler(object):
    def __init__(self, args):
        self.train_batch_size = args.train_batch_size
        self.epochs = args.epochs
        self.start_epoch = 0
        self.z_dim = args.z_dim
        self.num_channels = 3
        self.image_size = 64
        self.disentanglement_path = args.disentanglement_path

        if not os.path.exists(self.disentanglement_path):
            os.makedirs(self.disentanglement_path)

        self.encoder = module.ConvEncoder(self.z_dim, self.num_channels, self.image_size)
        self.decoder = module.ConvDecoder(self.z_dim, self.num_channels, self.image_size)
        # self.discriminator = module.SimpleNet(10)

        self.optimizer_enc = optim.Adam(self.encoder.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.optimizer_dec = optim.Adam(self.decoder.parameters(), lr=args.lr, betas=(0.5, 0.999))

        self.criterion = nn.MSELoss()

        self.device = torch.device("cuda:{}".format(args.gpu_id))
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        # print(self.encoder)
        # print(self.decoder)

        if self.device == 'cuda':
            cudnn.benchmark = True

    def train_epoch(self, trainloader, epoch):
        self.encoder.train()
        self.decoder.train()
        train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
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

        if epoch % 10 == 0:
            print('saving the output')
            vutils.save_image(inputs, os.path.join(self.disentanglement_path, 'real_samples.png'), normalize=True)
            # fake = netG(fixed_noise)
            vutils.save_image(outputs.detach(), os.path.join(self.disentanglement_path, 'recon_%03d.png' % (epoch)),
                              normalize=True)

    def train(self, trainset):
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.train_batch_size, shuffle=True,
                                                  num_workers=2)
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            self.train_epoch(trainloader, epoch)
