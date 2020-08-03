import module
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
import os
import data
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
import torchvision.utils as vutils
from torch.utils.data import Subset, DataLoader


class Reconstructor(object):
    def __init__(self, args):
        self.reconstruction_path = args.reconstruction_path
        if not os.path.exists(self.reconstruction_path):
            os.makedirs(self.reconstruction_path)

        self.beta = args.beta
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.epochs = args.epochs
        self.early_stop = args.early_stop
        self.early_stop_observation_period = args.early_stop_observation_period
        self.use_scheduler = False
        self.print_training = args.print_training

        self.z_dim = args.z_dim
        self.disc_input_dim = int(self.z_dim / 2)
        self.class_idx = range(0, self.disc_input_dim)
        self.membership_idx = range(self.disc_input_dim, self.z_dim)

        self.nets = dict()

        if args.dataset in ['MNIST', 'Fashion-MNIST', 'CIFAR-10', 'SVHN']:
            if args.dataset in ['MNIST', 'Fashion-MNIST']:
                self.num_channels = 1
            elif args.dataset in ['CIFAR-10', 'SVHN']:
                self.num_channels = 3

            self.nets['encoder'] = module.VAEConvEncoder(self.z_dim, self.num_channels)
            self.nets['decoder'] = module.VAEConvDecoder(self.z_dim, self.num_channels)

        elif args.dataset in ['adult', 'location']:
            self.nets['encoder'] = module.VAEFCEncoder(args.encoder_input_dim, self.z_dim)
            self.nets['decoder'] = module.FCDecoder(args.encoder_input_dim, self.z_dim)

        self.discs = {
            'class_fz': module.ClassDiscriminator(self.z_dim, args.class_num),
            'class_cz': module.ClassDiscriminator(self.disc_input_dim, args.class_num),
            'class_mz': module.ClassDiscriminator(self.disc_input_dim, args.class_num),

            'membership_fz': module.MembershipDiscriminator(self.z_dim, 1),
            'membership_cz': module.MembershipDiscriminator(self.disc_input_dim, 1),
            'membership_mz': module.MembershipDiscriminator(self.disc_input_dim, 1),
        }

        self.recon_loss = self.get_loss_function()
        self.class_loss = nn.CrossEntropyLoss(reduction='sum')
        self.membership_loss = nn.BCEWithLogitsLoss(reduction='sum')

        # optimizer
        self.optimizer = dict()
        for net_type in self.nets:
            self.optimizer[net_type] = optim.Adam(self.nets[net_type].parameters(), lr=args.recon_lr, betas=(0.5, 0.999))
        self.discriminator_lr = args.disc_lr
        for disc_type in self.discs:
            self.optimizer[disc_type] = optim.Adam(self.discs[disc_type].parameters(), lr=self.discriminator_lr,
                                                   betas=(0.5, 0.999))

        self.weights = {
            'recon': args.recon_weight,
            'class_cz': args.class_cz_weight,
            'class_mz': args.class_mz_weight,
            'membership_cz': args.membership_cz_weight,
            'membership_mz': args.membership_mz_weight,
        }

        self.scheduler_enc = StepLR(self.optimizer['encoder'], step_size=50, gamma=0.1)
        self.scheduler_dec = StepLR(self.optimizer['decoder'], step_size=50, gamma=0.1)

        # to device
        self.device = torch.device("cuda:{}".format(args.gpu_id))
        for net_type in self.nets:
            self.nets[net_type] = self.nets[net_type].to(self.device)
        for disc_type in self.discs:
            self.discs[disc_type] = self.discs[disc_type].to(self.device)

        self.disentangle = (self.weights['class_cz'] + self.weights['class_mz']
                            + self.weights['membership_cz'] + self.weights['membership_mz'] > 0)

        self.start_epoch = 0
        self.best_valid_loss = float("inf")
        # self.train_loss = 0
        self.early_stop_count = 0

        self.acc_dict = {
            'class_fz': 0, 'class_cz': 0, 'class_mz': 0,
            'membership_fz': 0, 'membership_cz': 0, 'membership_mz': 0,
        }
        self.best_acc_dict = {}

        if 'cuda' in str(self.device):
            cudnn.benchmark = True

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
        for net_type in self.nets:
            self.nets[net_type].load_state_dict(checkpoint[net_type])
        for disc_type in self.discs:
            self.discs[disc_type].load_state_dict(checkpoint[disc_type])
        self.start_epoch = checkpoint['epoch']

    def train_epoch(self, train_ref_loader, epoch):
        for net_type in self.nets:
            self.nets[net_type].train()
        for disc_type in self.discs:
            self.discs[disc_type].train()

        total = 0

        losses = {
            'MSE': 0., 'KLD': 0.,
            'class_fz': 0., 'class_cz': 0., 'class_mz': 0.,
            'membership_fz': 0., 'membership_cz': 0., 'membership_mz': 0.,
        }

        corrects = {
            'MSE': 0., 'KLD': 0.,
            'class_fz': 0., 'class_cz': 0., 'class_mz': 0.,
            'membership_fz': 0., 'membership_cz': 0., 'membership_mz': 0.,
        }

        for batch_idx, (inputs, targets, inputs_ref, targets_ref) in enumerate(train_ref_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs_ref, targets_ref = inputs_ref.to(self.device), targets_ref.to(self.device)

            total += targets.size(0)

            # ---- Reconstruction (Encoder & Decoder) ---- #
            recon_loss, MSE, KLD = self.train_reconstructor(inputs)
            losses['MSE'] += MSE
            losses['KLD'] += KLD

            # ---- Class discriminators ---- #
            correct_class_fz, loss_class_fz = self.train_disc_class_fz(inputs, targets)
            correct_class_cz, loss_class_cz = self.train_disc_class_cz(inputs, targets)
            correct_class_mz, loss_class_mz = self.train_disc_class_mz(inputs, targets)

            corrects['class_fz'] += correct_class_fz
            corrects['class_cz'] += correct_class_cz
            corrects['class_mz'] += correct_class_mz
            losses['class_fz'] += loss_class_fz
            losses['class_cz'] += loss_class_cz
            losses['class_mz'] += loss_class_mz

            # ---- Membership discriminators ---- #
            correct_membership_fz, loss_membership_fz = self.train_disc_membership_fz(inputs, targets,
                                                                                      inputs_ref, targets_ref)
            correct_membership_cz, loss_membership_cz = self.train_disc_membership_cz(inputs, targets,
                                                                                      inputs_ref, targets_ref)
            correct_membership_mz, loss_membership_mz = self.train_disc_membership_mz(inputs, targets,
                                                                                      inputs_ref, targets_ref)
            corrects['membership_fz'] += correct_membership_fz
            corrects['membership_cz'] += correct_membership_cz
            corrects['membership_mz'] += correct_membership_mz
            losses['membership_fz'] += loss_membership_fz
            losses['membership_cz'] += loss_membership_cz
            losses['membership_mz'] += loss_membership_mz

            if self.disentangle:
                self.disentangle_z(inputs, targets)

        # todo : loop
        self.acc_dict['class_fz'] = corrects['class_fz'] / total
        self.acc_dict['class_cz'] = corrects['class_cz'] / total
        self.acc_dict['class_mz'] = corrects['class_mz'] / total

        self.acc_dict['membership_fz'] = corrects['membership_fz'] / (2 * total)
        self.acc_dict['membership_cz'] = corrects['membership_cz'] / (2 * total)
        self.acc_dict['membership_mz'] = corrects['membership_mz'] / (2 * total)

        if self.print_training:
            print(
                '\nEpoch: {:>3}, Acc) Class (fz, cz, mz) : {:.4f}, {:.4f}, {:.4f}, Membership (fz, cz, mz) : {:.4f}, {:.4f}, {:.4f}'.format(
                    epoch, self.acc_dict['class_fz'], self.acc_dict['class_cz'], self.acc_dict['class_mz'],
                    self.acc_dict['membership_fz'], self.acc_dict['membership_cz'], self.acc_dict['membership_mz'], ))

            for loss_type in losses:
                losses[loss_type] = losses[loss_type] / (batch_idx + 1)
            print(
                'Losses) MSE: {:.2f}, KLD: {:.2f}, Class (fz, cz, mz): {:.2f}, {:.2f}, {:.2f}, Membership (fz, cz, mz): {:.2f}, {:.2f}, {:.2f},'.format(
                    losses['MSE'], losses['KLD'], losses['class_fz'], losses['class_cz'], losses['class_mz'],
                    losses['membership_fz'], losses['membership_cz'], losses['membership_mz'], ))

    def train_reconstructor(self, inputs):
        self.optimizer['encoder'].zero_grad()
        self.optimizer['decoder'].zero_grad()
        mu, logvar = self.nets['encoder'](inputs)
        z = self.reparameterize(mu, logvar)
        recons = self.nets['decoder'](z)
        recon_loss, MSE, KLD = self.recon_loss(recons, inputs, mu, logvar)
        recon_loss = self.weights['recon'] * recon_loss
        recon_loss.backward()
        self.optimizer['encoder'].step()
        self.optimizer['decoder'].step()
        return recon_loss.item(), MSE.item(), KLD.item()

    def train_disc_class_fz(self, inputs, targets):
        self.optimizer['class_fz'].zero_grad()
        z = self.inference_z(inputs)
        pred = self.discs['class_fz'](z)
        class_loss_full = self.class_loss(pred, targets)
        class_loss_full.backward()
        self.optimizer['class_fz'].step()

        _, pred_class_from_full = pred.max(1)
        return pred_class_from_full.eq(targets).sum().item(), class_loss_full.item()

    def train_disc_class_cz(self, inputs, targets):
        self.optimizer['class_cz'].zero_grad()
        z = self.inference_z(inputs)
        content_z, _ = self.split_class_membership(z)
        pred = self.discs['class_cz'](content_z)
        class_loss_content = self.class_loss(pred, targets)
        class_loss_content.backward()
        self.optimizer['class_cz'].step()

        _, pred_class_from_content = pred.max(1)
        return pred_class_from_content.eq(targets).sum().item(), class_loss_content.item()

    def train_disc_class_mz(self, inputs, targets):
        self.optimizer['class_mz'].zero_grad()
        z = self.inference_z(inputs)
        _, style_z = self.split_class_membership(z)
        pred = self.discs['class_mz'](style_z)
        class_loss_style = self.class_loss(pred, targets)
        class_loss_style.backward()
        self.optimizer['class_mz'].step()

        _, pred_class_from_style = pred.max(1)
        return pred_class_from_style.eq(targets).sum().item(), class_loss_style.item()

    def train_disc_membership_fz(self, inputs, targets, inputs_ref, targets_ref):
        self.optimizer['membership_fz'].zero_grad()
        z = self.inference_z(inputs)
        pred = self.discs['membership_fz'](z)
        in_loss = self.membership_loss(pred, torch.ones_like(pred))

        z_ref = self.inference_z(inputs_ref)
        pred_ref = self.discs['membership_fz'](z_ref)
        out_loss = self.membership_loss(pred_ref, torch.zeros_like(pred_ref))

        membership_loss = in_loss + out_loss
        membership_loss.backward()
        self.optimizer['membership_fz'].step()

        pred = pred.cpu().detach().numpy().squeeze()
        pred_ref = pred_ref.cpu().detach().numpy().squeeze()
        pred_concat = np.concatenate((pred, pred_ref))
        inout_concat = np.concatenate((np.ones_like(pred), np.zeros_like(pred_ref)))

        return np.sum(inout_concat == np.round(pred_concat)), membership_loss.item()

    def train_disc_membership_cz(self, inputs, targets, inputs_ref, targets_ref):
        self.optimizer['membership_cz'].zero_grad()
        z = self.inference_z(inputs)
        content_z, _ = self.split_class_membership(z)
        pred = self.discs['membership_cz'](content_z)
        in_loss = self.membership_loss(pred, torch.ones_like(pred))

        z_ref = self.inference_z(inputs_ref)
        content_z_ref, _ = self.split_class_membership(z_ref)
        pred_ref = self.discs['membership_cz'](content_z_ref)
        out_loss = self.membership_loss(pred_ref, torch.zeros_like(pred_ref))

        membership_loss = in_loss + out_loss
        membership_loss.backward()
        self.optimizer['membership_cz'].step()

        pred = pred.cpu().detach().numpy().squeeze()
        pred_ref = pred_ref.cpu().detach().numpy().squeeze()
        pred_concat = np.concatenate((pred, pred_ref))
        inout_concat = np.concatenate((np.ones_like(pred), np.zeros_like(pred_ref)))

        return np.sum(inout_concat == np.round(pred_concat)), membership_loss.item()

    def train_disc_membership_mz(self, inputs, targets, inputs_ref, targets_ref):
        self.optimizer['membership_mz'].zero_grad()
        z = self.inference_z(inputs)
        _, style_z = self.split_class_membership(z)
        pred = self.discs['membership_mz'](style_z)
        in_loss = self.membership_loss(pred, torch.ones_like(pred))

        z_ref = self.inference_z(inputs_ref)
        _, style_z_ref = self.split_class_membership(z_ref)
        pred_ref = self.discs['membership_mz'](style_z_ref)
        out_loss = self.membership_loss(pred_ref, torch.zeros_like(pred_ref))

        membership_loss = in_loss + out_loss
        membership_loss.backward()
        self.optimizer['membership_mz'].step()

        pred = pred.cpu().detach().numpy().squeeze()
        pred_ref = pred_ref.cpu().detach().numpy().squeeze()
        pred_concat = np.concatenate((pred, pred_ref))
        inout_concat = np.concatenate((np.ones_like(pred), np.zeros_like(pred_ref)))

        return np.sum(inout_concat == np.round(pred_concat)), membership_loss.item()

    def disentangle_z(self, inputs, targets):
        self.optimizer['encoder'].zero_grad()
        loss = 0

        z = self.inference_z(inputs)
        cz, mz = self.split_class_membership(z)

        if self.weights['class_cz'] != 0:
            pred = self.discs['class_cz'](cz)
            loss += self.weights['class_mz'] * self.class_loss(pred, targets)

        if self.weights['class_mz'] != 0:
            pred = self.discs['class_mz'](mz)
            loss += -self.weights['class_mz'] * self.class_loss(pred, targets)

        if self.weights['membership_cz'] != 0:
            pred = self.discs['membership_cz'](cz)
            loss += - self.weights['membership_cz'] * self.membership_loss(pred, torch.ones_like(pred))

        if self.weights['membership_mz'] != 0:
            pred = self.discs['membership_mz'](mz)
            loss += self.weights['membership_mz'] * self.membership_loss(pred, torch.ones_like(pred))

        loss.backward()
        self.optimizer['encoder'].step()

    def inference(self, loader, epoch, type='valid'):
        for net_type in self.nets:
            self.nets[net_type].eval()
        for disc_type in self.discs:
            self.discs[disc_type].eval()

        loss = 0
        correct_class_from_content = 0
        correct_class_from_style = 1

        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                mu, logvar = self.nets['encoder'](inputs)
                z = self.reparameterize(mu, logvar)

                recons = self.nets['decoder'](z)
                recon_loss, MSE, KLD = self.recon_loss(recons, inputs, mu, logvar)
                loss += recon_loss.item()

                total += targets.size(0)
                content_z, style_z = self.split_class_membership(z)

                # -- Class (valid) -- #
                _, pred_class_from_content = self.discs['class_cz'](content_z).max(1)
                _, pred_class_from_style = self.discs['class_mz'](style_z).max(1)

                correct_class_from_content += pred_class_from_content.eq(targets).sum().item()
                correct_class_from_style += pred_class_from_style.eq(targets).sum().item()

        if type == 'valid':
            if loss < self.best_valid_loss:
                state = {
                    'best_valid_loss': loss,
                    'epoch': epoch,
                }

                for net_type in self.nets:
                    state[net_type] = self.nets[net_type].state_dict()
                for disc_type in self.discs:
                    state[disc_type] = self.discs[disc_type].state_dict()

                torch.save(state, os.path.join(self.reconstruction_path, 'ckpt.pth'))
                self.best_valid_loss = loss
                self.early_stop_count = 0
                self.best_acc_dict = self.acc_dict

                np.save(os.path.join(self.reconstruction_path, 'acc.npy'), self.best_acc_dict)
                vutils.save_image(recons, os.path.join(self.reconstruction_path, '{}.png'.format(epoch)), nrow=10)

            else:
                self.early_stop_count += 1
                if self.print_training:
                    print('Early stop count: {}'.format(self.early_stop_count))

            if self.early_stop_count == self.early_stop_observation_period:
                print(self.best_acc_dict)
                if self.print_training:
                    print('Early stop count == {}; Terminate training\n'.format(self.early_stop_observation_period))
                self.train_flag = False

    def train(self, train_set, valid_set=None, ref_set=None):
        print('==> Start training {}'.format(self.reconstruction_path))
        self.train_flag = True
        if self.early_stop:
            valid_loader = DataLoader(valid_set, batch_size=self.train_batch_size, shuffle=True, num_workers=2)
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            permutated_idx = np.random.permutation(ref_set.__len__())
            ref_set = Subset(ref_set, permutated_idx)
            train_ref_set = data.DoubleDataset(train_set, ref_set)
            train_ref_loader = DataLoader(train_ref_set, batch_size=self.train_batch_size, shuffle=True, num_workers=2)
            if self.train_flag:
                self.train_epoch(train_ref_loader, epoch)
                if self.use_scheduler:
                    self.scheduler_enc.step()
                    self.scheduler_dec.step()
                if self.early_stop:
                    self.inference(valid_loader, epoch, type='valid')
            else:
                break

    def reconstruct(self, dataset_dict, reconstruction_type_list):
        print('==> Reconstruct datasets')
        try:
            self.load()
        except FileNotFoundError:
            print('There is no pre-trained model; First, train a reconstructor.')
            sys.exit(1)
        self.nets['encoder'].eval()
        self.nets['decoder'].eval()

        mse_list = []
        recon_dict = dict()

        for reconstruction_type in reconstruction_type_list:
            recon_datasets_dict = {}
            print(reconstruction_type)
            for dataset_type, dataset in dataset_dict.items():
                loader = DataLoader(dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=2)
                raws = []
                recons = []
                labels = []
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(loader):
                        inputs = inputs.to(self.device)
                        mu, logvar = self.nets['encoder'](inputs)

                        z = torch.zeros_like(mu).to(self.device)

                        mu_content, mu_style = self.split_class_membership(mu)
                        logvar_content, logvar_style = self.split_class_membership(logvar)

                        # print('mu_content')
                        # print(mu_content)
                        # print('mu_style')
                        # print(mu_style)
                        # print('std_content')
                        # # print('logvar_content')
                        # std_content = torch.exp(0.5*logvar_content)
                        # print(std_content)
                        # print('std_style')
                        # # print('logvar_style')
                        # std_style = torch.exp(0.5*logvar_style)
                        # print(std_style)
                        #
                        # print('Content')
                        # print('mu', torch.min(torch.abs(mu_content)), torch.max(torch.abs(mu_content)))
                        # print('std', torch.min(torch.abs(std_content)), torch.max(torch.abs(std_content)))
                        #
                        # print('Style')
                        # print('mu', torch.min(torch.abs(mu_style)), torch.max(torch.abs(mu_style)))
                        # print('std', torch.min(torch.abs(std_style)), torch.max(torch.abs(std_style)))
                        # sys.exit(1)

                        if reconstruction_type == 'cb_mb':
                            z[:, self.class_idx] = mu_content
                            z[:, self.membership_idx] = mu_style

                        elif reconstruction_type == 'cb_ms':
                            z[:, self.class_idx] = mu_content
                            z[:, self.membership_idx] = self.reparameterize(mu_style, logvar_style)

                        elif reconstruction_type == 'cs_mb':
                            z[:, self.class_idx] = self.reparameterize(mu_content, logvar_content)
                            z[:, self.membership_idx] = mu_style

                        elif reconstruction_type == 'cs_ms':
                            z[:, self.class_idx] = self.reparameterize(mu_content, logvar_content)
                            z[:, self.membership_idx] = self.reparameterize(mu_style, logvar_style)

                        elif reconstruction_type == 'cz_mb':
                            z[:, self.class_idx] = torch.zeros_like(mu_content).to(self.device)
                            z[:, self.membership_idx] = mu_style

                        elif reconstruction_type == 'cb_mz':
                            z[:, self.class_idx] = mu_content
                            z[:, self.membership_idx] = torch.zeros_like(mu_style).to(self.device)

                        elif reconstruction_type == 'cb_mn':
                            z[:, self.class_idx] = mu_content
                            z[:, self.membership_idx] = torch.randn_like(mu_style).to(self.device)

                        # elif reconstruction_type == 'uniform_style':
                        #     z[:, self.class_idx] = mu_content
                        #     z[:, self.membership_idx] = torch.rand_like(mu_style).to(self.device)
                        #
                        # elif reconstruction_type == 'normal_style':
                        #     z[:, self.class_idx] = mu_content
                        #     z[:, self.membership_idx] = torch.randn_like(mu_style).to(self.device)

                        recons_batch = self.nets['decoder'](z).cpu()
                        labels_batch = targets

                        if len(recons) == 0:
                            raws = inputs.cpu()
                            recons = recons_batch
                            labels = labels_batch

                            if dataset_type == 'train':
                                vutils.save_image(recons, os.path.join(self.reconstruction_path,
                                                                       '{}.png'.format(reconstruction_type)), nrow=10)
                                recon_dict[reconstruction_type] = recons

                        else:
                            raws = torch.cat((raws, inputs.cpu()), axis=0)
                            recons = torch.cat((recons, recons_batch), axis=0)
                            labels = torch.cat((labels, labels_batch), axis=0)

                recon_datasets_dict[dataset_type] = {
                    'recons': recons,
                    'labels': labels,
                }

                mse_list.append(F.mse_loss(recons, raws).item())

            # todo : refactor dict to CustomDataset
            torch.save(recon_datasets_dict,
                       os.path.join(self.reconstruction_path, 'recon_{}.pt'.format(reconstruction_type)))

        np.save(os.path.join(self.reconstruction_path, 'mse.npy'), mse_list)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + std * eps

    def inference_z(self, z):
        mu, logvar = self.nets['encoder'](z)
        return self.reparameterize(mu, logvar)

    def split_class_membership(self, z):
        class_z = z[:, self.class_idx]
        membership_z = z[:, self.membership_idx]

        return class_z, membership_z

    def get_loss_function(self):
        def loss_function(recon_x, x, mu, logvar):
            MSE = F.mse_loss(recon_x, x, reduction='sum')
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).sum()
            # print(MSE)
            # print(KLD)
            return MSE + self.beta * KLD, MSE, KLD

        return loss_function
