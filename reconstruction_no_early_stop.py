import module
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
import os
import data
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.nn import functional as F
import torchvision.utils as vutils
from torch.utils.data import Subset, DataLoader
from torch.autograd import Variable
import torch.autograd as autograd
import torchvision.transforms as transforms


# Distinct Encoders + Distinct Discriminators
class DistinctReconstructor(object):
    def __init__(self, args):
        self.reconstruction_path = args.reconstruction_path
        if not os.path.exists(self.reconstruction_path):
            os.makedirs(self.reconstruction_path)

        self.beta = args.beta
        self.train_batch_size = args.recon_train_batch_size
        self.test_batch_size = args.test_batch_size
        self.epochs = args.epochs
        # self.early_stop = args.early_stop_recon
        self.early_stop = False
        self.early_stop_observation_period = args.early_stop_observation_period
        self.print_training = args.print_training
        self.class_num = args.class_num
        self.disentangle_with_reparameterization = args.disentangle_with_reparameterization
        self.share_encoder = args.share_encoder
        self.train_flag = False
        self.resume = args.resume
        self.adversarial_loss_mode = args.adversarial_loss_mode
        self.gradient_penalty_weight = args.gradient_penalty_weight
        self.reduction = 'sum'
        # self.reduction = 'mean'

        # self.transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.disentanglement_start_epoch = 0
        self.save_step_size = 100 
        self.scheduler_step_size = 100 

        self.small_recon_weight = args.small_recon_weight
        self.z_dim = args.z_dim

        # class (pos/neg) mem (pos/neg)
        self.encoder_name_list = ['pn', 'pp', 'np', 'nn']

        self.z_idx = dict()
        self.base_z_dim = int(self.z_dim / 4)
        for encoder_idx, encoder_name in enumerate(self.encoder_name_list):
            self.z_idx[encoder_name] = range(encoder_idx * self.base_z_dim, (encoder_idx + 1) * self.base_z_dim)

        # Enc/Dec
        self.encoders = dict()
        if args.dataset in ['MNIST', 'Fashion-MNIST', 'CIFAR-10', 'SVHN']:
            if args.dataset in ['MNIST', 'Fashion-MNIST']:
                self.num_channels = 1
            elif args.dataset in ['CIFAR-10', 'SVHN']:
                self.num_channels = 3

            for encoder_name in self.encoder_name_list:
                self.encoders[encoder_name] = module.VAEConvEncoder(self.z_dim, self.num_channels)

            self.decoder = module.VAEConvDecoder(self.z_dim, self.num_channels)

        # Discriminators
        self.class_discs = dict()
        self.membership_discs = dict()
        for encoder_name in self.encoder_name_list:
            # self.class_discs[encoder_name] = module.ClassDiscriminator(self.base_z_dim, args.class_num)
            # self.membership_discs[encoder_name] = module.MembershipDiscriminator(self.base_z_dim + args.class_num, 1)
            self.class_discs[encoder_name] = module.ClassDiscriminatorImproved(self.base_z_dim, args.class_num)
            self.membership_discs[encoder_name] = module.MembershipDiscriminatorImproved(self.base_z_dim, args.class_num)
        self.rf_disc = module.Discriminator()

        # Optimizer
        self.encoders_opt = dict()
        self.class_discs_opt = dict()
        self.membership_discs_opt = dict()
        for encoder_name in self.encoder_name_list:
            self.encoders_opt[encoder_name] = optim.Adam(self.encoders[encoder_name].parameters(), lr=args.recon_lr, betas=(0.5, 0.999))
            self.class_discs_opt[encoder_name] = optim.Adam(self.class_discs[encoder_name].parameters(), lr=args.recon_lr, betas=(0.5, 0.999))
            self.membership_discs_opt[encoder_name] = optim.Adam(self.membership_discs[encoder_name].parameters(), lr=args.recon_lr, betas=(0.5, 0.999))

        self.decoder_opt = optim.Adam(self.decoder.parameters(), lr=args.recon_lr, betas=(0.5, 0.999))
        self.rf_disc_opt = optim.Adam(self.rf_disc.parameters(), lr=args.recon_lr, betas=(0.5, 0.999))

        # Scheduler 
        self.encoders_opt_scheduler = dict()
        self.class_discs_opt_scheduler = dict()
        self.membership_discs_opt_scheduler = dict()

        if self.early_stop:
            for encoder_name in self.encoder_name_list:
                self.encoders_opt_scheduler[encoder_name] = ReduceLROnPlateau(self.encoders_opt[encoder_name], 'min', patience=self.early_stop_observation_period, threshold=0)
                self.class_discs_opt_scheduler[encoder_name] = ReduceLROnPlateau(self.class_discs_opt[encoder_name], 'min', patience=self.early_stop_observation_period, threshold=0)
                self.membership_discs_opt_scheduler[encoder_name] = ReduceLROnPlateau(self.membership_discs_opt[encoder_name], 'min', patience=self.early_stop_observation_period, threshold=0)

            self.decoder_opt_scheduler = ReduceLROnPlateau(self.decoder_opt, 'min', patience=self.early_stop_observation_period, threshold=0)
            self.rf_disc_opt_scheduler = ReduceLROnPlateau(self.rf_disc_opt, 'min', patience=self.early_stop_observation_period, threshold=0)
        else:
            for encoder_name in self.encoder_name_list:
                self.encoders_opt_scheduler[encoder_name] = StepLR(self.encoders_opt[encoder_name], self.scheduler_step_size)
                self.class_discs_opt_scheduler[encoder_name] = StepLR(self.class_discs_opt[encoder_name], self.scheduler_step_size)
                self.membership_discs_opt_scheduler[encoder_name] = StepLR(self.membership_discs_opt[encoder_name], self.scheduler_step_size)

            self.decoder_opt_scheduler = StepLR(self.decoder_opt, self.scheduler_step_size)
            self.rf_disc_opt_scheduler = StepLR(self.rf_disc_opt, self.scheduler_step_size)


        # Loss
        self.vae_loss = self.get_loss_function()
        self.class_loss = nn.CrossEntropyLoss(reduction=self.reduction)
        self.membership_loss = nn.BCEWithLogitsLoss(reduction=self.reduction)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=self.reduction)

        self.weights = {
            'recon': args.recon_weight,
            'real_fake': args.real_fake_weight,
            'class_pos': args.class_pos_weight,
            'class_neg': args.class_neg_weight,
            'membership_pos': args.membership_pos_weight,
            'membership_neg': args.membership_neg_weight,
        }

        # To device
        self.device = torch.device("cuda:{}".format(args.gpu_id))
        for encoder_name in self.encoder_name_list:
            self.encoders[encoder_name] = self.encoders[encoder_name].to(self.device)
            self.class_discs[encoder_name] = self.class_discs[encoder_name].to(self.device)
            self.membership_discs[encoder_name] = self.membership_discs[encoder_name].to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.rf_disc = self.rf_disc.to(self.device)

        self.start_epoch = 0
        self.best_valid_loss = float("inf")
        self.early_stop_count = 0
        self.early_stop_count_total = 0
        self.EARLY_STOP_COUNT_TOTAL_MAX = 4 

        self.class_acc_dict = {
            'pn': 0., 'pp': 0., 'np': 0., 'nn': 0.,
        }
        self.membership_acc_dict = {
            'pn': 0., 'pp': 0., 'np': 0., 'nn': 0.,
        }
        # self.class_loss_dict = {
        #     'pn': 0., 'pp': 0., 'np': 0., 'nn': 0.,
        # }
        # self.membership_loss_dict = {
        #     'pn': 0., 'pp': 0., 'np': 0., 'nn': 0.,
        # }
        self.best_class_acc_dict = {}
        self.best_membership_acc_dict = {}

        if 'cuda' in str(self.device):
            cudnn.benchmark = True

    #########################
    # -- Base operations -- #
    #########################

    def load(self, epoch=-1):
        # print('Epoch : {:03d}'.format(epoch))

        if not self.early_stop:
            checkpoint = torch.load(os.path.join(self.reconstruction_path, 'ckpt{:03d}.pth'.format(epoch+1)), map_location=self.device)
        else:
            checkpoint = torch.load(os.path.join(self.reconstruction_path, 'ckpt.pth'), map_location=self.device)

        for encoder_name in self.encoder_name_list:
            self.encoders[encoder_name].load_state_dict(checkpoint['enc_' + encoder_name])
            self.class_discs[encoder_name].load_state_dict(checkpoint['class_disc_' + encoder_name])
            self.membership_discs[encoder_name].load_state_dict(checkpoint['membership_disc_' + encoder_name])
            self.decoder.load_state_dict(checkpoint['dec'])

        self.start_epoch = checkpoint['epoch']
        print('====> Load checkpoint {} (Epoch: {})'.format(self.reconstruction_path, self.start_epoch+1))

    def train_epoch(self, train_ref_loader, epoch):
        for encoder_name in self.encoder_name_list:
            self.encoders[encoder_name].train()
            self.class_discs[encoder_name].train()
            self.membership_discs[encoder_name].train()

        total = 0

        losses = {
            'MSE': 0, 'KLD': 0, 'RF': 0,
            'class': {'pn': 0, 'pp': 0, 'np': 0, 'nn': 0},
            'membership': {'pn': 0, 'pp': 0, 'np': 0, 'nn': 0},
        }

        corrects = {
            'class': {'pn': 0, 'pp': 0, 'np': 0, 'nn': 0},
            'membership': {'pn': 0, 'pp': 0, 'np': 0, 'nn': 0},
        }

        for batch_idx, (x, y, x_ref, y_ref) in enumerate(train_ref_loader):
            x, y = x.to(self.device), y.to(self.device)
            x_ref, y_ref = x_ref.to(self.device), y_ref.to(self.device)


            total += y.size(0)

            # ---- Reconstruction (Encoder & Decoder & RF discriminator) ---- #
            # _, MSE, KLD = self.train_encoders(x, y, epoch)
            # self.train_decoder(x)

            _, MSE, KLD = self.train_reconstructor(x)
            losses['MSE'] += MSE
            losses['KLD'] += KLD

            if self.weights['real_fake'] > 0:
                losses['RF'] += self.train_rf_discriminator(x)

            # ---- Discriminators ---- #
            for encoder_name in self.encoder_name_list:
                correct, loss = self.train_class_discriminator(encoder_name, x, y)
                corrects['class'][encoder_name] += correct
                losses['class'][encoder_name] += loss

                correct, loss = self.train_membership_discriminator(encoder_name, x, y, x_ref, y_ref)
                corrects['membership'][encoder_name] += correct
                losses['membership'][encoder_name] += loss

            if epoch > self.disentanglement_start_epoch - 1:
                self.disentangle_encoders(x, y)

            if batch_idx == 0:
                mu = torch.zeros((x.shape[0], self.z_dim)).to(self.device)
                logvar = torch.zeros((x.shape[0], self.z_dim)).to(self.device)
                for encoder_idx, encoder_name in enumerate(self.encoder_name_list):
                    mu_, logvar_ = self.encoders[encoder_name](x)
                    mu[:, self.z_idx[encoder_name]] = mu_[ :, self.z_idx[encoder_name]]
                    logvar[:, self.z_idx[encoder_name]] = logvar_[:, self.z_idx[encoder_name]]

                z = self.reparameterize(mu, logvar)
                recons = self.decoder(z)
                vutils.save_image(recons, os.path.join( self.reconstruction_path, '{}.png'.format(epoch)), nrow=8, normalize=True)

        for encoder_name in self.encoder_name_list:
            # self.class_loss_dict[encoder_name] = losses['class'][encoder_name] 
            # self.membership_loss_dict[encoder_name] = losses['membership'][encoder_name]  

            self.class_acc_dict[encoder_name] = corrects['class'][encoder_name] / total
            self.membership_acc_dict[encoder_name] = corrects['membership'][encoder_name] / (2 * total)

        if self.print_training:
            # class_loss = 'class) '
            # membership_loss = 'membership) '
            # for encoder_name in self.encoder_name_list:
            #     class_loss += '{}: {:.4f}, '.format(encoder_name, self.class_loss_dict[encoder_name])
            #     membership_loss += '{}: {:.4f}, '.format(encoder_name, self.membership_loss_dict[encoder_name])
            # print('Epoch: {:>3},'.format(epoch), class_loss, membership_loss)

            class_acc = 'class) '
            membership_acc = 'membership) '
            for encoder_name in self.encoder_name_list:
                class_acc += '{}: {:.4f}, '.format(encoder_name, self.class_acc_dict[encoder_name])
                membership_acc += '{}: {:.4f}, '.format(encoder_name, self.membership_acc_dict[encoder_name])
            print('Epoch: {:>3},'.format(epoch), class_acc, membership_acc)
            print(losses)

            print()

        

    def train_encoders(self, x, y, epoch):
        total_loss = 0

        for encoder_idx, encoder_name in enumerate(self.encoder_name_list):
            self.encoders_opt[encoder_name].zero_grad()

            mu = torch.zeros((x.shape[0], self.z_dim)).to(self.device)
            logvar = torch.zeros((x.shape[0], self.z_dim)).to(self.device)

            for encoder_name_ in self.encoder_name_list:
                mu_, logvar_ = self.encoders[encoder_name_](x)
                mu[:, self.z_idx[encoder_name_]] = mu_[:, self.z_idx[encoder_name_]]
                logvar[:, self.z_idx[encoder_name_]] = logvar_[:, self.z_idx[encoder_name_]]

            z = self.reparameterize(mu, logvar)

            recons = self.decoder(z)
            vae_loss, MSE, KLD = self.vae_loss(recons, x, mu, logvar)

            # real_logit = self.rf_disc(x)
            fake_logit = self.rf_disc(recons)

            if self.adversarial_loss_mode == 'gan':
                rf_loss = self.bce_loss(fake_logit, torch.ones_like(fake_logit))
            elif 'wgan' in self.adversarial_loss_mode:
                if self.reduction == 'sum':
                    rf_loss = -torch.sum(fake_logit)
                elif self.reduction == 'mean':
                    rf_loss = -torch.mean(fake_logit)

            loss = self.weights['recon'] * vae_loss + self.weights['real_fake'] * rf_loss
            if encoder_idx == len(self.encoder_name_list) - 1:
                total_loss += loss

            if epoch > self.disentanglement_start_epoch - 1:
                self.encoders_opt[encoder_name].zero_grad()

                class_loss = self.calculate_class_loss(encoder_name, x, y)
                membership_loss = self.calculate_membership_loss(encoder_name, x, y)

                total_loss += class_loss + membership_loss
                loss = class_loss + membership_loss + loss

            loss.backward()
            self.encoders_opt[encoder_name].step()

        return total_loss.item(), MSE.item(), KLD.item()

    def calculate_class_loss(self, encoder_name, x, y):
        if encoder_name[0] == 'p':
            class_weight = 1. * self.weights['class_pos']
        elif encoder_name[0] == 'n':
            class_weight = -1. * self.weights['class_neg']

        mu_, logvar_ = self.encoders[encoder_name](x)
        mu = mu_[:, self.z_idx[encoder_name]]
        logvar = logvar_[:, self.z_idx[encoder_name]]

        z = self.reparameterize(mu, logvar)
        class_pred = self.class_discs[encoder_name](z)
        class_loss = class_weight * self.class_loss(class_pred, y)

        return class_loss

    def calculate_membership_loss(self, encoder_name, x, y, in_flag=True):
        targets_onehot = torch.zeros((len(y), self.class_num)).to(self.device)
        targets_onehot = targets_onehot.scatter_(1, y.reshape((-1, 1)), 1)

        mu_, logvar_ = self.encoders[encoder_name](x)
        mu = mu_[:, self.z_idx[encoder_name]]
        logvar = logvar_[:, self.z_idx[encoder_name]]

        z = self.reparameterize(mu, logvar)
        z = torch.cat((z, targets_onehot), dim=1)
        mem_pred = self.membership_discs[encoder_name](z)

        if encoder_name[1] == 'p':
            membership_weight = self.weights['membership_pos']
            if in_flag:
                membership_label = torch.ones_like(mem_pred)
            else:
                membership_label = torch.zeros_like(mem_pred)
        elif encoder_name[1] == 'n':
            membership_weight = self.weights['membership_neg']
            if in_flag:
                membership_label = torch.zeros_like(mem_pred)
            else:
                membership_label = torch.ones_like(mem_pred)

        membership_loss = membership_weight * self.membership_loss(mem_pred, membership_label)

        return membership_loss

    def train_reconstructor(self, x):
        for encoder_name in self.encoder_name_list:
            self.encoders_opt[encoder_name].zero_grad()
        self.decoder_opt.zero_grad()

        mu = torch.zeros((x.shape[0], self.z_dim)).to(self.device)
        logvar = torch.zeros((x.shape[0], self.z_dim)).to(self.device)
        for encoder_idx, encoder_name in enumerate(self.encoder_name_list):
            mu_, logvar_ = self.encoders[encoder_name](x)
            mu[:, self.z_idx[encoder_name]] = mu_[:, self.z_idx[encoder_name]]
            logvar[:, self.z_idx[encoder_name]] = logvar_[:, self.z_idx[encoder_name]]

        z = self.reparameterize(mu, logvar)

        recons = self.decoder(z)
        vae_loss, MSE, KLD = self.vae_loss(recons, x, mu, logvar)
        # recon_loss = self.weights['recon'] * recon_loss
        # recon_loss.backward()
        loss = self.weights['recon'] * vae_loss

        if self.weights['real_fake'] > 0:
            fake_logit = self.rf_disc(recons)

            if self.adversarial_loss_mode == 'gan':
                rf_loss = self.bce_loss(fake_logit, torch.ones_like(fake_logit))
            elif 'wgan' in self.adversarial_loss_mode:
                if self.reduction == 'sum':
                    rf_loss = -torch.sum(fake_logit)
                elif self.reduction == 'mean':
                    rf_loss = -torch.mean(fake_logit)

            loss += self.weights['real_fake'] * rf_loss

        loss.backward()

        for encoder_name in self.encoder_name_list:
            self.encoders_opt[encoder_name].step()
        self.decoder_opt.step()

        return loss.item(), MSE.item(), KLD.item()

    def disentangle_encoders(self, x, y):
        targets_onehot = torch.zeros((len(y), self.class_num)).to(self.device)
        targets_onehot = targets_onehot.scatter_(1, y.reshape((-1, 1)), 1)

        for encoder_name in self.encoder_name_list:
            if encoder_name[0] == 'p':
                class_weight = 1. * self.weights['class_pos']
            elif encoder_name[0] == 'n':
                class_weight = -1. * self.weights['class_neg']

            # if encoder_name[1] == 'p':
            #     membership_weight = 1. * self.weights['membership_pos']
            # elif encoder_name[1] == 'n':
            #     membership_weight = -1. * self.weights['membership_neg']


            self.encoders_opt[encoder_name].zero_grad()

            mu_, logvar_ = self.encoders[encoder_name](x)
            mu = mu_[:, self.z_idx[encoder_name]]
            logvar = logvar_[:, self.z_idx[encoder_name]]

            mu_dec = torch.zeros_like(mu_)
            logvar_dec = torch.zeros_like(logvar_)

            mu_dec[:, self.z_idx[encoder_name]] = mu
            logvar_dec[:, self.z_idx[encoder_name]] = logvar

            z_dec = self.reparameterize(mu_dec, logvar_dec)

            z = self.reparameterize(mu, logvar)
            class_pred = self.class_discs[encoder_name](z)
            class_loss = class_weight * self.class_loss(class_pred, y)

            z = torch.cat((z, targets_onehot), dim=1)
            mem_pred = self.membership_discs[encoder_name](z)
            if encoder_name[1] == 'p':
                membership_weight = self.weights['membership_pos']
                membership_label = torch.ones_like(mem_pred)
            elif encoder_name[1] == 'n':
                membership_weight = self.weights['membership_neg']
                membership_label = torch.zeros_like(mem_pred)
            # membership_loss = membership_weight * self.membership_loss(mem_pred, torch.ones_like(mem_pred))
            membership_loss = membership_weight * self.membership_loss(mem_pred, membership_label)

            recons = self.decoder(z_dec)
            recon_loss, _, _ = self.vae_loss(recons, x, mu_dec, logvar_dec)

            loss = class_loss + membership_loss + self.small_recon_weight * recon_loss
            loss.backward()
            self.encoders_opt[encoder_name].step()

    # def train_decoder(self, x):
    #     self.decoder_opt.zero_grad()

    #     mu = torch.zeros((x.shape[0], self.z_dim)).to(self.device)
    #     logvar = torch.zeros((x.shape[0], self.z_dim)).to(self.device)
    #     for encoder_idx, encoder_name in enumerate(self.encoder_name_list):
    #         mu_, logvar_ = self.encoders[encoder_name](x)
    #         mu[:, self.z_idx[encoder_name]] = mu_[:, self.z_idx[encoder_name]]
    #         logvar[:, self.z_idx[encoder_name]] = logvar_[:, self.z_idx[encoder_name]]

    #     z = self.reparameterize(mu, logvar)

    #     recons = self.decoder(z)
    #     vae_loss, MSE, KLD = self.vae_loss(recons, x, mu, logvar)

    #     fake_logit = self.rf_disc(recons)

    #     if self.adversarial_loss_mode == 'gan':
    #         rf_loss = self.bce_loss(fake_logit, torch.ones_like(fake_logit))
    #     elif 'wgan' in self.adversarial_loss_mode:
    #         rf_loss = -torch.sum(fake_logit)

    #     recon_loss = self.weights['recon'] * vae_loss + self.weights['real_fake'] * rf_loss
    #     recon_loss.backward()

    #     self.decoder_opt.step()

    def train_rf_discriminator(self, x):
        self.rf_disc_opt.zero_grad()

        mu = torch.zeros((x.shape[0], self.z_dim)).to(self.device)
        logvar = torch.zeros((x.shape[0], self.z_dim)).to(self.device)
        for encoder_idx, encoder_name in enumerate(self.encoder_name_list):
            mu_, logvar_ = self.encoders[encoder_name](x)
            mu[:, self.z_idx[encoder_name]] = mu_[:, self.z_idx[encoder_name]]
            logvar[:, self.z_idx[encoder_name]] = logvar_[:, self.z_idx[encoder_name]]

        z = self.reparameterize(mu, logvar)
        recons = self.decoder(z)

        real_logit = self.rf_disc(x)
        fake_logit = self.rf_disc(recons)
        if self.adversarial_loss_mode == 'gan':
            rf_loss = self.bce_loss(real_logit, torch.ones_like(real_logit)) + self.bce_loss(fake_logit, torch.zeros_like(fake_logit))
        elif self.adversarial_loss_mode == 'wgan':
            if self.reduction == 'sum':
                rf_loss = - torch.sum(real_logit) + torch.sum(fake_logit)
            elif self.reduction == 'mean':
                rf_loss = - torch.mean(real_logit) + torch.mean(fake_logit)
        elif self.adversarial_loss_mode == 'wgan-gp':
            if self.reduction == 'sum':
                rf_loss = - torch.sum(real_logit) + torch.sum(fake_logit)
            elif self.reduction == 'mean':
                rf_loss = - torch.mean(real_logit) + torch.mean(fake_logit)
            rf_loss += self.gradient_penalty_weight * compute_gradient_penalty(self.rf_disc, x.data, recons.data)

        loss = self.weights['real_fake'] * rf_loss

        loss.backward()
        self.rf_disc_opt.step()

        return loss.item()

    def train_class_discriminator(self, encoder_name, x, y):
        self.class_discs_opt[encoder_name].zero_grad()

        mu_, logvar_ = self.encoders[encoder_name](x)
        mu = mu_[:, self.z_idx[encoder_name]]
        logvar = logvar_[:, self.z_idx[encoder_name]]
        z = self.reparameterize(mu, logvar)

        pred = self.class_discs[encoder_name](z)
        class_loss = self.class_loss(pred, y)
        class_loss.backward()

        self.class_discs_opt[encoder_name].step()

        _, pred_class = pred.max(1)
        return pred_class.eq(y).sum().item(), class_loss.item()

    def train_membership_discriminator(self, encoder_name, x, y, x_ref, y_ref):
        self.membership_discs_opt[encoder_name].zero_grad()

        mu_, logvar_ = self.encoders[encoder_name](x)
        mu = mu_[:, self.z_idx[encoder_name]]
        logvar = logvar_[:, self.z_idx[encoder_name]]
        z = self.reparameterize(mu, logvar)

        targets_onehot = torch.zeros((len(y), self.class_num)).to(self.device)
        targets_onehot = targets_onehot.scatter_(1, y.reshape((-1, 1)), 1)
        z = torch.cat((z, targets_onehot), dim=1)
        pred = self.membership_discs[encoder_name](z)
        in_loss = self.membership_loss(pred, torch.ones_like(pred))

        mu_, logvar_ = self.encoders[encoder_name](x_ref)
        mu = mu_[:, self.z_idx[encoder_name]]
        logvar = logvar_[:, self.z_idx[encoder_name]]
        z_ref = self.reparameterize(mu, logvar)

        targets_ref_onehot = torch.zeros((len(y_ref), self.class_num)).to(self.device)
        targets_ref_onehot = targets_ref_onehot.scatter_(1, y_ref.reshape((-1, 1)), 1)
        z_ref = torch.cat((z_ref, targets_ref_onehot), dim=1)
        pred_ref = self.membership_discs[encoder_name](z_ref)
        out_loss = self.membership_loss(pred_ref, torch.zeros_like(pred_ref))

        membership_loss = in_loss + out_loss
        membership_loss.backward()
        self.membership_discs_opt[encoder_name].step()

        pred = torch.sigmoid(pred).cpu().detach().numpy().squeeze(axis=1)
        pred_ref = torch.sigmoid(pred_ref).cpu().detach().numpy().squeeze(axis=1)
        pred_concat = np.concatenate((pred, pred_ref))
        inout_concat = np.concatenate((np.ones_like(pred), np.zeros_like(pred_ref)))

        return np.sum(inout_concat == np.round(pred_concat)), membership_loss.item()

    def inference(self, loader, epoch, inference_type='valid'):
        for encoder_name in self.encoder_name_list:
            self.encoders[encoder_name].eval()
            self.class_discs[encoder_name].eval()
            self.membership_discs[encoder_name].eval()
        self.decoder.eval()

        loss = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)

                mu = torch.zeros((x.shape[0], self.z_dim)).to(self.device)
                logvar = torch.zeros((x.shape[0], self.z_dim)).to(self.device)
                for encoder_idx, encoder_name in enumerate(self.encoder_name_list):
                    mu_, logvar_ = self.encoders[encoder_name](x)
                    mu[:, self.z_idx[encoder_name]] = mu_[ :, self.z_idx[encoder_name]]
                    logvar[:, self.z_idx[encoder_name]] = logvar_[:, self.z_idx[encoder_name]]

                    loss += self.calculate_class_loss(encoder_name, x, y).item()
                    loss += self.calculate_membership_loss(encoder_name, x, y, in_flag=False).item()

                z = self.reparameterize(mu, logvar)
                recons = self.decoder(z)
                recon_loss, _, _ = self.vae_loss(recons, x, mu, logvar)
                loss += recon_loss.item()

        if inference_type == 'valid':
            # update schedulers
            for encoder_name in self.encoder_name_list:
                self.encoders_opt_scheduler[encoder_name].step(loss)
                self.class_discs_opt_scheduler[encoder_name].step(loss)
                self.membership_discs_opt_scheduler[encoder_name].step(loss)
            self.decoder_opt_scheduler.step(loss)
            self.rf_disc_opt_scheduler.step(loss)

            if loss < self.best_valid_loss:
                # print(loss, self.best_valid_loss)
                state = {
                    'best_valid_loss': loss,
                    'epoch': epoch,
                }

                for encoder_name in self.encoder_name_list:
                    state['enc_' + encoder_name] = self.encoders[encoder_name].state_dict()
                    state['dec'] = self.decoder.state_dict()
                    state['class_disc_' + encoder_name] = self.class_discs[encoder_name].state_dict()
                    state['membership_disc_' + encoder_name] = self.membership_discs[encoder_name].state_dict()

                torch.save(state, os.path.join(self.reconstruction_path, 'ckpt.pth'))
                self.best_valid_loss = loss
                self.early_stop_count = 0
                self.best_class_acc_dict = self.class_acc_dict.copy()
                self.best_membership_acc_dict = self.membership_acc_dict.copy()

                np.save(os.path.join(self.reconstruction_path, 'class_acc.npy'), self.best_class_acc_dict)
                np.save(os.path.join(self.reconstruction_path, 'membership_acc.npy'), self.best_membership_acc_dict)
                vutils.save_image(recons, os.path.join( self.reconstruction_path, '{}.png'.format(epoch)), nrow=10, normalize=True)
                np.save(os.path.join(self.reconstruction_path, 'last_epoch.npy'), epoch)

                # print(self.best_class_acc_dict)
                # print(self.best_membership_acc_dict)

            else:
                self.early_stop_count += 1
                if self.print_training:
                    print('Early stop count: {}'.format(self.early_stop_count))

            if self.early_stop_count == self.early_stop_observation_period:
                print(self.best_class_acc_dict)
                print(self.best_membership_acc_dict)
                self.early_stop_count = 0
                self.early_stop_count_total += 1

                if self.early_stop_count_total == self.EARLY_STOP_COUNT_TOTAL_MAX:
                    self.train_flag = False

                if self.print_training:
                    # print('Early stop count == {}; Terminate training\n'.format( self.early_stop_observation_period))
                    print('Loading best ckpt')
                self.load() 
            
            return loss

    def train(self, train_set, valid_set=None, ref_set=None):
        print('==> Start training {}'.format(self.reconstruction_path))

        if self.resume:
            print('==> Resuming from checkpoint..')
            try:
                last_epoch = np.load(os.path.join(self.reconstruction_path, 'last_epoch.npy'))
                # print(last_epoch)
                self.load(last_epoch)
            except FileNotFoundError:
                print('There is no pre-trained model; Train model from scratch')

        self.train_flag = True
        if self.early_stop:
            valid_loader = DataLoader(valid_set, batch_size=self.test_batch_size, shuffle=True, num_workers=2)

        # if ref_set is not None:
            # ref_set = self.transform(ref_set)
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            permutated_idx = np.random.permutation(ref_set.__len__())
            ref_set = Subset(ref_set, permutated_idx)
            train_ref_set = data.DoubleDataset(train_set, ref_set)
            train_ref_loader = DataLoader(train_ref_set, batch_size=self.train_batch_size, shuffle=True, num_workers=2)
            if self.train_flag:
                self.train_epoch(train_ref_loader, epoch)

                if self.early_stop:
                    self.inference(valid_loader, epoch, inference_type='valid')
                else:

                    if epoch > self.disentanglement_start_epoch - 1:
                        for encoder_name in self.encoder_name_list:
                            self.encoders_opt_scheduler[encoder_name].step()
                            self.class_discs_opt_scheduler[encoder_name].step()
                            self.membership_discs_opt_scheduler[encoder_name].step()
                        self.decoder_opt_scheduler.step()
                        if self.weights['real_fake'] > 0:
                            self.rf_disc_opt_scheduler.step()

                    if (epoch+1) % self.save_step_size == 0:
                        print('Save at {}'.format(epoch+1))

                        state = {
                            # 'best_valid_loss': loss,
                            'epoch': epoch,
                        }

                        for encoder_name in self.encoder_name_list:
                            state['enc_' + encoder_name] = self.encoders[encoder_name].state_dict()
                            state['dec'] = self.decoder.state_dict()
                            state['class_disc_' + encoder_name] = self.class_discs[encoder_name].state_dict()
                            state['membership_disc_' + encoder_name] = self.membership_discs[encoder_name].state_dict()

                        torch.save(state, os.path.join(self.reconstruction_path, 'ckpt{:03d}.pth'.format(epoch+1)))
                        # self.best_valid_loss = loss
                        # self.early_stop_count = 0
                        # self.best_class_acc_dict = self.class_acc_dict
                        # self.best_membership_acc_dict = self.membership_acc_dict

                        np.save(os.path.join(self.reconstruction_path, 'class_acc{:03d}.npy'.format(epoch+1)), self.class_acc_dict)
                        np.save(os.path.join(self.reconstruction_path, 'membership_acc{:03d}.npy'.format(epoch+1)), self.membership_acc_dict)
                        np.save(os.path.join(self.reconstruction_path, 'last_epoch.npy'), epoch)
                        # vutils.save_image(recons, os.path.join(self.reconstruction_path, '{}.png'.format(epoch)), nrow=10)


                # if self.early_stop:
                #     val_loss = self.inference(valid_loader, epoch, inference_type='valid')
                #     for encoder_name in self.encoder_name_list:
                #         self.encoders_opt_scheduler[encoder_name].step(val_loss)
                #         self.class_discs_opt_scheduler[encoder_name].step(val_loss)
                #         self.membership_discs_opt_scheduler[encoder_name].step(val_loss)
                #     self.decoder_opt_scheduler.step(val_loss)
                #     self.rf_disc_opt_scheduler.step(val_loss)

                # else:
                #     if (epoch+1) % self.save_step_size == 0:
                #         print('Save at {}'.format(epoch+1))

                #         state = {
                #             # 'best_valid_loss': loss,
                #             'epoch': epoch,
                #         }

                #         for encoder_name in self.encoder_name_list:
                #             state['enc_' + encoder_name] = self.encoders[encoder_name].state_dict()
                #             state['dec'] = self.decoder.state_dict()
                #             state['class_disc_' + encoder_name] = self.class_discs[encoder_name].state_dict()
                #             state['membership_disc_' + encoder_name] = self.membership_discs[encoder_name].state_dict()

                #         torch.save(state, os.path.join(self.reconstruction_path, 'ckpt{:03d}.pth'.format(epoch+1)))
                #         # self.best_valid_loss = loss
                #         # self.early_stop_count = 0
                #         # self.best_class_acc_dict = self.class_acc_dict
                #         # self.best_membership_acc_dict = self.membership_acc_dict

                #         np.save(os.path.join(self.reconstruction_path, 'class_acc{:03d}.npy'.format(epoch+1)), self.class_acc_dict)
                #         np.save(os.path.join(self.reconstruction_path, 'membership_acc{:03d}.npy'.format(epoch+1)), self.membership_acc_dict)
                #         np.save(os.path.join(self.reconstruction_path, 'last_epoch.npy'), epoch)
                #         # vutils.save_image(recons, os.path.join(self.reconstruction_path, '{}.png'.format(epoch)), nrow=10)

            else:
                break

    def swap_membership(self, dataset_dict):
        pass

    def reconstruct(self, dataset_dict, recon_type_list):
        recon_flag = {
            'pn_pp_np_nn': [1, 1, 1, 1],
            'pn_pp_nn': [1, 1, 0, 1],
            'pn_pp': [1, 1, 0, 0],
            'pn_nn': [1, 0, 0, 1],
            'pp_np': [0, 1, 1, 0],
            'np_nn': [0, 0, 1, 1],
            'pn': [1, 0, 0, 0],
            'pp': [0, 1, 0, 0],
            'np': [0, 0, 1, 0],
            'nn': [0, 0, 0, 1],
        }

        if self.early_stop:
            last_epoch = np.load(os.path.join(self.reconstruction_path, 'last_epoch.npy'))
            epoch_list = [last_epoch]
        else:
            epoch_list = range(self.save_step_size-1, self.epochs, self.save_step_size)

        for epoch in epoch_list:
            try:
                self.load(epoch)
            except FileNotFoundError:
                print('There is no pre-trained model; First, train a reconstructor.')
                sys.exit(1)
            for encoder_name in self.encoder_name_list:
                self.encoders[encoder_name].eval()
                self.class_discs[encoder_name].eval()
                self.membership_discs[encoder_name].eval()
            self.decoder.eval()

            mse_list = []
            recon_dict = dict()

            for recon_idx, recon_type in enumerate(recon_type_list):
                recon_datasets_dict = {}

                for dataset_type, dataset in dataset_dict.items():
                    loader = DataLoader(dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=2)
                    raws = []
                    recons = []
                    labels = []
                    with torch.no_grad():
                        for batch_idx, (inputs, targets) in enumerate(loader):
                            inputs = inputs.to(self.device)

                            mu = torch.zeros((inputs.shape[0], self.z_dim)).to(self.device)
                            logvar = torch.zeros((inputs.shape[0], self.z_dim)).to(self.device)
                            for encoder_idx, encoder_name in enumerate(self.encoder_name_list):
                                mu_, logvar_ = self.encoders[encoder_name](inputs)
                                if recon_flag[recon_type][encoder_idx] == 1:
                                    mu[:, self.z_idx[encoder_name]] = mu_[:, self.z_idx[encoder_name]]
                                    logvar[:, self.z_idx[encoder_name]] = logvar_[:, self.z_idx[encoder_name]]

                            z = mu

                            recons_batch = self.decoder(z).cpu()
                            labels_batch = targets

                            if len(recons) == 0:
                                raws = inputs.cpu()
                                recons = recons_batch
                                labels = labels_batch

                                if dataset_type == 'train':
                                    if not self.early_stop:
                                        vutils.save_image(recons, os.path.join(self.reconstruction_path, '{}{:03d}.png'.format(recon_type, epoch+1)), nrow=10, normalize=True)
                                    else:
                                        vutils.save_image(recons, os.path.join(self.reconstruction_path, '{}.png'.format(recon_type)), nrow=10, normalize=True)
                                    recon_dict[recon_type] = recons

                                    if recon_idx == 0:
                                        vutils.save_image(raws, os.path.join( self.reconstruction_path, 'raw.png'), nrow=10, normalize=True)

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
                if not self.early_stop:
                    torch.save(recon_datasets_dict, os.path.join(self.reconstruction_path, 'recon_{}{:03d}.pt'.format(recon_type, epoch+1)))
                else:
                    torch.save(recon_datasets_dict, os.path.join(self.reconstruction_path, 'recon_{}.pt'.format(recon_type)))
            if not self.early_stop:
                np.save(os.path.join(self.reconstruction_path, 'mse{:03d}.npy'.format(epoch+1)), mse_list)
            else:
                np.save(os.path.join( self.reconstruction_path, 'mse.npy'), mse_list)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + std * eps

    def get_loss_function(self):
        def loss_function(recon_x, x, mu, logvar):
            MSE = F.mse_loss(recon_x, x, reduction=self.reduction)
            if self.reduction == 'sum':
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).sum()
            elif self.reduction == 'mean':
                KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()).mean()

            return MSE + self.beta * KLD, MSE, self.beta * KLD

        return loss_function


Tensor = torch.cuda.FloatTensor 
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    # real_samples = real_samples.squeeze()
    # fake_samples = fake_samples.squeeze()

    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates

    fake = fake.squeeze()
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty