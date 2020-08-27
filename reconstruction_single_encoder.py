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


class SingleReconstructor(object):
    def __init__(self, args):
        self.reconstruction_path = args.reconstruction_path
        if not os.path.exists(self.reconstruction_path):
            os.makedirs(self.reconstruction_path)

        self.beta = args.beta
        self.train_batch_size = args.recon_train_batch_size
        self.test_batch_size = args.test_batch_size
        self.epochs = args.epochs
        self.early_stop = args.early_stop
        self.early_stop_observation_period = args.early_stop_observation_period
        self.use_scheduler = False
        self.print_training = args.print_training
        self.class_num = args.class_num
        self.disentangle_with_reparameterization = args.disentangle_with_reparameterization

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

            'membership_fz': module.MembershipDiscriminator(self.z_dim + args.class_num, 1),
            'membership_cz': module.MembershipDiscriminator(self.disc_input_dim + args.class_num, 1),
            'membership_mz': module.MembershipDiscriminator(self.disc_input_dim + args.class_num, 1),
        }

        self.recon_loss = self.get_loss_function()
        self.class_loss = nn.CrossEntropyLoss(reduction='sum')
        self.membership_loss = nn.BCEWithLogitsLoss(reduction='sum')

        # optimizer
        self.optimizer = dict()
        for net_type in self.nets:
            self.optimizer[net_type] = optim.Adam(self.nets[net_type].parameters(), lr=args.recon_lr,
                                                  betas=(0.5, 0.999))
        self.discriminator_lr = args.disc_lr
        for disc_type in self.discs:
            self.optimizer[disc_type] = optim.Adam(self.discs[disc_type].parameters(), lr=self.discriminator_lr,
                                                   betas=(0.5, 0.999))

        self.weights = {
            'recon': args.recon_weight,
            'class_fz': args.class_fz_weight,
            'class_cz': args.class_cz_weight,
            'class_mz': args.class_mz_weight,
            'membership_fz': args.membership_fz_weight,
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

        self.disentangle = (
                self.weights['class_fz'] + self.weights['class_cz'] + self.weights['class_mz'] +
                self.weights['membership_fz'] + self.weights['membership_cz'] + self.weights['membership_mz'] > 0)

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
        # print('====> Loading checkpoint {}'.format(self.reconstruction_path))
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
        class_z, _ = self.split_class_membership(z)
        pred = self.discs['class_cz'](class_z)
        class_loss = self.class_loss(pred, targets)
        class_loss.backward()
        self.optimizer['class_cz'].step()

        _, pred_class = pred.max(1)
        return pred_class.eq(targets).sum().item(), class_loss.item()

    def train_disc_class_mz(self, inputs, targets):
        self.optimizer['class_mz'].zero_grad()
        z = self.inference_z(inputs)
        _, membership_z = self.split_class_membership(z)
        pred = self.discs['class_mz'](membership_z)
        class_loss_membership = self.class_loss(pred, targets)
        class_loss_membership.backward()
        self.optimizer['class_mz'].step()

        _, pred_class_from_membership = pred.max(1)
        return pred_class_from_membership.eq(targets).sum().item(), class_loss_membership.item()

    def train_disc_membership_fz(self, inputs, targets, inputs_ref, targets_ref):
        self.optimizer['membership_fz'].zero_grad()

        z = self.inference_z(inputs)
        targets_onehot = torch.zeros((len(targets), self.class_num)).to(self.device)
        targets_onehot = targets_onehot.scatter_(1, targets.reshape((-1, 1)), 1)
        z = torch.cat((z, targets_onehot), dim=1)
        pred = self.discs['membership_fz'](z)
        in_loss = self.membership_loss(pred, torch.ones_like(pred))

        z_ref = self.inference_z(inputs_ref)
        targets_ref_onehot = torch.zeros((len(targets_ref), self.class_num)).to(self.device)
        targets_ref_onehot = targets_ref_onehot.scatter_(1, targets_ref.reshape((-1, 1)), 1)
        z_ref = torch.cat((z_ref, targets_ref_onehot), dim=1)
        pred_ref = self.discs['membership_fz'](z_ref)
        out_loss = self.membership_loss(pred_ref, torch.zeros_like(pred_ref))

        membership_loss = in_loss + out_loss
        membership_loss.backward()
        self.optimizer['membership_fz'].step()

        pred = pred.cpu().detach().numpy().squeeze(axis=1)
        pred_ref = pred_ref.cpu().detach().numpy().squeeze(axis=1)
        pred_concat = np.concatenate((pred, pred_ref))
        inout_concat = np.concatenate((np.ones_like(pred), np.zeros_like(pred_ref)))

        return np.sum(inout_concat == np.round(pred_concat)), membership_loss.item()

    def train_disc_membership_cz(self, inputs, targets, inputs_ref, targets_ref):
        self.optimizer['membership_cz'].zero_grad()

        z = self.inference_z(inputs)
        class_z, _ = self.split_class_membership(z)
        targets_onehot = torch.zeros((len(targets), self.class_num)).to(self.device)
        targets_onehot = targets_onehot.scatter_(1, targets.reshape((-1, 1)), 1)
        class_z = torch.cat((class_z, targets_onehot), dim=1)
        pred = self.discs['membership_cz'](class_z)
        in_loss = self.membership_loss(pred, torch.ones_like(pred))

        z_ref = self.inference_z(inputs_ref)
        class_z_ref, _ = self.split_class_membership(z_ref)
        targets_ref_onehot = torch.zeros((len(targets_ref), self.class_num)).to(self.device)
        targets_ref_onehot = targets_ref_onehot.scatter_(1, targets_ref.reshape((-1, 1)), 1)
        class_z_ref = torch.cat((class_z_ref, targets_ref_onehot), dim=1)
        pred_ref = self.discs['membership_cz'](class_z_ref)
        out_loss = self.membership_loss(pred_ref, torch.zeros_like(pred_ref))

        membership_loss = in_loss + out_loss
        membership_loss.backward()
        self.optimizer['membership_cz'].step()

        pred = pred.cpu().detach().numpy().squeeze(axis=1)
        pred_ref = pred_ref.cpu().detach().numpy().squeeze(axis=1)
        pred_concat = np.concatenate((pred, pred_ref))
        inout_concat = np.concatenate((np.ones_like(pred), np.zeros_like(pred_ref)))

        return np.sum(inout_concat == np.round(pred_concat)), membership_loss.item()

    def train_disc_membership_mz(self, inputs, targets, inputs_ref, targets_ref):
        self.optimizer['membership_mz'].zero_grad()

        z = self.inference_z(inputs)
        _, membership_z = self.split_class_membership(z)
        targets_onehot = torch.zeros((len(targets), self.class_num)).to(self.device)
        targets_onehot = targets_onehot.scatter_(1, targets.reshape((-1, 1)), 1)
        membership_z = torch.cat((membership_z, targets_onehot), dim=1)
        pred = self.discs['membership_mz'](membership_z)
        in_loss = self.membership_loss(pred, torch.ones_like(pred))

        z_ref = self.inference_z(inputs_ref)
        _, membership_z_ref = self.split_class_membership(z_ref)
        targets_ref_onehot = torch.zeros((len(targets_ref), self.class_num)).to(self.device)
        targets_ref_onehot = targets_ref_onehot.scatter_(1, targets_ref.reshape((-1, 1)), 1)
        membership_z_ref = torch.cat((membership_z_ref, targets_ref_onehot), dim=1)
        pred_ref = self.discs['membership_mz'](membership_z_ref)
        out_loss = self.membership_loss(pred_ref, torch.zeros_like(pred_ref))

        membership_loss = in_loss + out_loss
        membership_loss.backward()
        self.optimizer['membership_mz'].step()

        pred = pred.cpu().detach().numpy().squeeze(axis=1)
        pred_ref = pred_ref.cpu().detach().numpy().squeeze(axis=1)
        pred_concat = np.concatenate((pred, pred_ref))
        inout_concat = np.concatenate((np.ones_like(pred), np.zeros_like(pred_ref)))

        return np.sum(inout_concat == np.round(pred_concat)), membership_loss.item()

    def disentangle_z(self, inputs, targets):
        self.optimizer['encoder'].zero_grad()
        loss = 0

        z = self.inference_z(inputs)
        cz, mz = self.split_class_membership(z)
        targets_onehot = torch.zeros((len(targets), self.class_num)).to(self.device)
        targets_onehot = targets_onehot.scatter_(1, targets.reshape((-1, 1)), 1)

        if self.weights['class_fz'] != 0:
            pred = self.discs['class_fz'](z)
            loss += self.weights['class_fz'] * self.class_loss(pred, targets)

        if self.weights['class_cz'] != 0:
            pred = self.discs['class_cz'](cz)
            loss += self.weights['class_cz'] * self.class_loss(pred, targets)

        if self.weights['class_mz'] != 0:
            pred = self.discs['class_mz'](mz)
            loss += -self.weights['class_mz'] * self.class_loss(pred, targets)

        if self.weights['membership_fz'] != 0:
            pred = self.discs['membership_fz'](torch.cat((z, targets_onehot), dim=1))
            loss += - self.weights['membership_fz'] * self.membership_loss(pred, torch.ones_like(pred))

        if self.weights['membership_cz'] != 0:
            pred = self.discs['membership_cz'](torch.cat((cz, targets_onehot), dim=1))
            loss += - self.weights['membership_cz'] * self.membership_loss(pred, torch.ones_like(pred))

        if self.weights['membership_mz'] != 0:
            pred = self.discs['membership_mz'](torch.cat((mz, targets_onehot), dim=1))
            loss += self.weights['membership_mz'] * self.membership_loss(pred, torch.ones_like(pred))

        loss.backward()
        self.optimizer['encoder'].step()

    def inference(self, loader, epoch, type='valid'):
        for net_type in self.nets:
            self.nets[net_type].eval()
        for disc_type in self.discs:
            self.discs[disc_type].eval()

        loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                mu, logvar = self.nets['encoder'](inputs)
                z = self.reparameterize(mu, logvar)

                recons = self.nets['decoder'](z)
                recon_loss, MSE, KLD = self.recon_loss(recons, inputs, mu, logvar)
                loss += recon_loss.item()

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
            valid_loader = DataLoader(valid_set, batch_size=self.test_batch_size, shuffle=True, num_workers=2)
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
        try:
            self.load()
        except FileNotFoundError:
            print('There is no pre-trained model; First, train a reconstructor.')
            sys.exit(1)
        self.nets['encoder'].eval()
        self.nets['decoder'].eval()

        mse_list = []
        recon_dict = dict()

        for recon_idx, reconstruction_type in enumerate(reconstruction_type_list):
            recon_datasets_dict = {}
            for dataset_type, dataset in dataset_dict.items():
                # loader = DataLoader(dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=2)
                loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)
                raws = []
                recons = []
                labels = []
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(loader):
                        inputs = inputs.to(self.device)
                        mu, logvar = self.nets['encoder'](inputs)

                        z = torch.zeros_like(mu).to(self.device)

                        mu_class, mu_membership = self.split_class_membership(mu)
                        logvar_class, logvar_membership = self.split_class_membership(logvar)

                        # ---- Swap ---- #
                        mu, logvar = self.nets['encoder'](inputs)

                        z[0][self.class_idx] = mu[0][self.class_idx]
                        z[1][self.class_idx] = mu[1][self.class_idx]
                        z[0][self.membership_idx] = mu[1][self.membership_idx]
                        z[1][self.membership_idx] = mu[0][self.membership_idx]
                        # ---- Swap ---- #

                        # if reconstruction_type == 'cb_mb':
                        #     z[:, self.class_idx] = mu_class
                        #     z[:, self.membership_idx] = mu_membership
                        # elif reconstruction_type == 'cr_mr':
                        #     z[:, self.class_idx] = self.reparameterize(mu_class, logvar_class)
                        #     z[:, self.membership_idx] = self.reparameterize(mu_membership, logvar_membership)
                        #
                        # elif reconstruction_type == 'cb_mz':
                        #     z[:, self.class_idx] = mu_class
                        #     z[:, self.membership_idx] = torch.zeros_like(mu_membership).to(self.device)
                        # elif reconstruction_type == 'cz_mb':
                        #     z[:, self.class_idx] = torch.zeros_like(mu_class).to(self.device)
                        #     z[:, self.membership_idx] = mu_membership
                        # elif reconstruction_type == 'cs1.2_ms0.8':  # scaling
                        #     z[:, self.class_idx] = mu_class * 1.2
                        #     z[:, self.membership_idx] = mu_membership * 0.8
                        # elif reconstruction_type == 'cb_ms0.8':  # scaling
                        #     z[:, self.class_idx] = mu_class
                        #     z[:, self.membership_idx] = mu_membership * 0.8
                        # elif reconstruction_type == 'cb_ms0.5':  # scaling
                        #     z[:, self.class_idx] = mu_class
                        #     z[:, self.membership_idx] = mu_membership * 0.5
                        # elif reconstruction_type == 'cb_ms0.25':  # scaling
                        #     z[:, self.class_idx] = mu_class
                        #     z[:, self.membership_idx] = mu_membership * 0.25
                        # elif reconstruction_type == 'cb_ms0.1':  # scaling
                        #     z[:, self.class_idx] = mu_class
                        #     z[:, self.membership_idx] = mu_membership * 0.1
                        # elif reconstruction_type == 'cb_mb_n1':  # + noise
                        #     z[:, self.class_idx] = mu_class
                        #     z[:, self.membership_idx] = mu_membership + torch.randn_like(mu_membership).to(self.device)
                        # elif reconstruction_type == 'cb_mb_n0.5':  # + noise
                        #     z[:, self.class_idx] = mu_class
                        #     z[:, self.membership_idx] = mu_membership + 0.5 * torch.randn_like(mu_membership).to(
                        #         self.device)
                        # elif reconstruction_type == 'cb_mb_n0.1':  # + noise
                        #     z[:, self.class_idx] = mu_class
                        #     z[:, self.membership_idx] = mu_membership + 0.1 * torch.randn_like(mu_membership).to(
                        #         self.device)
                        # elif reconstruction_type == 'cb_mr':
                        #     z[:, self.class_idx] = mu_class
                        #     z[:, self.membership_idx] = self.reparameterize(mu_membership, logvar_membership)
                        # elif reconstruction_type == 'cb_ms0.5_n0.5':  # scaling
                        #     z[:, self.class_idx] = mu_class
                        #     z[:, self.membership_idx] = mu_membership * 0.5 + 0.5 * torch.randn_like(mu_membership).to(
                        #         self.device)
                        # elif reconstruction_type == 'cb_ms0.5_n0.1':  # scaling
                        #     z[:, self.class_idx] = mu_class
                        #     z[:, self.membership_idx] = mu_membership * 0.5 + 0.1 * torch.randn_like(mu_membership).to(
                        #         self.device)
                        # elif reconstruction_type == 'cb_ms0.8_n0.2':  # scaling
                        #     z[:, self.class_idx] = mu_class
                        #     z[:, self.membership_idx] = mu_membership * 0.8 + 0.2 * torch.randn_like(mu_membership).to(
                        #         self.device)
                        # elif reconstruction_type == 'cb_mConstant':
                        #     z[:, self.class_idx] = mu_class
                        #     for idx in range(z.shape[0]):
                        #         z[idx, self.membership_idx] = mu_membership[0]
                        # elif reconstruction_type == 'cb_mConstant0.8':
                        #     z[:, self.class_idx] = mu_class
                        #     mu_membership_constant = 0.8 * mu_membership[0]
                        #     for idx in range(z.shape[0]):
                        #         z[idx, self.membership_idx] = mu_membership_constant
                        # elif reconstruction_type == 'cb_mInter0.8':
                        #     z[:, self.class_idx] = mu_class
                        #     mu_membership_constant = 0.2 * mu_membership[0]
                        #     for idx in range(z.shape[0]):
                        #         z[idx, self.membership_idx] = 0.8 * mu_membership[idx] + mu_membership_constant
                        #
                        # elif reconstruction_type == 'cb_mAvg':
                        #     z[:, self.class_idx] = mu_class
                        #     mu_membership_constant = torch.mean(mu_membership, dim=0)
                        #     for idx in range(z.shape[0]):
                        #         z[idx, self.membership_idx] = mu_membership_constant
                        #
                        # elif reconstruction_type == 'cb_mr1.2':
                        #     z[:, self.class_idx] = mu_class
                        #     std = torch.exp(0.5 * logvar_membership)
                        #     eps = torch.randn_like(std)
                        #     z[:, self.membership_idx] = mu_membership + 1.2 * std * eps
                        #
                        # elif reconstruction_type == 'cb_mr2.0':
                        #     z[:, self.class_idx] = mu_class
                        #     std = torch.exp(0.5 * logvar_membership)
                        #     eps = torch.randn_like(std)
                        #     z[:, self.membership_idx] = mu_membership + 2. * std * eps

                        # print(mu_membership.shape)
                        # print(mu_membership[0].shape)
                        # z[:, self.membership_idx] = mu_membership[0]
                        # print(torch.repeat_interleave(mu_membership[0], mu_membership.shape[0], 1).shape)
                        # sys.exit(1)

                        # if reconstruction_type == 'cb_mb_sb':
                        #     z[:, self.class_idx] = mu_class
                        #     z[:, self.membership_idx] = mu_membership
                        #     z[:, self.style_idx] = mu_style
                        #
                        # elif reconstruction_type == 'cb_mb_sz':
                        #     z[:, self.class_idx] = mu_class
                        #     z[:, self.membership_idx] = mu_membership
                        #     z[:, self.style_idx] = torch.zeros_like(mu_style).to(self.device)
                        #
                        # elif reconstruction_type == 'cb_mz_sb':
                        #     z[:, self.class_idx] = mu_class
                        #     z[:, self.membership_idx] = torch.zeros_like(mu_membership).to(self.device)
                        #     z[:, self.style_idx] = mu_style
                        #
                        # elif reconstruction_type == 'cb_mz_sz':
                        #     z[:, self.class_idx] = mu_class
                        #     z[:, self.membership_idx] = torch.zeros_like(mu_membership).to(self.device)
                        #     z[:, self.style_idx] = torch.zeros_like(mu_style).to(self.device)
                        #
                        # elif reconstruction_type == 'cz_mb_sb':
                        #     z[:, self.class_idx] = torch.zeros_like(mu_class).to(self.device)
                        #     z[:, self.membership_idx] = mu_membership
                        #     z[:, self.style_idx] = mu_style
                        #
                        # elif reconstruction_type == 'cz_mb_sz':
                        #     z[:, self.class_idx] = torch.zeros_like(mu_class).to(self.device)
                        #     z[:, self.membership_idx] = mu_membership
                        #     z[:, self.style_idx] = torch.zeros_like(mu_style).to(self.device)

                        #
                        # elif reconstruction_type == 'cr_mb':
                        #     z[:, self.class_idx] = self.reparameterize(mu_class, logvar_class)
                        #     z[:, self.membership_idx] = mu_membership
                        #
                        # elif reconstruction_type == 'cr_mr':
                        #     z[:, self.class_idx] = self.reparameterize(mu_class, logvar_class)
                        #     z[:, self.membership_idx] = self.reparameterize(mu_membership, logvar_membership)
                        #
                        # elif reconstruction_type == 'cb_mn':
                        #     z[:, self.class_idx] = mu_class
                        #     z[:, self.membership_idx] = torch.randn_like(mu_membership).to(self.device)

                        recons_batch = self.nets['decoder'](z).cpu()
                        labels_batch = targets

                        # if len(recons) == 0:
                        raws = inputs.cpu()
                        recons = recons_batch
                        labels = labels_batch

                        if dataset_type == 'train':
                            # vutils.save_image(recons, os.path.join(self.reconstruction_path,
                            #                                        'swap/swap{}.png'.format(batch_idx)), nrow=10)
                            # recon_dict[reconstruction_type] = recons
                            #
                            # if recon_idx == 0:
                            #     vutils.save_image(raws, os.path.join(self.reconstruction_path, 'swap/raw{}.png'.format(batch_idx)), nrow=10)

                            vutils.save_image(torch.cat([raws, recons], dim=0), os.path.join(self.reconstruction_path, 'swap/{}.png'.format(batch_idx)), nrow=2)

                        # else:
                        #     raws = torch.cat((raws, inputs.cpu()), axis=0)
                        #     recons = torch.cat((recons, recons_batch), axis=0)
                        #     labels = torch.cat((labels, labels_batch), axis=0)

                recon_datasets_dict[dataset_type] = {
                    'recons': recons,
                    'labels': labels,
                }

                mse_list.append(F.mse_loss(recons, raws).item())

            # todo : refactor dict to CustomDataset
            torch.save(recon_datasets_dict,
                       os.path.join(self.reconstruction_path, 'recon_{}.pt'.format(reconstruction_type)))

        np.save(os.path.join(self.reconstruction_path, 'mse.npy'), mse_list)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + std * eps

    def inference_z(self, z):
        mu, logvar = self.nets['encoder'](z)
        if self.disentangle_with_reparameterization:
            return self.reparameterize(mu, logvar)
        else:
            return mu

    def split_class_membership(self, z):
        class_z = z[:, self.class_idx]
        membership_z = z[:, self.membership_idx]

        return class_z, membership_z

    def get_loss_function(self):
        def loss_function(recon_x, x, mu, logvar):
            MSE = F.mse_loss(recon_x, x, reduction='sum')
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).sum()
            return MSE + self.beta * KLD, MSE, KLD

        return loss_function
