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


# Distinct Encoders + Distinct Discriminators
class Reconstructor(object):
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
        self.share_encoder = args.share_encoder
        self.train_flag = False

        # self.small_recon_weight = 0.001
        self.small_recon_weight = 0.01
        # self.small_recon_weight = 0.1

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
            self.class_discs[encoder_name] = module.ClassDiscriminator(self.base_z_dim, args.class_num)
            self.membership_discs[encoder_name] = module.MembershipDiscriminator(self.base_z_dim + args.class_num, 1)

        # Optimizer
        self.encoders_opt = dict()
        self.class_discs_opt = dict()
        self.membership_discs_opt = dict()
        for encoder_name in self.encoder_name_list:
            self.encoders_opt[encoder_name] = optim.Adam(self.encoders[encoder_name].parameters(), lr=args.recon_lr,
                                                         betas=(0.5, 0.999))
            self.class_discs_opt[encoder_name] = optim.Adam(self.class_discs[encoder_name].parameters(),
                                                            lr=args.disc_lr, betas=(0.5, 0.999))
            self.membership_discs_opt[encoder_name] = optim.Adam(self.membership_discs[encoder_name].parameters(),
                                                                 lr=args.disc_lr, betas=(0.5, 0.999))

        self.decoder_opt = optim.Adam(self.decoder.parameters(), lr=args.recon_lr, betas=(0.5, 0.999))

        # Loss
        self.recon_loss = self.get_loss_function()
        self.class_loss = nn.CrossEntropyLoss(reduction='sum')
        self.membership_loss = nn.BCEWithLogitsLoss(reduction='sum')

        self.weights = {
            'recon': args.recon_weight,
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

        self.disentangle = (
                self.weights['class_pos'] + self.weights['class_neg'] + self.weights['membership_pos']
                + self.weights['membership_neg'] > 0)

        self.start_epoch = 0
        self.best_valid_loss = float("inf")
        # self.train_loss = 0
        self.early_stop_count = 0

        self.class_acc_dict = {
            'pn': 0., 'pp': 0., 'np': 0., 'nn': 0.,
        }
        self.membership_acc_dict = {
            'pn': 0., 'pp': 0., 'np': 0., 'nn': 0.,
        }
        self.best_class_acc_dict = {}
        self.best_membership_acc_dict = {}

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
        checkpoint = torch.load(os.path.join(self.reconstruction_path, 'ckpt.pth'), map_location=self.device)
        for encoder_name in self.encoder_name_list:
            self.encoders[encoder_name].load_state_dict(checkpoint['enc_' + encoder_name])
            self.class_discs[encoder_name].load_state_dict(checkpoint['class_disc_' + encoder_name])
            self.membership_discs[encoder_name].load_state_dict(checkpoint['membership_disc_' + encoder_name])
            self.decoder.load_state_dict(checkpoint['dec'])

        self.start_epoch = checkpoint['epoch']

    def train_epoch(self, train_ref_loader, epoch):
        for encoder_name in self.encoder_name_list:
            self.encoders[encoder_name].train()
            self.class_discs[encoder_name].train()
            self.membership_discs[encoder_name].train()

        total = 0

        losses = {
            'MSE': 0, 'KLD': 0,
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

            # ---- Reconstruction (Encoder & Decoder) ---- #
            recon_loss, MSE, KLD = self.train_reconstructor(x)
            losses['MSE'] += MSE
            losses['KLD'] += KLD

            for encoder_name in self.encoder_name_list:
                correct, loss = self.train_class_discriminator(encoder_name, x, y)
                corrects['class'][encoder_name] += correct
                losses['class'][encoder_name] += loss

                correct, loss = self.train_membership_discriminator(encoder_name, x, y, x_ref, y_ref)
                corrects['membership'][encoder_name] += correct
                losses['membership'][encoder_name] += loss

            if self.disentangle:
                self.disentangle_encoders(x, y)

        for encoder_name in self.encoder_name_list:
            self.class_acc_dict[encoder_name] = corrects['class'][encoder_name] / total
            self.membership_acc_dict[encoder_name] = corrects['membership'][encoder_name] / (2 * total)

        if self.print_training:
            class_acc = 'class) '
            membership_acc = 'membership) '
            for encoder_name in self.encoder_name_list:
                class_acc += '{}: {:.4f}, '.format(encoder_name, self.class_acc_dict[encoder_name])
                membership_acc += '{}: {:.4f}, '.format(encoder_name, self.membership_acc_dict[encoder_name])
            print('Epoch: {:>3},'.format(epoch), class_acc, membership_acc)

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
        recon_loss, MSE, KLD = self.recon_loss(recons, x, mu, logvar)
        recon_loss = self.weights['recon'] * recon_loss
        recon_loss.backward()

        for encoder_name in self.encoder_name_list:
            self.encoders_opt[encoder_name].step()
        self.decoder_opt.step()

        return recon_loss.item(), MSE.item(), KLD.item()

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

        pred = pred.cpu().detach().numpy().squeeze(axis=1)
        pred_ref = pred_ref.cpu().detach().numpy().squeeze(axis=1)
        pred_concat = np.concatenate((pred, pred_ref))
        inout_concat = np.concatenate((np.ones_like(pred), np.zeros_like(pred_ref)))

        return np.sum(inout_concat == np.round(pred_concat)), membership_loss.item()

    def disentangle_encoders(self, x, y):
        targets_onehot = torch.zeros((len(y), self.class_num)).to(self.device)
        targets_onehot = targets_onehot.scatter_(1, y.reshape((-1, 1)), 1)

        for encoder_name in self.encoder_name_list:
            if encoder_name[0] == 'p':
                class_weight = 1. * self.weights['class_pos']
            elif encoder_name[0] == 'n':
                class_weight = -1. * self.weights['class_neg']

            if encoder_name[1] == 'p':
                membership_weight = 1. * self.weights['membership_pos']
            elif encoder_name[1] == 'n':
                membership_weight = -1. * self.weights['membership_neg']

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
            membership_loss = membership_weight * self.membership_loss(mem_pred, torch.ones_like(mem_pred))

            recons = self.decoder(z_dec)
            recon_loss, _, _ = self.recon_loss(recons, x, mu_dec, logvar_dec)

            loss = class_loss + membership_loss + self.small_recon_weight * recon_loss
            loss.backward()
            self.encoders_opt[encoder_name].step()

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
                    mu[:, self.z_idx[encoder_name]] = mu_[:, self.z_idx[encoder_name]]
                    logvar[:, self.z_idx[encoder_name]] = logvar_[:, self.z_idx[encoder_name]]

                z = self.reparameterize(mu, logvar)

                recons = self.decoder(z)
                recon_loss, MSE, KLD = self.recon_loss(recons, x, mu, logvar)
                loss += recon_loss.item()

        if inference_type == 'valid':
            if loss < self.best_valid_loss:
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
                self.best_class_acc_dict = self.class_acc_dict
                self.best_membership_acc_dict = self.membership_acc_dict

                np.save(os.path.join(self.reconstruction_path, 'class_acc.npy'), self.best_class_acc_dict)
                np.save(os.path.join(self.reconstruction_path, 'membership_acc.npy'), self.best_membership_acc_dict)
                vutils.save_image(recons, os.path.join(self.reconstruction_path, '{}.png'.format(epoch)), nrow=10)

            else:
                self.early_stop_count += 1
                if self.print_training:
                    print('Early stop count: {}'.format(self.early_stop_count))

            if self.early_stop_count == self.early_stop_observation_period:
                print(self.best_class_acc_dict)
                print(self.best_membership_acc_dict)
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
                    self.inference(valid_loader, epoch, inference_type='valid')
            else:
                break

    def reconstruct(self, dataset_dict, recon_type_list):
        try:
            self.load()
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

        recon_flag = {
            'pn_pp_np_nn': [1, 1, 1, 1],
            'pn_pp_nn': [1, 1, 0, 1],
            'pn_pp': [1, 1, 0, 0],
            'pp_np': [0, 1, 1, 0],
            'np_nn': [0, 0, 1, 1],
            'pn': [1, 0, 0, 0],
            'pp': [0, 1, 0, 0],
            'np': [0, 0, 1, 0],
            'nn': [0, 0, 0, 1],
        }

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
                                vutils.save_image(recons, os.path.join(self.reconstruction_path,
                                                                       '{}.png'.format(recon_type)), nrow=10)
                                recon_dict[recon_type] = recons

                                if recon_idx == 0:
                                    vutils.save_image(raws, os.path.join(self.reconstruction_path, 'raw.png'), nrow=10)

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
                       os.path.join(self.reconstruction_path, 'recon_{}.pt'.format(recon_type)))

        np.save(os.path.join(self.reconstruction_path, 'mse.npy'), mse_list)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + std * eps

    def inference_z(self, x):
        class_mu, class_logvar = self.nets['class_encoder'](x)
        membership_mu, membership_logvar = self.nets['membership_encoder'](x)

        mu = torch.cat([class_mu, membership_mu], dim=1)
        logvar = torch.cat([class_logvar, membership_logvar], dim=1)

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
