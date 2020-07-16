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


class ReconstructorAE(object):
    def __init__(self, args):
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.epochs = args.epochs
        self.early_stop = args.early_stop
        self.early_stop_observation_period = args.early_stop_observation_period
        self.use_scheduler = False
        self.architecture = args.architecture
        self.print_training = args.print_training

        self.z_dim = args.z_dim
        self.disc_input_dim = int(self.z_dim / 2)

        if args.dataset in ['MNIST', 'Fashion-MNIST']:
            self.num_channels = 1
        else:
            self.num_channels = 3

        self.image_size = 64

        self.recon_weight = 1.
        self.class_weight = args.class_weight
        self.membership_weight = args.membership_weight

        self.disentanglement_type = args.disentanglement_type
        self.reconstruction_path = args.reconstruction_path

        if not os.path.exists(self.reconstruction_path):
            os.makedirs(self.reconstruction_path)

        if args.dataset in ['MNIST', 'Fashion-MNIST', 'CIFAR-10']:
            self.encoder = module.AEConvEncoder(self.z_dim, self.num_channels)
            self.decoder = module.AEConvDecoder(self.z_dim, self.num_channels)
            self.class_classifier_with_content = module.ClassDiscriminator(self.disc_input_dim, 10)
            self.class_classifier_with_style = module.ClassDiscriminator(self.disc_input_dim, 10)
        elif args.dataset in ['adult', 'location']:
            if args.architecture == 'A':
                self.encoder = module.FCNEncoderA(args.encoder_input_dim, self.z_dim)
                self.decoder = module.FCNDecoderA(args.encoder_input_dim, self.z_dim)

                self.class_classifier_with_full = module.ClassDiscriminator(self.z_dim, args.class_num)
                self.class_classifier_with_content = module.ClassDiscriminator(self.disc_input_dim, args.class_num)
                self.class_classifier_with_style = module.ClassDiscriminator(self.disc_input_dim, args.class_num)

                self.membership_classifier_with_full = module.MembershipDiscriminator(self.z_dim, 1)
                self.membership_classifier_with_content = module.MembershipDiscriminator(self.disc_input_dim, 1)
                self.membership_classifier_with_style = module.MembershipDiscriminator(self.disc_input_dim, 1)
                self.recon_loss = nn.MSELoss()
            elif args.architecture == 'B':
                self.encoder = module.FCNEncoderB(args.encoder_input_dim, self.z_dim)
                self.decoder = module.FCNDecoderB(args.encoder_input_dim, self.z_dim)

                self.class_classifier_with_full = module.ClassDiscriminator(self.z_dim, args.class_num)
                self.class_classifier_with_content = module.ClassDiscriminator(self.disc_input_dim, args.class_num)
                self.class_classifier_with_style = module.ClassDiscriminator(self.disc_input_dim, args.class_num)

                self.membership_classifier_with_full = module.MembershipDiscriminator(self.z_dim, 1)
                self.membership_classifier_with_content = module.MembershipDiscriminator(self.disc_input_dim, 1)
                self.membership_classifier_with_style = module.MembershipDiscriminator(self.disc_input_dim, 1)
                self.recon_loss = nn.MSELoss()
            elif args.architecture == 'C':
                self.encoder = module.FCNEncoderC(args.encoder_input_dim, self.z_dim)
                self.decoder = module.FCNDecoderC(args.encoder_input_dim, self.z_dim)

                self.class_classifier_with_full = module.ClassDiscriminator(self.z_dim, args.class_num)
                self.class_classifier_with_content = module.ClassDiscriminator(self.disc_input_dim, args.class_num)
                self.class_classifier_with_style = module.ClassDiscriminator(self.disc_input_dim, args.class_num)

                self.membership_classifier_with_full = module.MembershipDiscriminator(self.z_dim, 1)
                self.membership_classifier_with_content = module.MembershipDiscriminator(self.disc_input_dim, 1)
                self.membership_classifier_with_style = module.MembershipDiscriminator(self.disc_input_dim, 1)
                self.recon_loss = nn.BCELoss()

        self.optimizer_enc = optim.Adam(self.encoder.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.optimizer_dec = optim.Adam(self.decoder.parameters(), lr=args.lr, betas=(0.5, 0.999))

        self.discriminator_lr = args.lr
        # self.discriminator_lr = 0.0002
        self.optimizer_class_full = optim.Adam(self.class_classifier_with_full.parameters(),
                                               lr=self.discriminator_lr, betas=(0.5, 0.999))
        self.optimizer_class_content = optim.Adam(self.class_classifier_with_content.parameters(),
                                                  lr=self.discriminator_lr, betas=(0.5, 0.999))
        self.optimizer_class_style = optim.Adam(self.class_classifier_with_style.parameters(),
                                                lr=self.discriminator_lr, betas=(0.5, 0.999))

        self.optimizer_membership_full = optim.Adam(self.membership_classifier_with_full.parameters(),
                                                    lr=self.discriminator_lr, betas=(0.5, 0.999))
        self.optimizer_membership_content = optim.Adam(self.membership_classifier_with_content.parameters(),
                                                       lr=self.discriminator_lr, betas=(0.5, 0.999))
        self.optimizer_membership_style = optim.Adam(self.membership_classifier_with_style.parameters(),
                                                     lr=self.discriminator_lr, betas=(0.5, 0.999))

        self.scheduler_enc = StepLR(self.optimizer_enc, step_size=100, gamma=0.1)
        self.scheduler_dec = StepLR(self.optimizer_dec, step_size=100, gamma=0.1)

        self.class_loss = nn.CrossEntropyLoss()
        self.membership_loss = nn.BCEWithLogitsLoss()

        self.device = torch.device("cuda:{}".format(args.gpu_id))
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        self.class_classifier_with_full = self.class_classifier_with_full.to(self.device)
        self.class_classifier_with_content = self.class_classifier_with_content.to(self.device)
        self.class_classifier_with_style = self.class_classifier_with_style.to(self.device)

        self.membership_classifier_with_full = self.membership_classifier_with_full.to(self.device)
        self.membership_classifier_with_content = self.membership_classifier_with_content.to(self.device)
        self.membership_classifier_with_style = self.membership_classifier_with_style.to(self.device)

        self.start_epoch = 0
        self.best_valid_loss = float("inf")
        self.train_loss = 0
        self.early_stop_count = 0

        self.class_acc_full = 0
        self.class_acc_content = 0
        self.class_acc_style = 0

        self.membership_acc_full = 0
        self.membership_acc_content = 0
        self.membership_acc_style = 0

        # print(self.encoder)
        # print(self.decoder)

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
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.class_classifier_with_full.load_state_dict(checkpoint['class_classifier_with_full'])
        self.class_classifier_with_content.load_state_dict(checkpoint['class_classifier_with_content'])
        self.class_classifier_with_style.load_state_dict(checkpoint['class_classifier_with_style'])
        self.membership_classifier_with_full.load_state_dict(checkpoint['membership_classifier_with_full'])
        self.membership_classifier_with_content.load_state_dict(checkpoint['membership_classifier_with_content'])
        self.membership_classifier_with_style.load_state_dict(checkpoint['membership_classifier_with_style'])
        self.start_epoch = checkpoint['epoch']

    def train_epoch(self, train_ref_loader, epoch):
        self.encoder.train()
        self.decoder.train()
        self.class_classifier_with_full.train()
        self.class_classifier_with_content.train()
        self.class_classifier_with_style.train()
        self.membership_classifier_with_full.train()
        self.membership_classifier_with_content.train()
        self.membership_classifier_with_style.train()

        recon_train_loss = 0
        correct_class_from_full = 0
        correct_class_from_content = 0
        correct_class_from_style = 0

        correct_membership_from_full = 0
        correct_membership_from_content = 0
        correct_membership_from_style = 0
        total = 0

        for batch_idx, (inputs, targets, inputs_ref, targets_ref) in enumerate(train_ref_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs_ref, targets_ref = inputs_ref.to(self.device), targets_ref.to(self.device)

            total += targets.size(0)

            # ---- Reconstruction (Encoder & Decoder) ---- #
            recon_loss = self.train_AE(inputs)

            # ---- Class classifiers ---- #
            correct_class_from_full += self.train_class_classifier_with_full(inputs, targets)
            correct_class_from_content += self.train_class_classifier_with_content(inputs, targets)
            correct_class_from_style += self.train_class_classifier_with_style(inputs, targets)

            # ---- Membership classifiers ---- #
            correct_membership_from_full += self.train_membership_classifier_with_full(inputs, targets, inputs_ref,
                                                                                       targets_ref)
            correct_membership_from_content += self.train_membership_classifier_with_content(inputs, targets,
                                                                                             inputs_ref, targets_ref)
            correct_membership_from_style += self.train_membership_classifier_with_style(inputs, targets, inputs_ref,
                                                                                         targets_ref)

            # ---- Disentanglement (Encoder & Classifiers) ---- #
            if self.disentanglement_type == 'type1':
                self.disentangle_type1(inputs, targets)

            elif self.disentanglement_type == 'type2':
                self.disentangle_type2(inputs, targets)

            elif self.disentanglement_type == 'type3':
                self.disentangle_type3(inputs, targets, inputs_ref, targets_ref)

            elif self.disentanglement_type == 'type4':
                self.disentangle_type4(inputs, targets, inputs_ref, targets_ref)

            elif self.disentanglement_type == 'type5':
                self.disentangle_type1(inputs, targets)
                self.disentangle_type4(inputs, targets, inputs_ref, targets_ref)

            elif self.disentanglement_type == 'type6':
                pass

            recon_train_loss += recon_loss.item()

        self.train_loss = recon_train_loss

        self.class_acc_full = correct_class_from_full / total
        self.class_acc_content = correct_class_from_content / total
        self.class_acc_style = correct_class_from_style / total

        self.membership_acc_full = correct_membership_from_full / (2 * total)
        self.membership_acc_content = correct_membership_from_content / (2 * total)
        self.membership_acc_style = correct_membership_from_style / (2 * total)

        if self.print_training:
            print(
                'Epoch: {:>3}, Train Loss: {:.4f}, Class Acc Full : {:.4f}, Class Acc Content : {:.4f}, Class Acc Style : {:.4f}, Membership Acc Full : {:.4f}, Membership Acc Content : {:.4f}, Membership Acc Style : {:.4f}'.format(
                    epoch, self.train_loss, self.class_acc_full, self.class_acc_content, self.class_acc_style,
                    self.membership_acc_full, self.membership_acc_content, self.membership_acc_style, ))

    def train_AE(self, inputs):
        self.optimizer_enc.zero_grad()
        self.optimizer_dec.zero_grad()
        z = self.encoder(inputs)
        recons = self.decoder(z)
        recon_loss = self.recon_weight * self.recon_loss(recons, inputs)
        recon_loss.backward()
        self.optimizer_enc.step()
        self.optimizer_dec.step()
        return recon_loss

    def train_class_classifier_with_full(self, inputs, targets):
        self.optimizer_class_full.zero_grad()
        z = self.encoder(inputs)
        pred = self.class_classifier_with_full(z)
        class_loss_full = self.class_loss(pred, targets)
        class_loss_full.backward()
        self.optimizer_class_full.step()

        _, pred_class_from_full = pred.max(1)
        return pred_class_from_full.eq(targets).sum().item()

    def train_class_classifier_with_content(self, inputs, targets):
        self.optimizer_class_content.zero_grad()
        z = self.encoder(inputs)
        content_z, _ = self.split_z(z)
        pred = self.class_classifier_with_content(content_z)
        class_loss_content = self.class_loss(pred, targets)
        class_loss_content.backward()
        self.optimizer_class_content.step()

        _, pred_class_from_content = pred.max(1)
        return pred_class_from_content.eq(targets).sum().item()

    def train_class_classifier_with_style(self, inputs, targets):
        self.optimizer_class_style.zero_grad()
        z = self.encoder(inputs)
        _, style_z = self.split_z(z)
        pred = self.class_classifier_with_style(style_z)
        class_loss_style = self.class_loss(pred, targets)
        class_loss_style.backward()
        self.optimizer_class_style.step()

        _, pred_class_from_style = pred.max(1)
        return pred_class_from_style.eq(targets).sum().item()

    def train_membership_classifier_with_full(self, inputs, targets, inputs_ref, targets_ref):
        self.optimizer_membership_full.zero_grad()
        z = self.encoder(inputs)
        pred = self.membership_classifier_with_full(z)
        in_loss = self.membership_loss(pred, torch.ones_like(pred))

        z_ref = self.encoder(inputs_ref)
        pred_ref = self.membership_classifier_with_full(z_ref)
        out_loss = self.membership_loss(pred_ref, torch.zeros_like(pred_ref))

        membership_loss = in_loss + out_loss
        membership_loss.backward()
        self.optimizer_membership_full.step()

        pred = pred.cpu().detach().numpy().squeeze()
        pred_ref = pred_ref.cpu().detach().numpy().squeeze()
        pred_concat = np.concatenate((pred, pred_ref))
        inout_concat = np.concatenate((np.ones_like(pred), np.zeros_like(pred_ref)))

        return np.sum(inout_concat == np.round(pred_concat))

    def train_membership_classifier_with_content(self, inputs, targets, inputs_ref, targets_ref):
        self.optimizer_membership_content.zero_grad()
        z = self.encoder(inputs)
        content_z, _ = self.split_z(z)
        pred = self.membership_classifier_with_content(content_z)
        in_loss = self.membership_loss(pred, torch.ones_like(pred))

        z_ref = self.encoder(inputs_ref)
        content_z_ref, _ = self.split_z(z_ref)
        pred_ref = self.membership_classifier_with_content(content_z_ref)
        out_loss = self.membership_loss(pred_ref, torch.zeros_like(pred_ref))

        membership_loss = in_loss + out_loss
        membership_loss.backward()
        self.optimizer_membership_content.step()

        pred = pred.cpu().detach().numpy().squeeze()
        pred_ref = pred_ref.cpu().detach().numpy().squeeze()
        pred_concat = np.concatenate((pred, pred_ref))
        inout_concat = np.concatenate((np.ones_like(pred), np.zeros_like(pred_ref)))

        return np.sum(inout_concat == np.round(pred_concat))
        # return metrics.accuracy_score(inout_concat, np.round(pred_concat))

    def train_membership_classifier_with_style(self, inputs, targets, inputs_ref, targets_ref):
        self.optimizer_membership_style.zero_grad()
        z = self.encoder(inputs)
        _, style_z = self.split_z(z)
        pred = self.membership_classifier_with_style(style_z)
        in_loss = self.membership_loss(pred, torch.ones_like(pred))

        z_ref = self.encoder(inputs_ref)
        _, style_z_ref = self.split_z(z_ref)
        pred_ref = self.membership_classifier_with_style(style_z_ref)
        out_loss = self.membership_loss(pred_ref, torch.zeros_like(pred_ref))

        membership_loss = in_loss + out_loss
        membership_loss.backward()
        self.optimizer_membership_style.step()

        pred = pred.cpu().detach().numpy().squeeze()
        pred_ref = pred_ref.cpu().detach().numpy().squeeze()
        pred_concat = np.concatenate((pred, pred_ref))
        inout_concat = np.concatenate((np.ones_like(pred), np.zeros_like(pred_ref)))

        return np.sum(inout_concat == np.round(pred_concat))

    def disentangle_type1(self, inputs, targets):
        self.optimizer_enc.zero_grad()
        z = self.encoder(inputs)
        _, style_z = self.split_z(z)
        pred = self.class_classifier_with_style(style_z)
        class_loss = -(self.class_weight * self.class_loss(pred, targets))
        class_loss.backward()
        self.optimizer_enc.step()

    def disentangle_type2(self, inputs, targets):
        self.optimizer_enc.zero_grad()
        z = self.encoder(inputs)
        content_z, _ = self.split_z(z)
        pred = self.class_classifier_with_content(content_z)
        class_loss = self.class_weight * self.class_loss(pred, targets)
        class_loss.backward()
        self.optimizer_enc.step()

    def disentangle_type3(self, inputs, targets, inputs_ref, targets_ref):
        self.optimizer_enc.zero_grad()
        z = self.encoder(inputs)
        _, style_z = self.split_z(z)
        pred = self.membership_classifier_with_style(style_z)
        in_loss = self.membership_loss(pred, torch.ones_like(pred))

        z_ref = self.encoder(inputs_ref)
        _, style_z_ref = self.split_z(z_ref)
        pred_ref = self.membership_classifier_with_style(style_z_ref)
        out_loss = self.membership_loss(pred_ref, torch.zeros_like(pred_ref))

        membership_loss = self.membership_weight * (in_loss + out_loss)
        membership_loss.backward()
        self.optimizer_enc.step()

    def disentangle_type4(self, inputs, targets, inputs_ref, targets_ref):
        self.optimizer_enc.zero_grad()
        z = self.encoder(inputs)
        content_z, _ = self.split_z(z)
        pred = self.membership_classifier_with_content(content_z)
        in_loss = self.membership_loss(pred, torch.ones_like(pred))

        # z_ref = self.encoder(inputs_ref)
        # content_z_ref, _ = self.split_z(z_ref)
        # pred_ref = self.membership_classifier_with_content(content_z_ref)
        # out_loss = self.membership_loss(pred_ref, torch.zeros_like(pred_ref))

        # todo : remove out_loss,
        membership_loss = - self.membership_weight * (in_loss)
        # membership_loss = - self.membership_weight * (in_loss + out_loss)
        membership_loss.backward()
        self.optimizer_enc.step()

    def inference(self, loader, epoch, type='valid'):
        self.encoder.eval()
        self.decoder.eval()
        self.class_classifier_with_content.eval()
        self.class_classifier_with_style.eval()

        loss = 0
        correct_class_from_content = 0
        correct_class_from_style = 0

        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                z = self.encoder(inputs)
                recons = self.decoder(z)
                recon_loss = self.recon_loss(recons, inputs)
                loss += recon_loss.item()

                total += targets.size(0)
                content_z, style_z = self.split_z(z)

                # -- Class (valid) -- #
                _, pred_class_from_content = self.class_classifier_with_content(content_z).max(1)
                _, pred_class_from_style = self.class_classifier_with_style(style_z).max(1)

                correct_class_from_content += pred_class_from_content.eq(targets).sum().item()
                correct_class_from_style += pred_class_from_style.eq(targets).sum().item()

        if type == 'valid':
            if loss < self.best_valid_loss:
                if self.print_training:
                    print('Saving..')
                state = {
                    'encoder': self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                    'class_classifier_with_full': self.class_classifier_with_full.state_dict(),
                    'class_classifier_with_content': self.class_classifier_with_content.state_dict(),
                    'class_classifier_with_style': self.class_classifier_with_style.state_dict(),
                    'membership_classifier_with_full': self.membership_classifier_with_full.state_dict(),
                    'membership_classifier_with_content': self.membership_classifier_with_content.state_dict(),
                    'membership_classifier_with_style': self.membership_classifier_with_style.state_dict(),
                    'best_valid_loss': loss,
                    # 'train_loss': self.train_loss,
                    'epoch': epoch,
                }
                torch.save(state, os.path.join(self.reconstruction_path, 'ckpt.pth'))
                self.best_valid_loss = loss
                self.early_stop_count = 0

                # Save acc
                acc_dict = {
                    'class_acc_full': self.class_acc_full,
                    'class_acc_content': self.class_acc_content,
                    'class_acc_style': self.class_acc_style,
                    'membership_acc_full': self.membership_acc_full,
                    'membership_acc_content': self.membership_acc_content,
                    'membership_acc_style': self.membership_acc_style,
                }

                np.save(os.path.join(self.reconstruction_path, 'acc.npy'), acc_dict)

            else:
                self.early_stop_count += 1
                if self.print_training:
                    print('Early stop count: {}'.format(self.early_stop_count))

            if self.early_stop_count == self.early_stop_observation_period:
                if self.print_training:
                    print('Early stop count == {}; Terminate training\n'.format(self.early_stop_observation_period))
                self.train_flag = False

    # def membership_inference(self, loader, epoch):
    #     self.encoder.eval()
    #     self.membership_classifier_with_content.eval()
    #     self.membership_classifier_with_style.eval()
    #
    #     correct_membership_from_content = 0
    #     correct_membership_from_style = 0
    #
    #     total = 0
    #     with torch.no_grad():
    #         for batch_idx, (inputs, targets) in enumerate(loader):
    #             inputs, targets = inputs.to(self.device), targets.to(self.device)
    #
    #             z = self.encoder(inputs)
    #
    #             total += targets.size(0)
    #             content_z, style_z = self.split_z(z)
    #
    #             # -- Membership (train: in, valid:out) -- #
    #             _, pred_membership_from_content = self.membership_classifier_with_content(content_z).max(1)
    #             _, pred_membership_from_style = self.membership_classifier_with_style(style_z).max(1)
    #
    #             correct_membership_from_content += pred_membership_from_content.eq(
    #                 torch.zeros_like(targets)).sum().item()
    #             correct_membership_from_style += pred_membership_from_style.eq(
    #                 torch.zeros_like(targets)).sum().item()

    def train(self, train_set, valid_set=None, ref_set=None):
        print('==> Start training {}'.format(self.reconstruction_path))
        self.train_flag = True

        train_ref_set = data.DoubleDataset(train_set, ref_set)
        train_ref_loader = torch.utils.data.DataLoader(train_ref_set, batch_size=self.train_batch_size, shuffle=True,
                                                       num_workers=2)

        if self.early_stop:
            valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=self.train_batch_size, shuffle=True,
                                                       num_workers=2)
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            if self.train_flag:
                self.train_epoch(train_ref_loader, epoch)
                if self.use_scheduler:
                    self.scheduler_enc.step()
                    self.scheduler_dec.step()
                if self.early_stop:
                    self.inference(valid_loader, epoch, type='valid')
            else:
                break

    def reconstruct(self, dataset_dict):
        print('==> Reconstruct datasets')
        try:
            self.load()
        except FileNotFoundError:
            print('There is no pre-trained model; First, train the disentangler.')
            sys.exit(1)
        self.encoder.eval()
        self.decoder.eval()

        mse_list = []

        for reconstruction_type in ['full_z', 'content_z', 'style_z']:
            recon_datasets_dict = {}
            for dataset_type, dataset in dataset_dict.items():
                loader = torch.utils.data.DataLoader(dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=2)
                raws = []
                recons = []
                labels = []
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(loader):
                        inputs = inputs.to(self.device)
                        z = self.encoder(inputs)

                        content_z, style_z = self.split_z(z)
                        if reconstruction_type == 'content_z':
                            z = torch.cat((content_z, torch.zeros_like(style_z)), axis=1)
                        elif reconstruction_type == 'style_z':
                            z = torch.cat((torch.zeros_like(content_z), style_z), axis=1)

                        recons_batch = self.decoder(z).cpu()
                        # recons_batch = torch.sigmoid(self.decoder(z)).cpu()
                        labels_batch = targets

                        if len(recons) == 0:
                            raws = inputs.cpu()
                            recons = recons_batch
                            labels = labels_batch

                        else:
                            raws = torch.cat((raws, inputs.cpu()), axis=0)
                            recons = torch.cat((recons, recons_batch), axis=0)
                            labels = torch.cat((labels, labels_batch), axis=0)

                recon_datasets_dict[dataset_type] = {
                    'recons': recons,
                    'labels': labels,
                }

                # print(reconstruction_type, dataset_type, F.mse_loss(recons, raws))
                mse_list.append(F.mse_loss(recons, raws).item())

            # todo : refactor dict to CustomDataset
            torch.save(recon_datasets_dict,
                       os.path.join(self.reconstruction_path, 'recon_{}.pt'.format(reconstruction_type)))

        print(mse_list)
        np.save(os.path.join(self.reconstruction_path, 'mse.npy'), mse_list)

    def split_z(self, z):
        content_z = z[:, 0:self.disc_input_dim]
        style_z = z[:, self.disc_input_dim:]

        return content_z, style_z

    def analyze_recons(self):
        recon_full_dict = torch.load(os.path.join(self.reconstruction_path, 'recon_{}.pt'.format('full_z')))
        recon_content_dict = torch.load(os.path.join(self.reconstruction_path, 'recon_{}.pt'.format('content_z')))
        recon_style_dict = torch.load(os.path.join(self.reconstruction_path, 'recon_{}.pt'.format('style_z')))

        for dataset_type in ['train', 'valid', 'test']:
            recon_full = recon_full_dict[dataset_type]['recons']
            recon_content = recon_content_dict[dataset_type]['recons']
            recon_style = recon_style_dict[dataset_type]['recons']

            print(F.mse_loss(recon_content, recon_full))
            print(F.mse_loss(recon_style, recon_full))
            # print(torch.std_mean(recon_full - recon_style))
            #
            # print(torch.std_mean(recon_full - recon_content))
            # print(torch.std_mean(recon_full - recon_style))

