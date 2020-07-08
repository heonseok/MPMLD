import module
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
import torchvision.utils as vutils
import os
import data


class ReconstructorAE(object):
    def __init__(self, args):
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.epochs = args.epochs
        self.early_stop = args.early_stop
        self.early_stop_observation_period = args.early_stop_observation_period

        self.z_dim = args.z_dim
        self.disc_input_dim = int(self.z_dim / 2)

        if args.dataset in ['MNIST', 'Fashion-MNIST']:
            self.num_channels = 1
        else:
            self.num_channels = 3

        self.image_size = 64

        self.disentanglement_type = args.disentanglement_type
        self.reconstruction_path = args.reconstruction_path

        if not os.path.exists(self.reconstruction_path):
            os.makedirs(self.reconstruction_path)

        if args.dataset in ['MNIST', 'Fashion-MNIST', 'CIFAR-10']:
            self.encoder = module.AEConvEncoder(self.z_dim, self.num_channels)
            self.decoder = module.AEConvDecoder(self.z_dim, self.num_channels)
            self.class_classifier_with_content = module.Discriminator(self.disc_input_dim, 10)
            self.class_classifier_with_style = module.Discriminator(self.disc_input_dim, 10)
        elif args.dataset in ['adult', 'location']:
            self.encoder = module.FCNEncoder(args.encoder_input_dim, self.z_dim)
            self.decoder = module.FCNDecoder(args.encoder_input_dim, self.z_dim)
            self.class_classifier_with_content = module.Discriminator(self.disc_input_dim, args.class_num)
            self.class_classifier_with_style = module.Discriminator(self.disc_input_dim, args.class_num)

        self.optimizer_enc = optim.Adam(self.encoder.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.optimizer_dec = optim.Adam(self.decoder.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.optimizer_class_content = optim.Adam(self.class_classifier_with_content.parameters(), lr=args.lr,
                                                  betas=(0.5, 0.999))
        self.optimizer_class_style = optim.Adam(self.class_classifier_with_style.parameters(), lr=args.lr,
                                                betas=(0.5, 0.999))

        # self.criterion = nn.BCEWithLogitsLoss()
        # self.criterion = nn.BCELoss()
        self.recon_loss = nn.MSELoss()
        self.class_loss = nn.CrossEntropyLoss()

        self.device = torch.device("cuda:{}".format(args.gpu_id))
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.class_classifier_with_content = self.class_classifier_with_content.to(self.device)
        self.class_classifier_with_style = self.class_classifier_with_style.to(self.device)

        self.start_epoch = 0
        self.best_valid_loss = float("inf")
        self.train_loss = 0
        self.early_stop_count = 0

        self.class_content_train_acc = 0
        self.class_style_train_acc = 0

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
        self.class_classifier_with_content.load_state_dict(checkpoint['class_classifier_with_content'])
        self.class_classifier_with_style.load_state_dict(checkpoint['class_classifier_with_style'])
        self.start_epoch = checkpoint['epoch']

    def train_epoch(self, train_ref_loader, epoch):
        self.encoder.train()
        self.decoder.train()
        self.class_classifier_with_content.train()
        self.class_classifier_with_style.train()
        recon_train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets, inputs_ref, targets_ref) in enumerate(train_ref_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # ---- Reconstruction (Encoder & Decoder) ---- #
            recon_loss = self.train_AE(inputs)

            # ---- Class classifiers ---- #
            self.train_class_classifier_with_content(inputs, targets)
            self.train_class_classifier_with_style(inputs, targets)

            # ---- Disentanglement (Encoder & Classifiers) ---- #
            if self.disentanglement_type == 'type1':
                self.disentangle_type1(inputs, targets)

            elif self.disentanglement_type == 'type2':
                self.disentangle_type2(inputs, targets)

            elif self.disentanglement_type == 'type3':
                inputs_ref, targets_ref = inputs_ref.to(self.device), targets_ref.to(self.device)
                self.disentangle_type3(inputs, targets, inputs_ref, targets_ref)

            recon_train_loss += recon_loss.item()
            # if self.disentanglement_type != 'base':
            #     _, predicted = pred_label.max(1)
            #     total += targets.size(0)
            #     correct += predicted.eq(targets).sum().item()

        self.train_loss = recon_train_loss
        # if self.disentanglement_type != 'base':
        #     self.train_acc = correct / total

    def train_class_classifier_with_style(self, inputs, targets):
        self.optimizer_class_style.zero_grad()
        z = self.encoder(inputs)
        _, style_z = self.split_z(z)
        pred_label = self.class_classifier_with_style(style_z)
        class_loss_style = self.class_loss(pred_label, targets)
        class_loss_style.backward()
        self.optimizer_class_style.step()
        return pred_label

    def train_class_classifier_with_content(self, inputs, targets):
        self.optimizer_class_content.zero_grad()
        z = self.encoder(inputs)
        content_z, _ = self.split_z(z)
        pred_label = self.class_classifier_with_content(content_z)
        class_loss_content = self.class_loss(pred_label, targets)
        class_loss_content.backward()
        self.optimizer_class_content.step()

    def train_AE(self, inputs):
        self.optimizer_enc.zero_grad()
        self.optimizer_dec.zero_grad()
        z = self.encoder(inputs)
        recons = self.decoder(z)
        recon_loss = self.recon_loss(recons, inputs)
        recon_loss.backward()
        self.optimizer_enc.step()
        self.optimizer_dec.step()
        return recon_loss

    def disentangle_type1(self, inputs, targets):
        self.optimizer_enc.zero_grad()
        z = self.encoder(inputs)
        _, style_z = self.split_z(z)
        pred_label = self.class_classifier_with_style(style_z)
        class_loss = -self.class_loss(pred_label, targets)
        class_loss.backward()
        self.optimizer_enc.step()

        self.optimizer_class_style.zero_grad()
        z = self.encoder(inputs)
        _, style_z = self.split_z(z)
        pred_label = self.class_classifier_with_style(style_z)
        class_loss = self.class_loss(pred_label, targets)
        class_loss.backward()
        self.optimizer_class_style.step()

        return pred_label

    def disentangle_type2(self, inputs, targets):
        self.optimizer_enc.zero_grad()
        z = self.encoder(inputs)
        content_z, _ = self.split_z(z)
        pred_label = self.class_classifier_with_content(content_z)
        class_loss = self.class_loss(pred_label, targets)
        class_loss.backward()
        self.optimizer_enc.step()

        self.optimizer_class_content.zero_grad()
        z = self.encoder(inputs)
        content_z, _ = self.split_z(z)
        pred_label = self.class_classifier_with_content(content_z)
        class_loss = self.class_loss(pred_label, targets)
        class_loss.backward()
        self.optimizer_class_content.step()

        return pred_label

    def disentangle_type3(self, inputs, targets, inputs_ref, targets_ref):
        pass

    def inference(self, loader, epoch, type='valid'):
        self.encoder.eval()
        self.decoder.eval()
        loss = 0
        correct_from_content = 0
        correct_from_style = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                z = self.encoder(inputs)
                recons = self.decoder(z)
                recon_loss = self.recon_loss(recons, inputs)
                loss += recon_loss.item()

                if self.disentanglement_type != 'base':
                    total += targets.size(0)

                    content_z, style_z = self.split_z(z)
                    pred_label_from_content = self.class_classifier_with_content(content_z)
                    pred_label_from_style = self.class_classifier_with_style(style_z)

                    _, predicted_from_content = pred_label_from_content.max(1)
                    _, predicted_from_style = pred_label_from_style.max(1)

                    correct_from_content += predicted_from_content.eq(targets).sum().item()
                    correct_from_style += predicted_from_style.eq(targets).sum().item()

        if type == 'valid':
            if self.disentanglement_type == 'base':
                print('Epoch: {:>3}, Train Loss: {:.4f}, Valid Loss: {:.4f}'.format(epoch, self.train_loss, loss))
            else:
                print(
                    'Epoch: {:>3}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid Acc from Content : {:.4f}, Valid Acc from Stlye : {:.4f}'.format(
                        epoch, self.train_loss, loss, correct_from_content / total, correct_from_style / total))

            if loss < self.best_valid_loss:
                print('Saving..')
                state = {
                    'encoder': self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                    'class_classifier_with_content': self.class_classifier_with_content.state_dict(),
                    'class_classifier_with_style': self.class_classifier_with_style.state_dict(),
                    'best_valid_loss': loss,
                    # 'train_loss': self.train_loss,
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
                if self.early_stop:
                    self.inference(valid_loader, epoch, type='valid')
            else:
                break

    # def train(self, trainset, validset=None, refset=None):
    #     print('==> Start training {}'.format(self.reconstruction_path))
    #     self.train_flag = True
    #     trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.train_batch_size, shuffle=True,
    #                                               num_workers=2)
    #     if self.early_stop:
    #         validloader = torch.utils.data.DataLoader(validset, batch_size=self.train_batch_size, shuffle=True,
    #                                                   num_workers=2)
    #     for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
    #         if self.train_flag:
    #             self.train_epoch(trainloader, epoch)
    #             if self.early_stop:
    #                 self.inference(validloader, epoch, type='valid')
    #         else:
    #             break

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

                    content_z, style_z = self.split_z(z)
                    if reconstruction_type == 'content_z':
                        z = torch.cat((content_z, torch.zeros_like(style_z)), axis=1)
                    elif reconstruction_type == 'style_z':
                        z = torch.cat((torch.zeros_like(content_z), style_z), axis=1)

                    recons_batch = self.decoder(z).cpu()
                    labels_batch = targets
                    if len(recons) == 0:
                        recons = recons_batch
                        labels = labels_batch

                        # save images
                        # vutils.save_image(recons, os.path.join(self.reconstruction_path,
                        #                                        'recon_{}_{}.png'.format(dataset_type,
                        #                                                                 reconstruction_type)),
                        #                   normalize=True, nrow=10)

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

    def split_z(self, z):
        content_z = z[:, 0:self.disc_input_dim]
        style_z = z[:, self.disc_input_dim:]

        return content_z, style_z
