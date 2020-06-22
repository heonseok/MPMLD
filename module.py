import torch.nn as nn
import torch
import torch.nn.init as init


def _init_layer(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
    if isinstance(m, torch.nn.Linear):
        init.kaiming_normal_(m.weight.data)


def init_layers(modules):
    for block in modules:
        from collections.abc import Iterable
        if isinstance(modules[block], Iterable):
            for m in modules[block]:
                _init_layer(m)
        else:
            _init_layer(modules[block])


class Flatten3D(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class Unsqueeze3D(nn.Module):
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        return x


class SimpleNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__()
        # self._latent_dim = latent_dim
        # self._num_channels = num_channels
        # self._image_size = image_size

        # assert image_size == 64, 'This model only works with image size 64x64.'

        self.main = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.ReLU(True),
            Flatten3D(),
            nn.Linear(256, latent_dim, bias=True)
        )

        init_layers(self._modules)

    def forward(self, x):
        return self.main(x)

    # def latent_dim(self):
    #     return self._latent_dim
    #
    # def num_channels(self):
    #     return self._num_channels
    #
    # def image_size(self):
    #     return self._image_size


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__()
        # self.latent_dim = latent_dim
        # self.num_channels = num_channels
        # self.image_size = image_size

        # assert image_size == 64, 'This model only works with image size 64x64.'

        self.main = nn.Sequential(
            Unsqueeze3D(),
            nn.Conv2d(latent_dim, 256, 1, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 3, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, 3, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 3, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, num_channels, 2, 1)
        )
        # output shape = bs x 3 x 64 x 64

        init_layers(self._modules)

    def forward(self, x):
        return self.main(x)

    # def init_layers(self):
    #     for block in self._modules:
    #         for m in self._modules[block]:
    #             if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    #                 init.xavier_normal_(m.weight.data)
    #             if isinstance(m, nn.Linear):
    #                 init.kaiming_normal_(m.weight.data)
