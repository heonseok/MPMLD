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


class SimpleDiscriminator(nn.Module):
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


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, output_dim=10):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ConvEncoderAE(nn.Module):
    def __init__(self, latent_dim, num_channels):
        super().__init__()

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
            nn.Linear(256, latent_dim, bias=True),
            # nn.ReLU(True),
        )

        init_layers(self._modules)

    def forward(self, x):
        return self.main(x)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, num_channels):
        super().__init__()

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
            nn.ConvTranspose2d(64, num_channels, 2, 1),
            # nn.Sigmoid(),
        )

        init_layers(self._modules)

    def forward(self, x):
        return self.main(x)


class ConvEncoderVAE(nn.Module):
    def __init__(self, latent_dim, num_channels):
        super().__init__()

        self.fc1 = nn.Linear(256, latent_dim, bias=True)
        self.fc2 = nn.Linear(256, latent_dim, bias=True)

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
            # nn.Linear(256, latent_dim, bias=True),
            # nn.ReLU(True),
        )

        init_layers(self._modules)

    def forward(self, x):
        x = self.main(x)
        return self.fc1(x), self.fc2(x)


class ConvDecoderVAE(nn.Module):
    def __init__(self, latent_dim, num_channels):
        super().__init__()

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
            nn.ConvTranspose2d(64, num_channels, 2, 1),
            nn.Sigmoid(),
        )

        init_layers(self._modules)

    def forward(self, x):
        return self.main(x)
