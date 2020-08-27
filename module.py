import torch.nn as nn
import torch
import torch.nn.init as init
from torch.nn import functional as F


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


class MIAttacker(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            # nn.Linear(input_dim, 128),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(0.2),
            nn.Linear(input_dim, int(input_dim / 2)),
            nn.BatchNorm1d(int(input_dim / 2)),
            # nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.Linear(int(input_dim / 2), 1),
            nn.Sigmoid(),
        )
        init_layers(self._modules)

    def forward(self, x):
        return self.net(x)


class ClassDiscriminator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.LeakyReLU(0.2),
            nn.Linear(128, 32),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.LeakyReLU(0.2),
            nn.Linear(32, output_dim),
        )
        init_layers(self._modules)

    def forward(self, x):
        return self.net(x)


class MembershipDiscriminator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.LeakyReLU(0.2),
            nn.Linear(128, 32),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.LeakyReLU(0.2),
            nn.Linear(32, output_dim),
            nn.Sigmoid(),
        )
        init_layers(self._modules)

    def forward(self, x):
        return self.net(x)


class FCClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.net1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
        )

        self.net2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
        )

        self.net3 = nn.Sequential(
            nn.Linear(128, output_dim),
        )
        init_layers(self._modules)

    def forward(self, x):
        x1 = self.net1(x)
        x2 = self.net2(x1)
        x3 = self.net3(x2)
        return x3

    def extract_features(self, x):
        x1 = self.net1(x)
        x2 = self.net2(x1)
        x3 = self.net3(x2)
        return x1, x2, x3


class FCNClassifierB(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=1),
        )
        init_layers(self._modules)

    def forward(self, x):
        return self.net(x)


class ConvClassifier(nn.Module):
    def __init__(self):
        super(ConvClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        # return output
        return x


class VAEFCEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.mu = nn.Linear(input_dim, latent_dim)
        self.logvar = nn.Linear(input_dim, latent_dim)

        # self.main = nn.Sequential(
        #     nn.Linear(input_dim, 2 * latent_dim, bias=True),
        #     nn.BatchNorm1d(2 * latent_dim),
        #     nn.ReLU(),
        #     # nn.ReLU(True),
        #     # nn.Linear(input_dim, latent_dim, bias=True),
        #     # nn.ReLU(True),
        # )

        init_layers(self._modules)

    def forward(self, x):
        # x = self.main(x)
        return self.mu(x), self.logvar(x)


class FCDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.main = nn.Sequential(
            # nn.Linear(latent_dim, 2 * latent_dim, bias=True),
            # nn.BatchNorm1d(2 * latent_dim),
            # nn.ReLU(True),
            nn.Linear(latent_dim, input_dim, bias=True),
            # nn.Sigmoid(),
            # nn.ReLU(True),
        )

        init_layers(self._modules)

    def forward(self, x):
        return self.main(x)


# class VAEFCDecoder(nn.Module):
#     def __init__(self, input_dim, latent_dim):
#         super().__init__()
#
#         self.main = nn.Sequential(
#             nn.Linear(latent_dim, input_dim, bias=True),
#             nn.Sigmoid(),
#         )
#
#         init_layers(self._modules)
#
#     def forward(self, x):
#         return self.main(x)


class VAEConvEncoder(nn.Module):
    def __init__(self, latent_dim, num_channels):
        super().__init__()

        self.mu = nn.Linear(2304, latent_dim)
        self.logvar = nn.Linear(2304, latent_dim)
        # self.mu = nn.Linear(64 * 7 * 7, latent_dim)
        # self.logvar = nn.Linear(64 * 7 * 7, latent_dim)

        self.main = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Sigmoid(),
            # nn.Tanh(),
            # nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Sigmoid(),
            # nn.Tanh(),
            # nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Sigmoid(),
            # nn.Tanh(),
            # nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=2, stride=2),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.Sigmoid(),
            # nn.Tanh(),
            # nn.LeakyReLU(0.2),
            Flatten3D(),
        )

        init_layers(self._modules)

    def forward(self, x):
        x = self.main(x)
        return self.mu(x), self.logvar(x)


class UnFlatten(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, input):  # latent dim : 64
        return input.view(input.size(0), self.size, 1, 1)


class VAEConvDecoder(nn.Module):
    def __init__(self, latent_dim, num_channels):
        super().__init__()
        # self.upsample = nn.Linear(latent_dim, 64 * 7 * 7)
        self.main = nn.Sequential(
            UnFlatten(latent_dim),
            nn.ConvTranspose2d(latent_dim, 1024, kernel_size=4, stride=1),
            # nn.BatchNorm2d(1024),
            nn.ReLU(),
            # nn.Sigmoid(),
            # nn.Tanh(),
            # nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.Sigmoid(),
            # nn.Tanh(),
            # nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.Sigmoid(),
            # nn.Tanh(),
            # nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, num_channels, kernel_size=4, stride=2, padding=1),
            # nn.sigmoid()
        )

        init_layers(self._modules)

    def forward(self, x):
        # return self.main(self.upsample(x).relu().view(-1, 64, 7, 7))
        return self.main(x)
