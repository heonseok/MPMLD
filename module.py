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
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        init_layers(self._modules)

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
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


class FCNClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.net = nn.Sequential(
            # nn.Linear(input_dim, 1024),
            # nn.LeakyReLU(0.2),
            # nn.Linear(1024, 512),
            # nn.LeakyReLU(0.2),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            # nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.LeakyReLU(0.2),
            nn.Linear(128, output_dim),
            # nn.Softmax(dim=1),
        )
        init_layers(self._modules)

    def forward(self, x):
        return self.net(x)


# class FCNClassifier(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, output_dim),
#             # nn.Softmax(dim=1),
#         )
#         init_layers(self._modules)
#
#     def forward(self, x):
#         return self.net(x)


# class FCNClassifier(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.BatchNorm1d(128),
#             nn.LeakyReLU(0.2),
#             nn.Linear(128, 32),
#             nn.BatchNorm1d(32),
#             nn.LeakyReLU(0.2),
#             nn.Linear(32, output_dim),
#         )
#
#     def forward(self, x):
#         return self.net(x)


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


class AEConvEncoder(nn.Module):
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


class AEConvDecoder(nn.Module):
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


class FCNEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(input_dim, latent_dim, bias=True),
        )

        init_layers(self._modules)

    def forward(self, x):
        return self.main(x)


class FCNDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(latent_dim, input_dim, bias=True),
        )

        init_layers(self._modules)

    def forward(self, x):
        return self.main(x)

# class FCNEncoder(nn.Module):
#     def __init__(self, input_dim, latent_dim):
#         super().__init__()
#
#         self.main = nn.Sequential(
#             nn.Linear(input_dim, 2 * latent_dim, bias=True),
#             nn.ReLU(True),
#             nn.Linear(2 * latent_dim, latent_dim, bias=True),
#             nn.ReLU(True),
#         )
#
#         init_layers(self._modules)
#
#     def forward(self, x):
#         return self.main(x)
#
#
# class FCNDecoder(nn.Module):
#     def __init__(self, input_dim, latent_dim):
#         super().__init__()
#
#         self.main = nn.Sequential(
#             nn.Linear(latent_dim, 2 * latent_dim, bias=True),
#             nn.ReLU(True),
#             nn.Linear(2 * latent_dim, input_dim, bias=True),
#             nn.ReLU(True),
#         )
#
#         init_layers(self._modules)
#
#     def forward(self, x):
#         return self.main(x)

class VAEFCNEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.mu = nn.Linear(2 * latent_dim, latent_dim)
        self.logvar = nn.Linear(2 * latent_dim, latent_dim)

        self.main = nn.Sequential(
            nn.Linear(input_dim, 2 * latent_dim, bias=True),
            nn.ReLU(),
            # nn.ReLU(True),
            # nn.Linear(input_dim, latent_dim, bias=True),
            # nn.ReLU(True),
        )

        init_layers(self._modules)

    def forward(self, x):
        x = self.main(x)
        return self.mu(x), self.logvar(x)


class VAEFCNDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(latent_dim, input_dim, bias=True),
            nn.Sigmoid(),
        )

        init_layers(self._modules)

    def forward(self, x):
        return self.main(x)


class VAEConvEncoder(nn.Module):
    def __init__(self, latent_dim, num_channels):
        super().__init__()

        self.mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.logvar = nn.Linear(64 * 7 * 7, latent_dim)

        self.main = nn.Sequential(
            nn.Conv2d(num_channels, 32, 4, 1, 2),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  7, 7
            nn.ReLU(True),
            Flatten3D(),
        )

        init_layers(self._modules)

    def forward(self, x):
        x = self.main(x)
        return self.mu(x), self.logvar(x)


class VAEConvDecoder(nn.Module):
    def __init__(self, latent_dim, num_channels):
        super().__init__()
        self.upsample = nn.Linear(latent_dim, 64 * 7 * 7)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  64,  14,  14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, 1),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 1, 2),  # B, 1, 28, 28
            nn.Sigmoid(),
        )

        init_layers(self._modules)

    def forward(self, x):
        return self.main(self.upsample(x).relu().view(-1, 64, 7, 7))
        # return self.main(x)


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
