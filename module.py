import torch.nn as nn
import torch
import torch.nn.init as init
from torch.nn import functional as F


def _get_norm_layer_2d(norm):
    if norm == 'none':
        return torchlib.Identity
    elif norm == 'batch_norm':
        return nn.BatchNorm2d
    elif norm == 'instance_norm':
        return functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm == 'layer_norm':
        return lambda num_features: nn.GroupNorm(1, num_features)
    else:
        raise NotImplementedError


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

# class MIAttacker(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()

#         self.net = nn.Sequential(
#             # nn.Linear(input_dim, 128),
#             # nn.BatchNorm1d(128),
#             # nn.LeakyReLU(0.2),
#             nn.Linear(input_dim, int(input_dim / 2)),
#             nn.BatchNorm1d(int(input_dim / 2)),
#             # nn.LeakyReLU(0.2),
#             nn.ReLU(),
#             nn.Linear(int(input_dim / 2), 1),
#             nn.Sigmoid(),
#         )
#         init_layers(self._modules)

#     def forward(self, x):
#         return self.net(x)

class MembershipDiscriminatorImproved(nn.Module):
    def __init__(self, z_dim, class_dim):
        super().__init__()
        self.z_dim = z_dim
        self.class_dim = class_dim

        print('Improved Membership Discriminator')

        self.features = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
        )

        self.labels = nn.Sequential(
            nn.Linear(class_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
        )

        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        init_layers(self._modules)

    def forward(self, x_):
        x = x_[:, 0:self.z_dim]
        y = x_[:, self.z_dim:self.z_dim + self.class_dim]

        x = self.features(x)
        y = self.labels(y)
        return self.net(torch.cat((x, y), dim=1))

class ClassDiscriminatorImproved(nn.Module):
    def __init__(self, z_dim, class_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, class_dim),
        )
        init_layers(self._modules)

    def forward(self, x):
        return self.net(x)

# For black-box attack
class MIAttacker(nn.Module):
    def __init__(self, class_dim):
        super().__init__()
        self.class_dim = class_dim

        self.features = nn.Sequential(
            nn.Linear(class_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
        )

        self.labels = nn.Sequential(
            nn.Linear(class_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
        )

        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        init_layers(self._modules)

    def forward(self, x, y):
        x = self.features(x)
        y = self.labels(y)
        return self.net(torch.cat((x, y), dim=1))

# For white-box attack : depth 1 hard coding
class ConvMIAttacker(nn.Module):
    def __init__(self, input_dim=512, class_dim=10, depth=1):
        super().__init__()

        self.input_dim = input_dim
        self.class_dim = class_dim

        # depth-dependent implementation
        self.features = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
        )

        self.labels = nn.Sequential(
            nn.Linear(class_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        init_layers(self._modules)

    def forward(self, x, y):
        x = self.features(x)
        y = self.labels(y)
        return self.net(torch.cat((x, y), dim=1))




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
        print('Membership Discriminator')

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
            # nn.Sigmoid(),
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

        self.net = nn.Sequential(*layers)

        # 2: logit
        self.d = nn.Sequential(
            nn.Conv2d(d, 1, kernel_size=4, stride=1, padding=0))

    def forward(self, x):
        lh = self.net(x)
        d = self.d(lh)
        return d, lh


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
            nn.ConvTranspose2d(256, num_channels,
                               kernel_size=4, stride=2, padding=1),
            # nn.sigmoid()
        )

        init_layers(self._modules)

    def forward(self, x):
        # return self.main(self.upsample(x).relu().view(-1, 64, 7, 7))
        return self.main(x)


# input noise dimension
nz = 100
# number of generator filters
ngf = 64
# number of discriminator filters
ndf = 64

nc = 3

class Discriminator(nn.Module):
    def __init__(self, ngpu=1):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            # nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(3, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

        init_layers(self._modules)

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(
                self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)
