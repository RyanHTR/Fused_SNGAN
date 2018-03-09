import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from spectral_norm import spectral_norm as SpectralNorm

NUM_CLASSES = 5


def Conv2dSN(in_channels, out_channels, kernel_size=3, stride=1, padding=1, spectral_norm=False):
    if spectral_norm:
        return SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)


class Normalize(nn.Module):
    '''
    Choose between InstanceNorm and Identity
    '''
    def __init__(self, name, in_channels):
        super(Normalize, self).__init__()
        if name == 'D':
            self.norm = Identity()
        else:
            self.norm = nn.InstanceNorm2d(in_channels, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return x


class UpsampleConv(nn.Module):
    '''
    Upsample tensor and then conv
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(UpsampleConv, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class DownsampleConv(nn.Module):
    '''
    Conv the tensor first and then downsample
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DownsampleConv, self).__init__()
        self.downsample = nn.AvgPool2d(2, stride=2)
        self.conv = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))

    def forward(self, x):
        x = self.conv(x)
        x = self.downsample(x)
        return x


class Identity(nn.Module):
    '''
    Return the input directly
    '''

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ResidualBlock(nn.Module):
    '''
    Residual Block
    3 mode: Upsample, Downsample, Normal
    '''

    def __init__(self, name, in_channels, out_channels, resample=None, spectral_norm=False):
        super(ResidualBlock, self).__init__()

        if resample == 'up':
            self.conv1 = UpsampleConv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.bn1 = Normalize(name, out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.bn2 = Normalize(name, out_channels)
            self.conv_shortcut = UpsampleConv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        elif resample == 'down':
            self.conv1 = Conv2dSN(in_channels, in_channels, kernel_size=3, stride=1, padding=1, spectral_norm=spectral_norm)
            self.bn1 = Normalize(name, in_channels)
            self.conv2 = DownsampleConv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.bn2 = Normalize(name, out_channels)
            self.conv_shortcut = DownsampleConv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        else:
            self.conv1 = Conv2dSN(in_channels, out_channels, kernel_size=3, stride=1, padding=1, spectral_norm=spectral_norm)
            self.bn1 = Normalize(name, out_channels)
            self.conv2 = Conv2dSN(out_channels, out_channels, kernel_size=3, stride=1, padding=1, spectral_norm=spectral_norm)
            self.bn2 = Normalize(name, out_channels)
            if in_channels == out_channels:
                self.conv_shortcut = Identity()
            else:
                self.conv_shortcut = Conv2dSN(in_channels, out_channels, kernel_size=1, stride=1, padding=0, spectral_norm=spectral_norm)

        self.relu = nn.ReLU()
        self.shortcut = self.conv_shortcut

    def forward(self, x):
        output = x
        output = self.shortcut(output)
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        output = residual + output
        output = self.relu(output)
        return output


class GeneratorStructural(nn.Module):
    '''
    Gs produces structural prior
    '''

    # initializers
    def __init__(self):
        super(GeneratorStructural, self).__init__()
        self.fc1 = nn.Linear(128, 4 * 4 * 1024)
        self.fc1_bn = nn.InstanceNorm2d(1024)
        self.res_block1 = ResidualBlock('G', 1024, 512, resample='up')
        self.relu = nn.ReLU()

    # weight_init
    def weight_init(self):
        for m in self._modules:
            normal_init(self._modules[m])

    # forward method
    def forward(self, z):
        x = self.fc1(z)
        x = x.view(-1, 1024, 4, 4)
        x = self.relu(self.fc1_bn(x))
        x = self.res_block1(x)
        return x


class GeneratorUnconditional(nn.Module):
    '''
    Gu produces unconditional images
    '''

    # initializers
    def __init__(self):
        super(GeneratorUnconditional, self).__init__()
        self.res_block1 = ResidualBlock('G', 512, 256, resample='up')
        self.res_block2 = ResidualBlock('G', 256, 128, resample='up')
        self.res_block3 = ResidualBlock('G', 128, 64, resample='up')
        self.res_block4 = ResidualBlock('G', 64, 32, resample='up')
        self.res_block5 = ResidualBlock('G', 32, 16, resample='up')
        self.bn1 = nn.InstanceNorm2d(16)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    # weight_init
    def weight_init(self):
        for m in self._modules:
            normal_init(self._modules[m])

    # forward method
    def forward(self, Ms):
        x = self.res_block1(Ms)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.tanh(x)
        return x


class GeneratorConditional(nn.Module):
    '''
    Gc produces conditional images
    '''

    # initializers
    def __init__(self):
        super(GeneratorConditional, self).__init__()
        self.fc1 = nn.Linear(NUM_CLASSES, NUM_CLASSES)
        self.res_block1 = ResidualBlock('G', 512 + NUM_CLASSES, 256, resample='up')
        self.res_block2 = ResidualBlock('G', 256, 128, resample='up')
        self.res_block3 = ResidualBlock('G', 128, 64, resample='up')
        self.res_block4 = ResidualBlock('G', 64, 32, resample='up')
        self.res_block5 = ResidualBlock('G', 32, 16, resample='up')
        self.bn1 = nn.InstanceNorm2d(16)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    # weight_init
    def weight_init(self):
        for m in self._modules:
            normal_init(self._modules[m])

    # forward method
    def forward(self, Ms, y):
        gaussian_y = self.fc1(y)
        My = gaussian_y.repeat(8, 8, 1, 1)
        My.transpose_(2, 1).transpose_(1, 0)
        My.transpose_(3, 2).transpose_(2, 1)
        M_cat = torch.cat([My, Ms], 1)
        x = self.res_block1(M_cat)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.tanh(x)
        return x


class DiscriminatorUnconditional(nn.Module):
    '''
    Du for unconditional images
    '''

    # initializers
    def __init__(self):
        super(DiscriminatorUnconditional, self).__init__()
        self.res_block1 = ResidualBlock('D', 3, 64, resample='down', spectral_norm=True)
        self.res_block2 = ResidualBlock('D', 64, 128, resample='down', spectral_norm=True)
        self.res_block3 = ResidualBlock('D', 128, 256, resample='down', spectral_norm=True)
        self.res_block4 = ResidualBlock('D', 256, 512, resample='down', spectral_norm=True)
        self.res_block5 = ResidualBlock('D', 512, 1024, resample='down', spectral_norm=True)
        self.res_block6 = ResidualBlock('D', 1024, 1024, spectral_norm=True)
        self.relu = nn.ReLU()
        self.fc1 = SpectralNorm(nn.Linear(1024, 1))

    # weight_init
    def weight_init(self):
        for m in self._modules:
            normal_init(self._modules[m])

    # forward method
    def forward(self, x_g):
        x = self.res_block1(x_g)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.res_block6(x)
        x = self.relu(x)
        x = torch.sum(torch.sum(x, 3), 2)
        x = self.fc1(x)
        return x


class DiscriminatorConditional(nn.Module):
    '''
    Dc for conditional images
    '''

    # initializers
    def __init__(self):
        super(DiscriminatorConditional, self).__init__()
        self.res_block1 = ResidualBlock('D', 3, 64, resample='down', spectral_norm=True)
        self.res_block2 = ResidualBlock('D', 64, 128, resample='down', spectral_norm=True)
        self.res_block3 = ResidualBlock('D', 128, 256, resample='down', spectral_norm=True)
        self.res_block4 = ResidualBlock('D', 256, 512, resample='down', spectral_norm=True)
        self.res_block5 = ResidualBlock('D', 512 + NUM_CLASSES, 1024, resample='down', spectral_norm=True)
        self.res_block6 = ResidualBlock('D', 1024, 1024, spectral_norm=True)
        self.relu = nn.ReLU()
        self.fc1 = SpectralNorm(nn.Linear(1024, 1))

    # weight_init
    def weight_init(self):
        for m in self._modules:
            normal_init(self._modules[m])

    # forward method
    def forward(self, x_g, y):
        y = y.repeat(16, 16, 1, 1)
        y.transpose_(2, 1).transpose_(1, 0)
        y.transpose_(3, 2).transpose_(2, 1)
        x = self.res_block1(x_g)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = torch.cat([x, y], 1)
        x = self.res_block5(x)
        x = self.res_block6(x)
        x = self.relu(x)
        x = torch.sum(torch.sum(x, 3), 2)
        x = self.fc1(x)
        return x


def normal_init(m):
    _weight_init(m)


def _weight_init(m):
    # 使用isinstance来判断m属于什么类型
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        # m中的weight，bias其实都是Variable，为了能学习参数以及后向传播
        m.weight.data.fill_(1)
        m.bias.data.zero_()
