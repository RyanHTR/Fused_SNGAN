import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 5

class GeneratorStructural(nn.Module):
    '''
    Gs produces structural prior
    '''

    # initializers
    def __init__(self):
        super(GeneratorStructural, self).__init__()
        self.fc1 = nn.Linear(128, 4 * 4 * 1024)
        self.fc1_bn = nn.BatchNorm2d(4 * 4 * 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv1_bn = nn.BatchNorm2d(512)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, z):
        x = F.relu(self.fc1_bn(self.fc1(z)))
        x = x.view(-1, 1024, 4, 4)
        x = F.relu(self.deconv1_bn(self.deconv1(x)))

        return x


class GeneratorUnconditional(nn.Module):
    '''
    Gu produces unconditional images
    '''

    # initializers
    def __init__(self):
        super(GeneratorUnconditional, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv1_bn = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv2_bn = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 3, kernel_size=5, stride=2, padding=2, output_padding=1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, Ms):
        x = F.relu(self.deconv1_bn(self.deconv1(Ms)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.tanh(self.deconv3(x))

        return x


class GeneratorConditional(nn.Module):
    '''
    Gc produces conditional images
    '''

    # initializers
    def __init__(self):
        super(GeneratorConditional, self).__init__()
        self.fc1 = nn.Linear(NUM_CLASSES, NUM_CLASSES)
        self.deconv1 = nn.ConvTranspose2d(512 + NUM_CLASSES, 256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv1_bn = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv2_bn = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 3, kernel_size=5, stride=2, padding=2, output_padding=1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, Ms, y):
        gaussian_y = self.fc1(y)
        My = gaussian_y.repeat(8, 8, 1, 1)
        My.transpose_(2, 1).transpose_(1, 0)
        My.transpose_(3, 2).transpose_(2, 1)
        M_cat = torch.cat([My, Ms], 1)
        x = F.relu(self.deconv1_bn(self.deconv1(M_cat)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.tanh(self.deconv3(x))

        return x


class DiscriminatorUnconditional(nn.Module):
    '''
    Du for unconditional images
    '''

    # initializers
    def __init__(self, d=128):
        super(DiscriminatorUnconditional, self).__init__()
        self.conv1 = nn.Conv2d(3, d, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(d, d * 2, kernel_size=4, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, kernel_size=4, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, kernel_size=4, stride=2, padding=1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, kernel_size=4, stride=1, padding=0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x_g):
        x = F.leaky_relu(self.conv1(x_g), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x


class DiscriminatorConditional(nn.Module):
    '''
    Dc for conditional images
    '''

    # initializers
    def __init__(self, d=128):
        super(DiscriminatorConditional, self).__init__()
        self.conv1 = nn.Conv2d(3, d, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(d, d * 2, kernel_size=4, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, kernel_size=4, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4 + NUM_CLASSES, d * 8, kernel_size=4, stride=2, padding=1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, kernel_size=4, stride=1, padding=0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x_g, y):
        y = y.repeat(8, 8, 1, 1)
        y.transpose_(2, 1).transpose_(1, 0)
        y.transpose_(3, 2).transpose_(2, 1)
        x = F.leaky_relu(self.conv1(x_g), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
