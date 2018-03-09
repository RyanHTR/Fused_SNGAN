import os, time, sys
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torch.autograd import Variable

from res_model import *
from util import *

# training parameters
NOISE_DIM = 128
BATCH_SIZE = 8
LR = 2e-4
TRAIN_EPOCHS = 100
LR_DECAY_EPOCH = 50
ITERS_PER_EPOCH = 130
TRAIN_ITER_DU = 5
TRAIN_ITER_DC = 5

# data_loader
CELEBA_DATA_DIR = '/home/huitr/S-Lab/DATASET/CelebA/CelebA/Img'
MAKEUP_DATA_DIR = '/home/huitr/S-Lab/DATASET/makeup-data5.0-big-YZ-v2/train'
IMG_SIZE = 256
NUM_CLASSES = 5
IS_CROP = False

if IS_CROP:
    transform = transforms.Compose([
        transforms.Scale(108),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
else:
    transform = transforms.Compose([
        transforms.Scale(IMG_SIZE),
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

celebA_folder = datasets.ImageFolder(CELEBA_DATA_DIR, transform)
train_loader_celebA = torch.utils.data.DataLoader(celebA_folder, batch_size=BATCH_SIZE, shuffle=True)

makeup_folder = datasets.ImageFolder(MAKEUP_DATA_DIR, transform)
train_loader_makeup = torch.utils.data.DataLoader(makeup_folder, batch_size=BATCH_SIZE, shuffle=True)

# temp = plt.imread(train_loader_celebA.dataset.imgs[0][0])

# if (temp.shape[0] != IMG_SIZE) or (temp.shape[1] != IMG_SIZE):
#     sys.stderr.write('Error! image size is not 256 x 256! run \"celebA_data_preprocess.py\" !!!')
#     sys.exit(1)

# network
Gs = GeneratorStructural()
Gu = GeneratorUnconditional()
Gc = GeneratorConditional()
Du = DiscriminatorUnconditional()
Dc = DiscriminatorConditional()

Gs.weight_init()
Gu.weight_init()
Gc.weight_init()
Du.weight_init()
Dc.weight_init()

Gs.cuda()
Gu.cuda()
Gc.cuda()
Du.cuda()
Dc.cuda()

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
Gs_optimizer = optim.Adam(Gs.parameters(), lr=LR, betas=(0., 0.9))
Gu_optimizer = optim.Adam(Gu.parameters(), lr=LR, betas=(0., 0.9))
Du_optimizer = optim.Adam(filter(lambda p: p.requires_grad, Du.parameters()), lr=LR, betas=(0., 0.9))

Gc_optimizer = optim.Adam(Gc.parameters(), lr=LR, betas=(0., 0.9))
Dc_optimizer = optim.Adam(filter(lambda p: p.requires_grad, Dc.parameters()), lr=LR, betas=(0., 0.9))

# results save folder
if not os.path.isdir('Fused_SNGAN_results'):
    os.mkdir('Fused_SNGAN_results')
if not os.path.isdir('Fused_SNGAN_results/Random_results'):
    os.mkdir('Fused_SNGAN_results/Random_results')
if not os.path.isdir('Fused_SNGAN_results/Fixed_results'):
    os.mkdir('Fused_SNGAN_results/Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

print('Training start!')
start_time = time.time()
for epoch in range(TRAIN_EPOCHS):
    Du_losses = []
    Gu_losses = []

    Dc_losses = []
    Gc_losses = []

    Du_iter_losses = []
    Gu_iter_losses = []

    Dc_iter_losses = []
    Gc_iter_losses = []

    if (epoch+1) == LR_DECAY_EPOCH:
        Gs_optimizer.param_groups[0]['lr'] /= 10
        Gu_optimizer.param_groups[0]['lr'] /= 10
        Du_optimizer.param_groups[0]['lr'] /= 10
        Gc_optimizer.param_groups[0]['lr'] /= 10
        Dc_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    num_iter = 0
    epoch_start_time = time.time()
    for i in range(ITERS_PER_EPOCH):

        x_uc_real, _ = train_loader_celebA.__iter__().__next__()

        # train unconditional discriminator Du
        for _ in range(TRAIN_ITER_DU):
            Du.zero_grad()

            if IS_CROP:
                x_uc_real = x_uc_real[:, :, 22:86, 22:86]

            y_real_ = torch.ones(BATCH_SIZE)
            y_fake = torch.zeros(BATCH_SIZE)

            x_uc_real_, y_real_, y_fake_ = Variable(x_uc_real.cuda()), Variable(y_real_.cuda()), Variable(y_fake.cuda())

            Du_real = Du(x_uc_real_).squeeze()
            # Du_real_loss = BCE_loss(Du_real, y_real_)
            Du_real_loss = torch.mean(torch.max(y_fake_, y_real_ - Du_real))

            z_ = torch.randn((BATCH_SIZE, NOISE_DIM))
            z_ = Variable(z_.cuda())
            Ms = Gs(z_)
            x_u_fake = Gu(Ms)

            Du_fake = Du(x_u_fake).squeeze()
            # Du_fake_loss = BCE_loss(Du_fake, y_fake_)
            Du_fake_loss = torch.mean(torch.max(y_fake_, y_real_ + Du_fake))

            Du_train_loss = Du_real_loss + Du_fake_loss

            Du_train_loss.backward(retain_graph=True)
            Du_optimizer.step()

            Du_losses.append(Du_train_loss.data[0])
            Du_iter_losses.append(Du_train_loss.data[0])

        # train unconditional generator Gu & Gs
        Gu.zero_grad()
        Gs.zero_grad()

        # TODO: not use new noise vector z?

        # Gu_train_loss = BCE_loss(Du_fake, y_real_)
        Gu_train_loss = -torch.mean(Du_fake)
        Gu_train_loss.backward(retain_graph=True)

        Gu_optimizer.step()
        Gs_optimizer.step()

        Gu_losses.append(Gu_train_loss.data[0])
        Gu_iter_losses.append(Gu_train_loss.data[0])

        # ==================================================== #

        x_c_real, cls_dirs = train_loader_makeup.__iter__().__next__()

        cls_dirs = cls_dirs.numpy()
        labels = np.eye(NUM_CLASSES, dtype=np.float32)[cls_dirs]
        y_c = torch.from_numpy(labels)

        x_c_real_, y_c_ = Variable(x_c_real.cuda()), Variable(y_c.cuda())

        # train conditional discriminator Dc
        for _ in range(TRAIN_ITER_DC):
            Dc.zero_grad()

            Dc_real = Dc(x_c_real_, y_c_).squeeze()
            # Dc_real_loss = BCE_loss(Dc_real, y_real_)
            Dc_real_loss = torch.mean(torch.max(y_fake_, y_real_ - Dc_real))

            x_c_fake = Gc(Ms, y_c_)
            Dc_fake = Dc(x_c_fake, y_c_).squeeze()
            # Dc_fake_loss = BCE_loss(Dc_fake, y_fake_)
            Dc_fake_loss = torch.mean(torch.max(y_fake_, y_real_ + Dc_fake))

            Dc_train_loss = Dc_real_loss + Dc_fake_loss

            Dc_train_loss.backward(retain_graph=True)
            Dc_optimizer.step()

            Dc_losses.append(Dc_train_loss.data[0])
            Dc_iter_losses.append(Dc_train_loss.data[0])

        # train conditional generator Gc
        Gc.zero_grad()

        # Gc_train_loss = BCE_loss(Dc_fake, y_real_)
        Gc_train_loss = -torch.mean(Dc_fake)
        Gc_train_loss.backward(retain_graph=True)

        Gc_optimizer.step()

        Gc_losses.append(Gc_train_loss.data[0])
        Gc_iter_losses.append(Gc_train_loss.data[0])

        num_iter += 1
        if num_iter % 10 == 0:
            print('Epoch/Iter: [%d/%d] - loss Du: %.3f, loss Gu: %.3f loss Dc: %.3f loss Gc: %.3f' %
                  (epoch + 1, num_iter, torch.mean(torch.FloatTensor(Du_iter_losses)),
                   torch.mean(torch.FloatTensor(Gu_iter_losses)), torch.mean(torch.FloatTensor(Dc_iter_losses)),
                   torch.mean(torch.FloatTensor(Gc_iter_losses))))
            Du_iter_losses.clear()
            Gu_iter_losses.clear()
            Dc_iter_losses.clear()
            Gc_iter_losses.clear()

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('Epoch/Total Epoch: [%d/%d] - ptime: %.2f, loss Du: %.3f, loss Gu: %.3f loss Dc: %.3f loss Gc: %.3f\n' %
          ((epoch + 1), TRAIN_EPOCHS, per_epoch_ptime, torch.mean(torch.FloatTensor(Du_losses)),
           torch.mean(torch.FloatTensor(Gu_losses)), torch.mean(torch.FloatTensor(Dc_losses)),
           torch.mean(torch.FloatTensor(Gc_losses))))

    p = 'Fused_SNGAN_results/Random_results/Fused_SNGAN_' + str(epoch + 1) + '.png'
    fixed_p = 'Fused_SNGAN_results/Fixed_results/Fused_SNGAN_' + str(epoch + 1) + '.png'
    try:
        show_result((epoch + 1), Gs, Gc, save=True, path=p, isFix=False)
        show_result((epoch + 1), Gs, Gc, save=True, path=fixed_p, isFix=True)
    except Exception as e:
        print('Some errors occur during showing results.')
        print(e)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(Du_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(Gu_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), TRAIN_EPOCHS, total_ptime))
print("Training finish!... save training results")
torch.save(Gs.state_dict(), "Fused_SNGAN_results/Gs_param.pkl")
torch.save(Gu.state_dict(), "Fused_SNGAN_results/Gu_param.pkl")
torch.save(Du.state_dict(), "Fused_SNGAN_results/Du_param.pkl")
torch.save(Gc.state_dict(), "Fused_SNGAN_results/Gc_param.pkl")
torch.save(Dc.state_dict(), "Fused_SNGAN_results/Dc_param.pkl")
with open('Fused_SNGAN_results/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path='Fused_SNGAN_results/Fused_SNGAN_train_hist.png')

try:
    images = []
    for e in range(TRAIN_EPOCHS):
        img_name = 'Fused_SNGAN_results/Fixed_results/Fused_SNGAN_' + str(e + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave('Fused_SNGAN_results/generation_animation.gif', images, fps=5)
except Exception as e:
    print('Some errors occur during generating GIF.')
    print(e)

