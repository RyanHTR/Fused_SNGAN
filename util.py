import matplotlib.pyplot as plt
import itertools
import torch
import os
import numpy as np
from torch.autograd import Variable

TEST_ROW = 5
DIM_NOISE = 128

fixed_z_ = torch.randn((TEST_ROW * TEST_ROW, DIM_NOISE))  # fixed noise
fixed_z_ = Variable(fixed_z_.cuda(), volatile=True)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def show_result(num_epoch, Gs, Gc, show=False, save=False, path='result.png', isFix=False):
    z_ = torch.randn((TEST_ROW*TEST_ROW, DIM_NOISE))
    z_ = Variable(z_.cuda(), volatile=True)

    y_c = []
    for i in range(TEST_ROW):
        y_c.append([i] * TEST_ROW)
    y_idx = np.array(y_c, dtype=np.int32)
    y_idx = y_idx.reshape(TEST_ROW*TEST_ROW)
    y_c = np.eye(5, dtype=np.float32)[y_idx]
    y_c = torch.from_numpy(y_c)
    y_c_ = Variable(y_c.cuda(), volatile=True)

    Gc.eval()
    if isFix:
        test_Ms = Gs(fixed_z_)
    else:
        test_Ms = Gs(z_)
    test_x_c_fake = Gc(test_Ms, y_c_)
    Gc.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow((test_x_c_fake[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def show_train_hist(hist, show=False, save=False, path='Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()