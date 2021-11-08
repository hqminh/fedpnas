import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pylab
import pickle
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
from torch.distributions import Normal, Gumbel, Uniform
import traceback
from sklearn.model_selection import train_test_split, KFold
from tqdm import trange, tqdm
from time import sleep
from copy import deepcopy
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from SNAS.operations import *

N_BASE_CELL = 3
N_SPEC_CELL = 1
BATCH_SIZE = 48

np.set_printoptions(precision=3)


def lr_decay(global_step, init_learning_rate=5e-4, min_learning_rate=1e-5, decay_rate=0.9996):
    lr = ((init_learning_rate - min_learning_rate) *
          pow(decay_rate, global_step) +
          min_learning_rate)
    return lr


def cuda_memory(device):
    t = torch.cuda.get_device_properties(device).total_memory
    c = torch.cuda.memory_cached(device)
    a = torch.cuda.memory_allocated(device)
    return f'Total:{t}, Cached:{c}, Alloc:{a}, Free:{t-c-a}'


def print_cuda_memory(device, prefix=''):
    print(prefix + cuda_memory(device))


def freeze(net):
    if isinstance(net, nn.Module):
        for p in net.parameters():
            p.requires_grad = False


def unfreeze(net):
    if isinstance(net, nn.Module):
        for p in net.parameters():
            p.requires_grad = True


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()


def get_data_slice(torch_dataset, start, end):
    X = torch.stack([torch_dataset[i][0] for i in range(start, end)])
    Y = [torch_dataset[i][1] for i in range(start, end)]
    return X, Y


def compute_accuracy(X, Y):
    return 1.0 * torch.sum(torch.eq(torch.argmax(X, dim=1), Y)) / X.shape[0]


def main():
    degrees = [-30, -15, 0, 15, 30]
    spec_xform = [
        transforms.RandomRotation(degrees=(d, d)) for d in degrees
    ]
    fig = plt.figure(figsize=(5, 1.5))
    for i in range(5):
        transform = transforms.Compose([
            transforms.Pad((2, 2, 2, 2)),
            spec_xform[i]])
        trainset = torchvision.datasets.MNIST(root=f'./data/client-{i + 1}/', train=True, download=True,
                                              transform=transform)
        fig.add_subplot(1, 5, i + 1)
        plt.imshow(trainset[0][0])
        plt.tight_layout()
    plt.savefig('./plot/example.png')


def main2():
    total_params = 0
    for op_name, op in FULL_OPS.items():
        print(op_name)
        op_norm = op(C=16, stride=1, affine=True)
        n_param_norm = 0
        for p in op_norm.parameters():
            n_param_norm += p.numel()
        total_params += n_param_norm
        print(n_param_norm)
    print(total_params, total_params * 32)


if __name__ == '__main__':
    main2()