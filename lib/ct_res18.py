'''
Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
Source code will be released under Apache-2.0 license if paper accepted.
'''
import os
import torch as th
import collections
from . import cifar10, base, classify
import yaml
from tqdm import tqdm
from termcolor import cprint, colored
from .utils import IMstd, IMmean, renorm, denorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as V
from .attacks import projectedGradientDescent


class Model(base.Model):
    def __init__(self, *, dataparallel=False):
        super(Model, self).__init__()
        self.dataparallel = dataparallel
        self.net = V.models.resnet18(False)
        self.net.fc = th.nn.Linear(512, 10)
        if dataparallel:
            cprint(f"X: _DATA_PARALLEL | {torch.cuda.device_count()} GPUs!", 'red')
            self.net = th.nn.DataParallel(self.net)
    def forward(self, x):
        return self.net(x)

    def getloader(self, kind:str='train', batchsize:int=1):
        '''
        get corresponding dataloaders
        '''
        config = yaml.load(open('config.yml', 'r').read(), Loader=yaml.SafeLoader)
        if kind == 'train':
            return cifar10.get_loader(os.path.expanduser(config['cifar10']['path']), batchsize, 'train')
        else:
            return cifar10.get_loader(os.path.expanduser(config['cifar10']['path']), batchsize, 'test')

    def loss(self, x, y, *, adv=False):
        device = self.net.conv1.weight.device if not self.dataparallel else self.net.module.conv1.weight.device
        images, labels = x.to(device), y.to(device).view(-1)
        if not adv:
            output = self.forward(images)
            loss = th.nn.functional.cross_entropy(output, labels)
            return output, loss
        else:
            # baseline: forward original samples
            self.eval()
            with th.no_grad():
                output, loss = self.loss(images, labels, adv=False)
                acc_orig = (output.max(1)[1].eq(labels).sum().item()/len(y))
                loss_orig = loss.detach()
            # start PGD attack
            # (8/255 ~ 0.031) https://arxiv.org/pdf/1706.06083.pdf
            xr, r = projectedGradientDescent(self, images, labels,
                    eps=0.03, alpha=2./255., maxiter=7, verbose=False,
                    device=device, targeted=False, unbound=False, rinit=True)
            # forward the PGD adversary
            self.train()
            output, loss = self.loss(xr, labels, adv=False)
            acc_adv = (output.max(1)[1].eq(labels).sum().item()/len(y))
            loss_adv = loss.detach()
            # mandatory report
            print('\t', colored('Orig loss', 'blue'),
                    '%.5f'%loss_orig, 'Acc', '%.3f'%acc_orig,
                    '|', colored('[Adv loss]', 'red'),
                    '%.5f'%loss.item(), 'Acc', '%.3f'%acc_adv)
            return (output, loss)

    def report(self, epoch, iteration, total, output, labels, loss):
        result = classify.cls_report(output, labels, loss)
        print(f'Eph[{epoch}][{iteration}/{total}]', result)

    def validate(self, dataloader):
        device = self.net.conv1.weight.device if not self.dataparallel else self.net.module.conv1.weight.device
        return classify.cls_validate(self, dataloader)

    def attack(self, att, loader, *, dconf, verbose=False):
        device = self.net.conv1.weight.device if not self.dataparallel else self.net.module.conv1.weight.device
        return classify.cls_attack(self, att, loader,
                dconf=dconf, device=device, verbose=verbose)


def getloader(kind:str='train', batchsize:int=1):
    '''
    get corresponding dataloaders (this is for distributed data parallel)
    '''
    config = yaml.load(open('config.yml', 'r').read(), Loader=yaml.SafeLoader)
    if kind == 'train':
        return cifar10.get_loader(os.path.expanduser(config['cifar10']['path']), batchsize, 'train', True)
    else:
        return cifar10.get_loader(os.path.expanduser(config['cifar10']['path']), batchsize, 'test')


if __name__ == '__main__':
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())