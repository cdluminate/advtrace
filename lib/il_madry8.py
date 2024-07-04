'''
Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
Source code will be released under Apache-2.0 license if paper accepted.
'''
import os
import torch as th
import collections
from . import ilsvrc, base, classify
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

    def getloader(self, kind:str='train', batchsize:int=1):
        '''
        get corresponding dataloaders
        '''
        config = yaml.load(open('config.yml', 'r').read(), Loader=yaml.SafeLoader)
        path = os.path.expanduser(config['imagenet']['path'])
        if kind == 'train':
            return ilsvrc.get_loader(path, batchsize, 'train')
        else:
            return ilsvrc.get_loader(path, batchsize, 'val')

    def __init__(self):
        super(Model, self).__init__()
        self.net = V.models.resnet50(True)

    def forward(self, x):
        return self.net(x)

    def loss(self, x, y, *, adv=False):
        device = self.net.fc.weight.device
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
        device = self.net.fc.weight.device
        return classify.cls_validate(self, dataloader)

    def attack(self, att, loader, *, dconf, verbose=False):
        device = self.net.fc.weight.device
        return classify.cls_attack(self, att, loader,
                dconf=dconf, device=device, verbose=verbose)
