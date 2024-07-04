'''
Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
Source code will be released under Apache-2.0 license if paper accepted.
'''
import os
import torch as th
import collections
from . import fashion, base
import yaml
from tqdm import tqdm
from .attacks import projectedGradientDescent
from termcolor import cprint, colored
from . import classify
import torch.nn.functional as F


class Model(base.Model):
    """
    LeNet-like convolutional neural network
    https://github.com/zalandoresearch/fashion-mnist/blob/master/benchmark/convnet.py
    """
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = th.nn.Linear(28 * 28, 512)
        self.fc2 = th.nn.Linear(512, 128)
        self.fc3 = th.nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.fc3(x)
        return x

    def getloader(self, kind:str='train', batchsize:int=1):
        '''
        get corresponding dataloaders
        '''
        config = yaml.load(open('config.yml', 'r').read(),
                Loader=yaml.SafeLoader)
        if kind == 'train':
            return fashion.get_loader(
                    os.path.expanduser(config['fashion-mnist']['path']),
                    batchsize, 'train')
        else:
            return fashion.get_loader(
                    os.path.expanduser(config['fashion-mnist']['path']),
                    batchsize, 't10k')

    def loss(self, x, y, *, adv=False):
        device = self.fc1.weight.device
        images, labels = x.to(device), y.to(device).view(-1)
        if not adv:
            output = self.forward(images)
            loss = th.nn.functional.cross_entropy(output, labels)
            return output, loss
        else: # toggle adversarial training
            # baseline: forward original samples
            self.eval()
            with th.no_grad():
                output, loss = self.loss(images, labels, adv=False)
                acc_orig = (output.max(1)[1].eq(labels).sum().item()/len(y))
                loss_orig = loss.detach()
            # start PGD attack
            xr, r = projectedGradientDescent(self, images, labels,
                    eps=0.3, alpha=2./255., maxiter=48, verbose=False,
                    device='cpu', targeted=False, unbound=False)
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
        device = self.fc1.weight.device
        return classify.cls_validate(self, dataloader)

    def attack(self, att, loader, *, dconf, verbose=False):
        device = self.fc1.weight.device
        return classify.cls_attack(self, att, loader,
                dconf=dconf, device=device, verbose=verbose)
