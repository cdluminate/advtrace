'''
Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
Source code will be released under Apache-2.0 license if paper accepted.
'''
import os
import torch as th
import collections
from . import fashion
import yaml
from tqdm import tqdm
from .attacks import projectedGradientDescent
from termcolor import cprint, colored
from . import fa_mlp


class Model(fa_mlp.Model):
    def loss(self, x, y, *, adv=True):  # note, adv is true
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
                    eps=0.06, alpha=2./255., maxiter=12, verbose=False,
                    device=device, targeted=False, unbound=False)
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
