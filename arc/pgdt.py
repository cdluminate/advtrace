'''
Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
Source code will be released under Apache-2.0 license if paper accepted.
'''
from termcolor import cprint, colored
import functools as ft
import math
import numpy as np
import os
import sys
import re
import random
import statistics
import torch as th
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import json
import torchattacks as atk
from .utils import IMmean, IMstd, renorm, denorm, xdnorm


def changelabel(labels: th.Tensor, maxclass: int) -> th.Tensor:
    '''
    change to another class for random targeted attack
    '''
    new = []
    for i in labels:
        candidates = tuple(set(range(maxclass)) - {i})
        new.append(random.choice(candidates))
    return th.tensor(new).to(labels.device)


def __dlr_loss(x, y):
    # borrowed from https://adversarial-attacks-pytorch.readthedocs.io/en/latest/_modules/torchattacks/attacks/apgd.html#APGD
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    ret = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind
                - x_sorted[:, -1] * (1. - ind)) / (
                    x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)
    return ret.sum()


def PGDtraj(model, images, labels, *, eps=8./255., alpha=2./255.,
        maxiter=100, verbose=False, device='cpu', targeted=False,
        losstype='ce'):
    '''
    the PGD that gives the whole trajectory instead of merely the endpoint.
    not random init. so this equals BIM

    reference:
    https://github.com/MadryLab/robustness/blob/master/robustness/attack_steps.py
    '''
    assert(type(images) == th.Tensor)
    assert(eps is not None)
    assert(losstype in ('ce', 'fa', 'dlr'))
    # prepare
    images = images.to(device).clone().detach()
    images_orig = images.clone().detach()
    images.requires_grad = True
    labels = labels.to(device).view(-1)
    traj = [images_orig.clone().detach()]
    model.eval()
    with th.no_grad():
        output_orig = model.forward(images).detach()
    # start attack

    def __loss(model, images, labels, losstype, targeted):
        output = model.forward(images)
        if losstype == 'ce':
            loss = F.cross_entropy(output, labels)
        elif losstype == 'fa':
            # feature adversary, namely euclidean distance
            loss = F.pairwise_distance(output, output_orig).sum()
        elif losstype == 'dlr':
            loss = __dlr_loss(output, labels)
        else:
            raise ValueError(f'unknwn loss type {losstype}')
        if not targeted:
            loss = -loss  # gradient ascent
        else:
            pass  # gradient descent
        return output, loss

    model.eval() # NOTE: model.train() may incur problems
    for iteration in range(maxiter):
        # setup optimizers, and clear the gradient
        optim = th.optim.SGD(model.parameters(), lr=0.)
        optim.zero_grad()
        optimx = th.optim.SGD([images], lr=1.)
        optimx.zero_grad()
        # forward
        output, loss = __loss(model, images, labels, losstype, targeted)
        # calculate gradient
        loss.backward()

        # calculate the update
        if images_orig.min() >= 0. and images_orig.max() <= 1.:
            # iterative? or single-step?
            if maxiter > 1:
                images.grad.data.copy_(alpha * th.sign(images.grad))
            else:
                images.grad.data.copy_(eps * th.sign(images.grad))
        elif images_orig.min() < 0. or images_orig.max() > 1.:
            if maxiter > 1:
                images.grad.data.copy_((alpha/IMstd[:,None,None]).to(device) * th.sign(images.grad))
            else:
                images.grad.data.copy_((eps/IMstd[:,None,None]).to(device) * th.sign(images.grad))
        else:
            # this only happens for ImageNet according to observation.
            print('! value range is weird. we assume imagenet handling')
            if maxiter > 1:
                images.grad.data.copy_((alpha/IMstd[:,None,None]).to(device) * th.sign(images.grad))
            else:
                images.grad.data.copy_((eps/IMstd[:,None,None]).to(device) * th.sign(images.grad))
        # perform update
        optimx.step()
        # project the input (L_\infty bound)
        if images_orig.min() >= 0. and images_orig.max() <= 1.:
            images = th.min(images, images_orig + eps)
            images = th.max(images, images_orig - eps)
            images = th.clamp(images, min=0., max=1.)
        elif images_orig.min() < 0.:
            images = th.min(images, images_orig + (eps/IMstd[:,None,None]).to(device))
            images = th.max(images, images_orig - (eps/IMstd[:,None,None]).to(device))
            images = th.max(images, renorm(th.zeros(images.shape, device=device)))
            images = th.min(images, renorm(th.ones(images.shape, device=device)))
        else:
            # this only happens for ImageNet according to observation.
            print('! value range is weird. we assume imagenet handling')
            images = th.min(images, images_orig + (eps/IMstd[:,None,None]).to(device))
            images = th.max(images, images_orig - (eps/IMstd[:,None,None]).to(device))
            images = th.max(images, renorm(th.zeros(images.shape, device=device)))
            images = th.min(images, renorm(th.ones(images.shape, device=device)))
        # detach from computation graph and prepare for the next round
        images = images.clone().detach()
        traj.append(images.clone().detach())
        images.requires_grad = True
        if maxiter > 1 and verbose:
            cprint('  (PGDT)', 'blue', end=' ')
            print(f'iter {iteration:3d}', f'\tloss= {loss.item():7.4f}',
                    f'\tL2m= {(images-images_orig).norm(2,dim=1).mean():7.4f}',
                    f'\tL0m= {(images-images_orig).abs().max(dim=1)[0].mean():7.4f}')

    xr = images.detach()
    xr.requires_grad = False
    r = (images - images_orig).detach()
    r.requires_grad = False
    if verbose:
        if images_orig.min() >= 0. and images_orig.max() <= 1.:
            tmp = r.view(r.shape[0], -1)
        elif images_orig.min() < 0.:
            tmp = r.mul(IMstd[:,None,None].to(r.device)).view(r.shape[0], -1)
        else:
            raise Exception

        cprint('r>', 'white', 'on_cyan', end=' ')
        print('Min', '%.3f'%tmp.min().item(),
                'Max', '%.3f'%tmp.max().item(),
                'Mean', '%.3f'%tmp.mean().item(),
                'L0', '%.3f'%tmp.norm(0, dim=1).mean().item(),
                'L1', '%.3f'%tmp.norm(1, dim=1).mean().item(),
                'L2', '%.3f'%tmp.norm(2, dim=1).mean().item())
    traj = th.stack(traj)
    print('PGDT>', 'trajectory shape', traj.shape)
    return (xr, r, traj)


def __BIMl2(model, images, labels, *, eps=2.0, alpha=0.2, maxiter=100):
    '''
    BIM-l2 implemntation
    https://adversarial-attacks-pytorch.readthedocs.io/en/latest/_modules/torchattacks/attacks/pgdl2.html#PGDL2
    '''
    device = images.device
    images_orig = images.clone().detach()
    labels_orig = labels.clone().detach()
    images = images.clone().detach()
    traj = [images_orig.clone().detach()]
    B = images.shape[0]
    for _ in range(maxiter):
        images.requires_grad = True
        outputs = model.forward(images)
        loss = F.cross_entropy(outputs, labels)
        # update images
        grad = th.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]
        g_norm = th.norm(grad.view(B, -1), p=2, dim=1) + 1e-10
        grad = grad / g_norm.view(B, 1, 1, 1)
        images = images.detach() + alpha * grad

        diff = images - images_orig
        d_norm = th.norm(diff.view(B, -1), p=2, dim=1)
        factor = eps / d_norm
        factor = th.min(factor, th.ones_like(d_norm))
        diff = diff * factor.view(-1, 1, 1, 1)

        images = th.clamp(images_orig + diff, min=0, max=1).detach()
        traj.append(images.clone().detach())
    traj = th.stack(traj)
    return images, traj

class ModelwPreprocess(th.nn.Module):
    def __init__(self, model):
        super(ModelwPreprocess, self).__init__()
        self.model = model
    def forward(self, input):
        return self.model(renorm(input))


def BIMl2T(model, images, labels, *, maxiter=100, verbose=False):
    # PGD l2
    model.eval()
    fmodel = ModelwPreprocess(model)
    xr, traj = __BIMl2(fmodel, denorm(images).detach(), labels,
            eps=2.0, alpha=0.2, maxiter=maxiter)
    xr = renorm(xr)
    r = (xr - images).detach()
    return (xr, r, traj)


def __BIMl8(model, images, labels, *, eps=8./255., alpha=2./255, maxiter=100):
    '''
    BIM-l8 implemntation
    https://adversarial-attacks-pytorch.readthedocs.io/en/latest/_modules/torchattacks/attacks/pgd.html
    The PGDtraj function contains too much historical burden left from old versions of pytorch
    '''
    device = images.device
    images_orig = images.clone().detach()
    labels_orig = labels.clone().detach()
    images = images.clone().detach()
    traj = [renorm(images_orig.clone()).detach()]
    B = images.shape[0]
    for _ in range(maxiter):
        images.requires_grad = True
        outputs = model.forward(images)
        loss = F.cross_entropy(outputs, labels)
        # update images
        grad = th.autograd.grad(loss, images,
                retain_graph=False, create_graph=False)[0]
        images = images.detach() + alpha * grad.sign()
        diff = th.clamp(images - images_orig, min=-eps, max=+eps)
        images = th.clamp(images_orig + diff, min=0, max=1).detach()

        traj.append(renorm(images.clone()).detach())
    traj = th.stack(traj)
    return images, traj


def BIMl8T(model, images, labels, *, maxiter=100, verbose=False):
    model.eval()
    fmodel = ModelwPreprocess(model)
    xr, traj = __BIMl8(fmodel, denorm(images).detach(), labels,
            eps=8./255., alpha=2./255., maxiter=maxiter)
    xr = renorm(xr)
    r = (xr - images).detach()
    return (xr, r, traj)


def __BIMl8n(model, images, labels, *, eps=8./255., alpha=2./255, maxiter=100):
    '''
    BIM-l8 implemntation
    https://adversarial-attacks-pytorch.readthedocs.io/en/latest/_modules/torchattacks/attacks/pgd.html
    The PGDtraj function contains too much historical burden left from old versions of pytorch
    n-postfix (this assumes input is imagenet-normalized
    '''
    device = images.device
    images_orig = images.clone().detach()
    labels_orig = labels.clone().detach()
    images = images.clone().detach()
    traj = [images_orig.clone().detach()]
    B = images.shape[0]
    optm = th.optim.SGD(model.parameters(), lr=0.)
    for _ in range(maxiter):
        optm.zero_grad()
        images.requires_grad = True
        outputs = model.forward(images)
        loss = F.cross_entropy(outputs, labels)
        # update images
        grad = th.autograd.grad(loss, images,
                retain_graph=False, create_graph=False)[0]
        #images = images.detach() + alpha * grad.sign()
        images = images.detach() + alpha/IMstd.view(1, 3, 1, 1).to(device) * grad.sign()
        #diff = th.clamp(images - images_orig, min=-eps, max=+eps)
        diff = images - images_orig
        diff = th.max(diff, -eps/IMstd.view(1,3,1,1).to(device))
        diff = th.min(diff, +eps/IMstd.view(1,3,1,1).to(device))
        #images = th.clamp(images + diff, min=0, max=1).detach()
        images = images_orig + diff
        images = th.max(images, renorm(th.zeros(images.shape, device=device)))
        images = th.min(images, renorm(th.ones(images.shape, device=device)))
        images = images.detach()

        traj.append(images.clone().detach())
    optm.zero_grad()
    traj = th.stack(traj)
    return images, traj

def BIMl8Tn(model, images, labels, *, maxiter=100, verbose=False):
    '''
    different from BIMl8. this directly takes imagenet-normalized image input.
    '''
    model.eval()
    #fmodel = ModelwPreprocess(model)  # not needed here.
    xr, traj = __BIMl8n(model, images.detach(), labels,
            eps=8./255., alpha=2./255., maxiter=maxiter)
    #xr = renorm(xr)
    r = (xr - images).detach()
    return (xr, r, traj)


class CWTraj__(atk.CW):
    '''
    we override a part of it to get the trajectory
    https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/cw.py
    '''
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        print('Using __CWTraj')
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10*torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        optimizer = optim.Adam([w], lr=self.lr)

        traj = []  # XXX
        for step in range(self.steps):
            print('__CWTraj step', step)
            # Get adversarial images
            adv_images = self.tanh_space(w)
            traj.append(adv_images.clone().detach())  # XXX

            # Calculate loss
            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs = self.model(adv_images)
            if self._targeted:
                f_loss = self.f(outputs, target_labels).sum()
            else:
                f_loss = self.f(outputs, labels).sum()

            cost = L2_loss + self.c*f_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update adversarial images
            _, pre = torch.max(outputs.detach(), 1)
            correct = (pre == labels).float()

            # filter out images that get either correct predictions or non-decreasing loss, 
            # i.e., only images that are both misclassified and loss-decreasing are left 
            mask = (1-correct)*(best_L2 > current_L2.detach())
            best_L2 = mask*current_L2.detach() + (1-mask)*best_L2

            mask = mask.view([-1]+[1]*(dim-1))
            best_adv_images = mask*adv_images.detach() + (1-mask)*best_adv_images

            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            #if step % max(self.steps//10,1) == 0:
            #    if cost.item() > prev_cost:
            #        return best_adv_images
            #    prev_cost = cost.item()
        traj.append(adv_images.clone().detach())
        traj = th.stack(traj)
        self.trajectory = traj
        print('Trajectory', traj.shape)

        return best_adv_images


def CWTraj(model, images, labels, *, verbose=False):
    '''
    wrapper to __CWTraj
    '''
    model.eval()
    fmodel = ModelwPreprocess(model)
    # lr: 0.01 too small, 0.1 too small, 1.0 too large, 0.5 too large, 0.3 ok
    # c: 1e-3. looks right. not sensitive
    attack = CWTraj__(fmodel, c=1e-3, kappa=0, steps=6, lr=0.3)
    xr = renorm(attack(denorm(images).detach(), labels))
    return xr, (xr - images).clone().detach(), \
            th.stack([renorm(x) for x in attack.trajectory])
