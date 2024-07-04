'''
Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
Source code will be released under Apache-2.0 license if paper accepted.
'''
from scipy import stats
from sklearn.decomposition import PCA, FastICA, NMF, TruncatedSVD
from sklearn.manifold import TSNE, Isomap, MDS, LocallyLinearEmbedding
from termcolor import cprint, colored
from tqdm import tqdm
import collections
import functools
import math
import numpy as np
import os, sys, re
import pylab as lab
import random
import statistics
import torch as th
import torch.nn.functional as F
import traceback
import json
from .utils import IMmean, IMstd, renorm, denorm, xdnorm, preproc_dict
import foolbox as fb
import torchattacks as atk
import rich
c = rich.get_console()


class ModelwPreprocess(th.nn.Module):
    def __init__(self, model):
        super(ModelwPreprocess, self).__init__()
        self.model = model
    def forward(self, input):
        return self.model(renorm(input))


def __sanity(xr, images, eps, name=''):
    '''
    sanity checker
    '''
    tmp = (denorm(xr)-denorm(images)).view(xr.shape[0], -1)
    c.print(f'[cyan]({name})>[/cyan]' if name else 'r>', end=' ')
    c.print('Eps', '%.3f |'%eps,
            'Min', '%.3f'%tmp.min().item(),
            'Max', '%.3f'%tmp.max().item(),
            'Mean', '%.3f'%tmp.mean().item(),
            'L0', '%.3f'%tmp.norm(0, dim=1).mean().item(),
            'L1', '%.3f'%tmp.norm(1, dim=1).mean().item(),
            'L2', '%.3f'%tmp.norm(2, dim=1).mean().item(),
            'L8', '%.3f'%tmp.norm(float('inf'), dim=1).mean().item(),
            )

def __dlr_loss(x, y):
    # borrowed from https://adversarial-attacks-pytorch.readthedocs.io/en/latest/_modules/torchattacks/attacks/apgd.html#APGD
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    ret = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind
                - x_sorted[:, -1] * (1. - ind)) / (
                    x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)
    return ret.mean()


def UniformNoise(model, images, labels, *, eps=8./255., verbose=False):
    '''
    apply uniform noise to the images.
    the input image has been imagenet-normalized.
    '''
    with th.no_grad():
        r = eps * (2 * th.rand(images.shape).to(images.device) - 1)  # [-eps,+eps]
        xr = renorm((denorm(images) + r).clamp(min=0., max=1.))
    if verbose:
        __sanity(xr, images, eps, name='Uniform')
    return xr, (xr - images).clone().detach()


def GaussianNoise(model, images, labels, *, eps=8./255., verbose=False):
    '''
    Apply gaussian noise.
    '''
    with th.no_grad():
        r = th.clamp(th.randn(images.shape).to(images.device) * eps / 3.0, min=-eps, max=eps)
        xr = renorm((denorm(images) + r).clamp(min=0., max=1.))
    if verbose:
        __sanity(xr, images, eps, name='Gaussian')
    return xr, (xr - images).clone().detach()


def __conveps(eps: float):
    '''
    convert epsilon parameter for l-2 attack since most attacks here are
    L-infty. we treat epsilon differently in l-2 attacks, while the external
    interface for the attack can still pass in 0, 2, 4, 8, 16.
    This makes writing automatic scripts easier.

    for l-2, the true eps are 0.5, 1, 2, 3 following
    https://github.com/MadryLab/robustness
    '''
    approx = lambda x, y: np.abs(x - y) < 1e-7
    if approx(eps, 2./255):
        neweps = 0.5
    elif approx(eps, 4./255):
        neweps = 1.0
    elif approx(eps, 8./255):
        neweps = 2.0
    elif approx(eps, 16./255):
        neweps = 3.0
    else:
        raise ValueError(f"don't know how to translate the epsilon parameter: {eps}")
        neweps = eps*255
    #c.print(f'[yellow]{__name__}> Note, the epsilon has been translated from {eps} (L-infty) to {neweps} (L-2)')
    return neweps


def CW(model, images, labels, *, eps=8./255.,
        maxiter=256, verbose=False, device='cpu'):
    '''
    https://github.com/bethgelab/foolbox/blob/master/foolbox/attacks/carlini_wagner.py
    https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks/carlini_wagner_l2.py

    eps will be converted to 0.25,0.5,1.0,2.0 since the default semantics is for L-inf.
    '''
    model.eval()
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), preprocessing=preproc_dict)
    #print('UT:CW>', 'acc.', fb.utils.accuracy(fmodel, denorm(images), labels))
    #! the images given to the attack should be witin [0, 1].
    #! and foolbox will normalize it again during attack.__call__(...)
    #! the resulting adv example is not normalized
    attack = fb.attacks.L2CarliniWagnerAttack(binary_search_steps=4, steps=maxiter)
    _, xr, _ = attack(fmodel, denorm(images).detach(), labels, epsilons=__conveps(eps))
    xr = renorm(xr)
    if verbose:
        __sanity(xr, images, __conveps(eps), name='CW')
        #print('UT:CW>', 'eps', eps*255.0, 'norm', r.norm().item(), 'acc',
        #        fb.utils.accuracy(fmodel, xr, labels))
    return xr, (xr - images).clone().detach()


def PGDl8(model, images, labels, *, eps=8./255., alpha=2./255,
        maxiter=100, verbose=False, device='cpu', targeted=False,
        random_start=True):
    '''
    PGDl8 --> PGD L-\infty. Use 8 as \infty
    '''
    model.eval()
    fmodel = fb.PyTorchModel(model, bounds=(0,1), preprocessing=preproc_dict)
    attack = fb.attacks.LinfProjectedGradientDescentAttack(
            abs_stepsize=alpha, steps=maxiter, random_start=random_start)
    _, xr, _ = attack(fmodel, denorm(images).detach(), labels, epsilons=eps)
    xr = renorm(xr)
    if verbose:
        __sanity(xr, images, eps, name='PGDl8')
        #print('UT:PGDl8>', 'eps', eps, 'norm', r.abs().max().item(), 'acc',
        #        fb.utils.accuracy(fmodel, xr, labels))
    return xr, (xr - images).clone().detach()


def PGDl8alt(model, images, labels, *, eps=8./255., alpha=2./255.,
        maxiter=100, verbose=False, device='cpu', targeted=False,
        random_start=False):
    '''
    PGD Linf from torchattacks package. (alternative implementation: torchattacks)
    '''
    model.eval()
    fmodel = ModelwPreprocess(model)
    attack = atk.PGD(fmodel, eps=eps, alpha=alpha, steps=maxiter)
    xr = renorm(attack(denorm(images).detach(), labels))
    if verbose:
        __sanity(xr, images, eps, name='PGDl8alt')
        #print('UT:PGDl8alt>', 'eps', eps, 'norm', r.abs().max().item())
    return xr, (xr - images).clone().detach()


def MIM(model, images, labels, *, eps=8./255., alpha=2./255.,
        maxiter=100, verbose=False, device='cpu'):
    '''
    MI-FGSM. BIM with momentum.
    '''
    model.eval()
    fmodel = ModelwPreprocess(model)
    attack = atk.MIFGSM(fmodel, eps=eps, alpha=alpha, steps=maxiter)
    xr = renorm(attack(denorm(images).detach(), labels))
    r = (xr - images).detach()
    if verbose:
        __sanity(xr, images, eps, name='MIM')
    return xr, r


def DIFGSM(model, images, labels, *, eps=8./255., alpha=2./255.,
        maxiter=20, verbose=False, device='cpu'):
    '''
    DI2-FGSM from torchattacks
    default step 20
    '''
    model.eval()
    fmodel = ModelwPreprocess(model)
    attack = atk.DIFGSM(fmodel, eps=eps, alpha=alpha, steps=20)
    xr = renorm(attack(denorm(images).detach(), labels))
    r = (xr - images).detach()
    if verbose:
        __sanity(xr, images, eps, name='DI-FGSM')
    return xr, r


def TIFGSM(model, images, labels, *, eps=8./255., alpha=2./255.,
        maxiter=20, verbose=False, device='cpu'):
    '''
    TI-FGSM from torchattacks
    '''
    model.eval()
    fmodel = ModelwPreprocess(model)
    attack = atk.TIFGSM(fmodel, eps=eps, alpha=alpha, steps=20)
    xr = renorm(attack(denorm(images).detach(), labels))
    r = (xr - images).detach()
    if verbose:
        __sanity(xr, images, eps, name='TI-FGSM')
    return xr, r


def Jitter(model, images, labels, *, eps=8./255., alpha=2./255., verbose=False):
    '''
    Jitter attack from torchattacks
    '''
    model.eval()
    fmodel = ModelwPreprocess(model)
    attack = atk.Jitter(fmodel, eps=eps, alpha=alpha)
    xr = renorm(attack(denorm(images).detach(), labels))
    r = (xr - images).detach()
    if verbose:
        __sanity(xr, images, eps, name='Jitter')
    return xr, r


def APGD(model, images, labels, *, eps=8./255., alpha=2./255.,
        maxiter=100, verbose=False, loss='ce'):
    '''
    APGD from Autoattack. External implementation.
    '''
    model.eval()
    fmodel = ModelwPreprocess(model)
    attack = atk.APGD(fmodel, norm='Linf', eps=eps, steps=maxiter, loss=loss)
    xr = renorm(attack(denorm(images).detach(), labels))
    r = (xr - images).detach()
    if verbose:
        __sanity(xr, images, eps, name='APGD')
    return (xr, r)


def APGDdlr(model, images, labels, *, eps=8./255., alpha=2./255.,
        maxiter=100, verbose=False):
    '''
    APGD using DLR loss.
    '''
    return APGD(model, images, labels, eps=eps, alpha=alpha, maxiter=maxiter,
            verbose=verbose, loss='dlr')


def FAB(model, images, labels, *, eps=8./255., alpha=2./255.,
        maxiter=100, verbose=False, device='cpu', targeted=False,
        random_start=False):
    '''
    external implementation.
    '''
    model.eval()
    fmodel = ModelwPreprocess(model)
    if images.shape[-1] == 224:
        Nclass = 1000
    elif images.shape[-1] == 32:
        Nclass = 10
    attack = atk.FAB(fmodel, norm='Linf', eps=eps, steps=100, n_classes=Nclass)
    xr = renorm(attack(denorm(images).detach(), labels))
    r = (xr - images).detach()
    if verbose:
        __sanity(xr, images, eps, name='FAB')
    return xr, r


def Square(model, images, labels, *, eps=8./255., alpha=2./255.,
        maxiter=100, verbose=False, device='cpu'):
    '''
    square attack (part of autoattac) from torchattacks
    '''
    model.eval()
    fmodel = ModelwPreprocess(model)
    attack = atk.Square(model, eps=eps)
    xr = renorm(attack(denorm(images).detach(), labels))
    r = (xr - images).detach()
    if verbose:
        __sanity(xr, images, eps, name='Square')
    return xr, r


def AutoAttack(model, images, labels, *, eps=8./255., alpha=2./255.,
        maxiter=100, verbose=False, device='cpu', targeted=False,
        random_start=False):
    '''
    external implementation.
    '''
    model.eval()
    fmodel = ModelwPreprocess(model)
    if images.shape[-1] == 224:
        Nclass = 1000
    elif images.shape[-1] == 32:
        Nclass = 10
    attack = atk.AutoAttack(fmodel, norm='Linf', eps=eps, n_classes=Nclass)
    xr = renorm(attack(denorm(images).detach(), labels))
    r = (xr - images).detach()
    if verbose:
        __sanity(xr, images, eps, name='AutoAttack')
    return xr, r


def PGDl2(model, images, labels, *, eps=8./255., alpha=2./255,
        maxiter=100, verbose=False, device='cpu', targeted=False,
        random_start=False):
    # PGD l2
    model.eval()
    fmodel = fb.PyTorchModel(model, bounds=(0,1), preprocessing=preproc_dict)
    attack = fb.attacks.L2ProjectedGradientDescentAttack(
            abs_stepsize=alpha, steps=maxiter, random_start=random_start)
    _, xr, _ = attack(fmodel, denorm(images).detach(), labels, epsilons=__conveps(eps))
    xr = renorm(xr)
    r = (xr - images).detach()
    if verbose:
        __sanity(xr, images, __conveps(eps), name='PGDl2')
    #print('UT:PGDl2>', 'eps', eps*255., 'norm', r.norm().item(), 'acc',
    #        fb.utils.accuracy(fmodel, xr, labels))
    return (xr, r)


def FMNl8(model, images, labels, *, eps=8./255., alpha=2./255,
        maxiter=100, verbose=False, device='cpu', targeted=False,
        random_start=False):
    '''
    https://foolbox.readthedocs.io/en/stable/modules/attacks.html#foolbox.attacks.LInfFMNAttack
    '''
    model.eval()
    fmodel = fb.PyTorchModel(model, bounds=(0,1), preprocessing=preproc_dict)
    attack = fb.attacks.LInfFMNAttack(steps=100, binary_search_steps=10)
    _, xr, _ = attack(fmodel, denorm(images).detach(), labels, epsilons=eps)
    r = (denorm(images) - xr).detach()
    print('UT:FMNl8>', 'eps', eps, 'norm', r.abs().max().item(), 'acc',
            fb.utils.accuracy(fmodel, xr, labels))
    xr = renorm(xr)
    return (xr, r)


def projectedGradientDescent(model, images, labels, *, eps=0.0, alpha=2./255.,
        maxiter=1, verbose=False, device='cpu', targeted=False, unbound=False,
        rinit=False, B_UAP=False, losstype='ce', norm='linf'):
    '''
    https://github.com/bethgelab/foolbox/blob/master/foolbox/attacks/gradient.py
    https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks/projected_gradient_descent.py
    This function implements BIM when rinit==False. It becomes PGD when rinit==True.
    B-UAP is the batch-wise universal (image-agnostic) adversarial perturbation
    '''
    assert(type(images) == th.Tensor)
    assert(eps is not None)
    assert(losstype in ('ce', 'fa', 'dlr'))
    # prepare
    images = images.to(device).clone().detach()
    images_orig = images.clone().detach()
    images.requires_grad = True
    labels = labels.to(device).view(-1)
    model.eval()
    with th.no_grad():
        output_orig = model.forward(images).detach()
    if norm == 'l2':
        eps = __conveps(eps)
        alpha = 0.2
    if norm == 'l2':
        raise Exception('this l2 implemention is buggy')
    # random start?
    if bool(os.getenv('RINIT', False)):
        rinit = True
    if rinit:
        if images_orig.min() >= 0. and images_orig.max() <= 1.:
            if not B_UAP:
                images = images + eps*2*(0.5-th.rand(images.shape, device=device))
            else:
                images = images + eps*2*(0.5-th.rand([1,*images.shape[1:]], device=device))
            images = th.clamp(images, min=0., max=1.)
            images = images.clone().detach()
            images.requires_grad = True
        elif images_orig.min() < 0.:
            if not B_UAP:
                images = images + (eps/IMstd[:,None,None]).to(device)*2*(0.5-th.rand(images.shape, device=device))
            else:
                images = images + (eps/IMstd[:,None,None]).to(device)*2*(0.5-th.rand([1,*images.shape[1:]], device=device))
            images = th.max(images, renorm(th.zeros(images.shape, device=device)))
            images = th.min(images, renorm(th.ones(images.shape, device=device)))
            images = images.clone().detach()
            images.requires_grad = True
        else:
            raise Exception
    # start attack
    model.eval() # NOTE: model.train() may incur problems
    for iteration in range(maxiter):
        # setup optimizers, and clear the gradient
        optim = th.optim.SGD(model.parameters(), lr=0.)
        optim.zero_grad()
        optimx = th.optim.SGD([images], lr=1.)
        optimx.zero_grad()
        # forward
        output = model.forward(images)
        if losstype == 'ce':
            loss = F.cross_entropy(output, labels)
        elif losstype == 'fa':
            loss = F.pairwise_distance(output, output_orig).sum()
        elif losstype == 'dlr':
            loss = __dlr_loss(output, labels)
        else:
            raise ValueError(losstype)
        if not targeted:
            loss = -loss  # gradient ascent
        else:
            pass  # gradient descent
        if norm == 'l2':
            # this requires relatively new version of pytorch.
            # the other code was written when pytorch was not so convenient
            g = th.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]
        else:
            loss.backward()

        # Batch UAP?
        if B_UAP:
            # batch-wise UAP. let's aggregate the adversarial perturbation
            with th.no_grad():
                aggrad = images.grad.mean(0, keepdim=True).detach().repeat([images.shape[0], 1, 1, 1])
                images.grad.data.copy_(aggrad)
        # iterative? or single-step?
        if norm == 'linf' and maxiter > 1:
            if images_orig.min() >= 0. and images_orig.max() <= 1.:
                images.grad.data.copy_(alpha * th.sign(images.grad))
            elif images_orig.min() < 0.:
                images.grad.data.copy_((alpha/IMstd[:,None,None]).to(device) * th.sign(images.grad))
            else:
                #raise Exception
                print('! weird value range')
                # according to observation, this may happen on imagenet
                images.grad.data.copy_((alpha/IMstd[:,None,None]).to(device) * th.sign(images.grad))
        elif norm == 'linf':
            if images.min() >= 0. and images.max() <= 1.:
                images.grad.data.copy_(eps * th.sign(images.grad))
            elif images.min() < 0.:
                images.grad.data.copy_((eps/IMstd[:,None,None]).to(device) * th.sign(images.grad))
            else:
                #raise Exception
                print('! weird value range')
                images.grad.data.copy_((eps/IMstd[:,None,None]).to(device) * th.sign(images.grad))
        elif norm == 'l2' and maxiter > 1:
            # https://adversarial-attacks-pytorch.readthedocs.io/en/latest/_modules/torchattacks/attacks/pgdl2.html#PGDL2
            pass
        elif norm == 'l2':
            raise Exception
        # update the input
        if norm == 'linf':
            optimx.step()
        # project the input (L_\infty bound)
        if not unbound:
            if images_orig.min() >= 0. and images_orig.max() <= 1.:
                images = th.min(images, images_orig + eps)
                images = th.max(images, images_orig - eps)
            elif images_orig.min() < 0.:
                images = th.min(images, images_orig + (eps/IMstd[:,None,None]).to(device))
                images = th.max(images, images_orig - (eps/IMstd[:,None,None]).to(device))
            else:
                #raise Exception
                print('! weird value range')
                images = th.min(images, images_orig + (eps/IMstd[:,None,None]).to(device))
                images = th.max(images, images_orig - (eps/IMstd[:,None,None]).to(device))
        if norm == 'linf' and images_orig.min() >= 0. and images_orig.max() <= 1.:
            images = th.clamp(images, min=0., max=1.)
        elif norm == 'linf' and images_orig.min() < 0.:
            images = th.max(images, renorm(th.zeros(images.shape, device=device)))
            images = th.min(images, renorm(th.ones(images.shape, device=device)))
        elif norm == 'linf':
            images = th.max(images, renorm(th.zeros(images.shape, device=device)))
            images = th.min(images, renorm(th.ones(images.shape, device=device)))
        elif norm == 'l2' and images_orig.min() < 0.:
            # https://adversarial-attacks-pytorch.readthedocs.io/en/latest/_modules/torchattacks/attacks/pgdl2.html#PGDL2
            # https://github.com/MadryLab/robustness/blob/master/robustness/attack_steps.py

            # madry:step
            # let y = (images - mu) / sigma  : normalize step
            # then images.grad = partial loss / partial y
            # and partial loss / partial images
            #     = partial loss / partial y * partial y / partial x
            #     = images.grad / sigma
            #g = images.grad / IMstd[:,None,None].to(device)
            g = g / IMstd[:, None, None].to(device)
            _l = len(images.shape) - 1
            images = denorm(images)  # restor to [0, 1]
            g_norm = th.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1]*_l))
            scaled_g = g / (g_norm + 1e-10)
            images = th.clamp(images + scaled_g * alpha, 0, 1)
            # madry:project
            #diff = denorm(images) - denorm(images_orig)
            diff = images - denorm(images_orig)
            diff = diff.renorm(p=2, dim=0, maxnorm=eps)
            images = th.clamp(images_orig + diff, min=0., max=1.)
            #
            images = renorm(images).clone().detach() # restore back to imagenet normalize
        else:
            raise Exception
        # detach from computation graph and prepare for the next round
        images = images.clone().detach()
        images.requires_grad = True
        if maxiter > 1 and verbose:
            cprint('  (PGD)' if not B_UAP else '  (B-UAP)', 'blue', end=' ')
            print(f'iter {iteration:3d}', f'\tloss= {loss.item():7.4f}',
                    f'\tL2m= {(images-images_orig).norm(2,dim=1).mean():7.4f}',
                    f'\tL0m= {(images-images_orig).abs().max(dim=1)[0].mean():7.4f}')
    if False: # visualization
        for i in range(images.shape[0]):
            npxo = images_orig[i].detach().cpu().squeeze().view(28,28).numpy()
            npx  = images[i].detach().cpu().squeeze().view(28,28).numpy()
            lab.subplot(121); lab.imshow(npxo, cmap='gray'); lab.colorbar()
            lab.subplot(122); lab.imshow(npx,  cmap='gray'); lab.colorbar()
            lab.show()
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
            #raise Exception
            tmp = r.mul(IMstd[:,None,None].to(r.device)).view(r.shape[0], -1)

        cprint('r>', 'white', 'on_cyan', end=' ')
        print('Min', '%.3f'%tmp.min().item(),
                'Max', '%.3f'%tmp.max().item(),
                'Mean', '%.3f'%tmp.mean().item(),
                'L0', '%.3f'%tmp.norm(0, dim=1).mean().item(),
                'L1', '%.3f'%tmp.norm(1, dim=1).mean().item(),
                'L2', '%.3f'%tmp.norm(2, dim=1).mean().item())
    return (xr, r)


def NES(model, images, labels, *, eps=8./255., maxprobe=1e4,
        spsa=False, verbose=False):
    '''
    natural evolution strategy
    implementation inspired by
    (1) https://arxiv.org/pdf/1804.08598.pdf
    (2) https://github.com/cdluminate/advorder/blob/main/lib/reorder.py
    (3) https://github.com/thu-ml/realsafe

    the input range of model is imagenet normalized. we will wrap this
    model into unnormalized again. the algorithm requires [0,1] range.
    '''
    assert(len(images.shape) == 4)
    # initialize parameters
    B, C, H, W = images.shape
    dimreduce = True if H*W>2000 else False  # this is used to differentiate cifar10 from imagenet
    Npop = 100
    lr = 2./255.
    sigma = eps / 0.5
    # dispatch based on shape
    if B == 1:
        # Do NES against one input sample
        net = ModelwPreprocess(model)
        net.model.eval()
        labels_ = labels.expand(Npop)

        pgd = denorm(images).clone().detach()
        images0 = denorm(images).clone().detach()
        # first evaluation
        with th.no_grad():
            output = net(pgd)
            score0 = F.cross_entropy(output, labels)

            for iteration in range(int(maxprobe/Npop)):
                # generate population
                if dimreduce:
                    _tmp = sigma * th.randn((Npop//2, 3, 32, 32), device=images.device)
                    if spsa:
                        _tmp = sigma * th.sign(_tmp)
                    perts = F.interpolate(_tmp, scale_factor=[7,7])
                else:
                    perts = sigma * th.randn((Npop//2, *pgd.shape[1:]), device=pgd.device)
                    if spsa:
                        perts = sigma * th.sign(perts)
                perts = th.cat([perts, -perts], dim=0).clamp(min=-eps, max=+eps)
                qx = (pgd + perts).clamp(min=0., max=1.)
                qx = th.min(images0.expand(Npop, *images0.shape[1:]) + eps, qx)
                qx = th.max(images0.expand(Npop, *images0.shape[1:]) - eps, qx)
                # evaluate population
                output = net(qx)
                score = F.cross_entropy(output, labels_, reduction='none') # vector instead of scalar
                # estimate gradient
                # \nabla_\mu \log N(x;\mu,\sigma) = (x-\mu)/(\sigma^2)
                grad = (score.view(-1, 1, 1, 1) * perts).mean(dim=0) / sigma
                # apply to sample
                pgd += (lr * th.sign(grad)) #.clamp(min=-eps, max=+eps)
                pgd = th.min(images0 + eps, pgd)
                pgd = th.max(images0 - eps, pgd)
                pgd = pgd.clamp(min=0., max=1.)
                # evaluate again
                output = net(pgd)
                score = F.cross_entropy(output, labels)
            # report
            if verbose:
                print('SPSA>' if spsa else 'NES>',
                        f'({iteration})', 'score_0', score0.item(),
                        'score', (score + score0).mean().item())
        pgd = renorm(pgd)
        return pgd, pgd - renorm(images0)
    else: # B>1
        # reduce and conquer
        xr, r = list(zip(*[
            NES(model, images[i].view(1,C,H,W), labels[i].view(-1),
                eps=eps, maxprobe=maxprobe, spsa=spsa, verbose=verbose) for i in range(B)]))
        xr = th.vstack(xr)
        r = th.vstack(r)
        if verbose:
            tmp = (denorm(xr)-denorm(images)).view(r.shape[0], -1)
            cprint('r>', 'white', 'on_cyan', end=' ')
            print('Min', '%.3f'%tmp.min().item(),
                    'Max', '%.3f'%tmp.max().item(),
                    'Mean', '%.3f'%tmp.mean().item(),
                    'L0', '%.3f'%tmp.norm(0, dim=1).mean().item(),
                    'L1', '%.3f'%tmp.norm(1, dim=1).mean().item(),
                    'L2', '%.3f'%tmp.norm(2, dim=1).mean().item(),
                    'L8', '%.3f'%tmp.norm(float('inf'), dim=1).mean().item(),
                    )
        return xr, r


def SPSA(model, images, labels, *, eps=8./255., maxprobe=1e4, verbose=False):
    '''
    SPSA attack. different sampling.
    '''
    return NES(model, images, labels, eps=eps, maxprobe=maxprobe,
            spsa=True, verbose=verbose)


def __BIMl2(model, images, labels, *, eps=2.0, alpha=0.2, maxiter=100):
    '''
    BIM-l2 implemntation
    https://adversarial-attacks-pytorch.readthedocs.io/en/latest/_modules/torchattacks/attacks/pgdl2.html#PGDL2
    '''
    device = images.device
    images_orig = images.clone().detach()
    labels_orig = labels.clone().detach()
    images = images.clone().detach()
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
    return images


def BIMl2(model, images, labels, *,
        eps=8./255., alpha=2./255, maxiter=100, verbose=False):
    # PGD l2
    model.eval()
    fmodel = ModelwPreprocess(model)
    xr = __BIMl2(fmodel, denorm(images).detach(), labels,
            eps=__conveps(eps), alpha=0.2, maxiter=maxiter)
    xr = renorm(xr)
    r = (xr - images).detach()
    if verbose:
        __sanity(xr, images, __conveps(eps), name='BIMl2')
    #print('UT:PGDl2>', 'eps', eps*255., 'norm', r.norm().item(), 'acc',
    #        fb.utils.accuracy(fmodel, xr, labels))
    return (xr, r)


def __LogitMatchingl8(model, images, labels, *, logit=None, eps=2.0, alpha=0.2, maxiter=100):
    '''
    Logit matching in adaptive attack (madry) Section 5.2
    '''
    assert logit is not None
    device = images.device
    images_orig = images.clone().detach()
    labels_orig = labels.clone().detach()
    images = images.clone().detach()
    for _ in range(maxiter):
        images.requires_grad = True
        outputs = model.forward(images)
        loss = F.mse_loss(outputs, logit, reduction='sum') # XXX: instead of ce loss
        # update images
        grad = th.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]
        images = images.detach() + alpha * grad.sign()
        diff = th.clamp(images - images_orig, min=-eps, max=+eps)
        images = th.clamp(images_orig + diff, min=0, max=1).detach()
    return images


def LogitMatchingl8(model, images, labels, *,
        logit=None,
        eps=8./255., alpha=2./255, maxiter=100, verbose=False):
    assert logit is not None
    model.eval()
    fmodel = ModelwPreprocess(model)
    xr = __LogitMatchingl8(fmodel, denorm(images).detach(), labels,
            logit=logit,
            eps=eps, alpha=alpha, maxiter=maxiter)
    xr = renorm(xr)
    r = (xr - images).detach()
    if verbose:
        __sanity(xr, images, eps, name='LogitMatchingl8')
    return (xr, r)


def Interpolate(model, images, labels, *, eps=8./255, alpha=2./255, maxiter=100, device='cuda', verbose=False):
    '''
    Section 5.13 of adaptive attack
    '''
    model.eval()
    (xr, r) = projectedGradientDescent(model, images, labels,
            eps=eps, alpha=alpha, maxiter=maxiter, device=device, verbose=verbose)
    lower = th.zeros(labels.nelement()).to(device)
    upper = th.ones(labels.nelement()).to(device)
    for _ in range(7):
        mid = (lower + upper) / 2.0
        #print('debug:', mid)
        interp = images.clone().detach() + mid.view(-1, 1, 1, 1) * r
        output = model.forward(interp)
        misclass: th.Tensor[bool] = (output.max(1)[1] == labels.view(-1))
        lower = th.where(misclass, lower, mid)
        upper = th.where(misclass, mid, upper)
    xr = images.clone().detach() + upper.view(-1, 1, 1, 1) * r
    r = (xr - images).detach()
    return (xr, r)
