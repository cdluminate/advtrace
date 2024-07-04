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
sys.path.extend(['.', '..'])
import pylab as lab
import random
import statistics
import torch as th
import torchvision as V
import traceback
import json
from .utils import IMmean, IMstd, renorm, denorm, xdnorm, preproc_dict
from .attacks import *
from .transfer import *
import fcntl
import gc
import foolbox as fb
from arc import *
from . import nss
import warnings
import time
#
import rich
from rich.progress import track
c = rich.get_console()

__TIME_PER_IMAGE__ = []

#######################################################################
def changelabel(labels: th.Tensor, maxclass: int) -> th.Tensor:
    '''
    change to another class for random targeted attack
    '''
    new = []
    for i in labels:
        candidates = tuple(set(range(maxclass)) - {i})
        new.append(random.choice(candidates))
    return th.tensor(new).to(labels.device)


def cls_report(output, labels, loss):
    pred = output.max(1)[1].cpu()
    acc = 100.*pred.eq(labels.cpu().flatten()).sum().item() / len(labels)
    return collections.namedtuple(
            'Res', ['loss', 'accu'])('%.2f'%loss.item(), '%.2f'%acc)

def cls_validate(model, dataloader):
    model.eval()
    test_loss, correct, total = 0, 0, 0
    with th.no_grad():
        for iteration, (images, labels) in enumerate(dataloader):
            labels = labels.view(-1)
            output, loss = model.loss(images, labels, adv=False)
            test_loss += loss.item()
            pred = output.max(1)[1].cpu().view(-1)
            correct += pred.eq(labels.cpu().flatten()).sum().item()
            total += len(labels)
    return collections.namedtuple('Result', ['loss', 'accuracy'])(
            test_loss, 100 * correct / total)

def cls_attack(model, attack, loader, *, dconf, device, verbose=False):
    '''
    generic attack method for classification models
    '''
    orig_correct, adv_correct, total = 0, 0, 0
    for N, (images, labels) in enumerate(loader):
        SKIP = os.getenv('SKIP', None)
        if SKIP and N < int(SKIP):
            print(f'Skipping batch <{N}>')
            continue
        current_tag = os.getenv('TAG', None)
        if current_tag is not None and current_tag == 'LM':
            passin_loader = loader
        else:
            passin_loader = None
        if attack == 'UT:FGSM':
            xr, r, out, loss, count = ClassAttackBatch(model, images, labels,
                eps=dconf['epsilon'], alpha=2./255., verbose=verbose,
                device=device, maxiter=1)
        elif attack == 'UT:PGD':
            xr, r, out, loss, count = ClassAttackBatch(model, images, labels,
                eps=dconf['epsilon'], alpha=2./255., verbose=verbose,
                device=device, maxiter=dconf['maxiter'])
        elif attack in ('UT:CW', 'UT:PGDl8', 'UT:PGDl2', 'UT:NES', 'UT:SPSA', 'UT:GA', 'UT:MIM'):
            xr, r, out, loss, count = ClassAttackBatch(model, images, labels,
                eps=dconf['epsilon'], alpha=2./255., verbose=verbose,
                device=device, maxiter=dconf['maxiter'], atype=attack)
        elif attack == 'UT:PGDT' or attack == 'ARC':
            xr, r, out, loss, count = AttackLoliDetectBatch(model, images, labels,
                eps=dconf['epsilon'], alpha=2./255., verbose=verbose,
                device=device, maxiter=dconf['maxiter'], loader=passin_loader)
        elif attack == 'NSS':
            xr, r, out, loss, count = AttackNSSDetectBatch(model, images, labels,
                eps=dconf['epsilon'], alpha=2./255., verbose=verbose,
                device=device, maxiter=dconf['maxiter'])
        else:
            raise ValueError(f"Attack {attack} is unsupported.")
        orig_correct += count[0]
        adv_correct += count[1]
        total += len(labels)
        if len(__TIME_PER_IMAGE__) > 0:
            print('Time Per Image:', np.mean(__TIME_PER_IMAGE__), np.std(__TIME_PER_IMAGE__))
    print('baseline=', '%.2f'%(100.*(orig_correct/total)),
            'adv=', '%.2f'%(100.*(adv_correct/total)),
            'advReduce=', '%.2f'%(100.*(orig_correct - adv_correct) / total))


def ClassAttackBatch(model, images, labels, *, eps, alpha=2./255., maxiter=1,
        verbose=False, device='cpu', atype=False, traj=False):
    assert(type(images) == th.Tensor)

    images = images.to(device)
    images_orig = images.clone().detach()
    images.requires_grad = True
    labels = labels.to(device).view(-1)

    # baseline: forward original samples
    model.eval()
    with th.no_grad():
        output, loss = model.loss(images, labels, adv=False)
        orig_correct = output.max(1)[1].eq(labels).sum().item()
        output_orig = output.clone().detach()
        loss_orig = loss.clone().detach()
    if verbose:
        cprint('* Orig Sample', 'white', 'on_magenta', end=' ')
        print(f'loss= {loss.item():.5f}', f'accuracy= {orig_correct/len(labels):.3f}')
        #print('> labels=', [x.item() for x in labels])
        #print('> predic=', [x.item() for x in output.max(1)[1]])

    # start attack
    if atype == 'UT:CW':
        xr, r = CW(model, images, labels, eps=eps,
                verbose=verbose, device=device)
    elif atype == 'UT:PGDl8':
        xr, r = PGDl8(model, images, labels, eps=eps, alpha=alpha,
                maxiter=maxiter, verbose=verbose, device=device,
                random_start=True)
    elif atype == 'UT:PGDl2':
        xr, r = PGDl2(model, images, labels, eps=eps, alpha=alpha,
                maxiter=maxiter, verbose=verbose, device=device,
                random_start=True)
    elif atype == 'UT:NES':
        xr, r = NES(model, images, labels, eps=eps, verbose=verbose)
    elif atype == 'UT:SPSA':
        xr, r = SPSA(model, images, labels, eps=eps, verbose=verbose)
    elif atype == 'UT:GA':
        xr, r = GaussianNoise(model, images, labels, eps=eps, verbose=verbose)
    elif atype == 'UT:UN':
        xr, r = UniformNoise(model, images, labels, eps=eps, verbose=verbose)
    elif atype == 'UT:MIM':
        xr, r = MIM(model, images, labels, eps=eps, alpha=alpha,
                maxiter=maxiter, verbose=verbose, device=device)
    else: # UT:PGD
        xr, r = projectedGradientDescent(model, images, labels, eps=eps,
                alpha=alpha, maxiter=maxiter, verbose=verbose,
                device=device, targeted=False, unbound=False)

    # adversarial: forward perturbed images
    model.eval()
    with th.no_grad():
        output, loss = model.loss(xr, labels, adv=False)
        adv_labels = output.max(1)[1].cpu().detach().numpy()
        adv_correct = output.max(1)[1].eq(labels).sum().item()
    if verbose:
        cprint('* Adve Sample', 'yellow', 'on_magenta', end=' ')
        print(f'loss= {loss.item():.5f}', f'accuracy= {adv_correct/len(labels):.3f}')
        #print('> labels=', [x.item() for x in labels])
        #print('> predic=', [x.item() for x in output.max(1)[1]])
    return xr, r, (output_orig, output), \
            (loss_orig, loss), (orig_correct, adv_correct)


def AttackLoliDetectBatch(model, images, labels, *, eps, alpha=2./255., maxiter=1,
        verbose=False, device='cpu', atype=False, traj=True, loader=None):
    assert(type(images) == th.Tensor)

    images = images.to(device)
    images_orig = images.clone().detach()
    images.requires_grad = True
    labels = labels.to(device).view(-1)

    # baseline: forward original samples
    model.eval()
    with th.no_grad():
        output, loss = model.loss(images, labels, adv=False)
        orig_correct = output.max(1)[1].eq(labels).sum().item()
        output_orig = output.clone().detach()
        loss_orig = loss.clone().detach()
    if verbose:
        cprint('* Orig Sample', 'white', 'on_magenta', end=' ')
        print(f'loss= {loss.item():.5f}', f'accuracy= {orig_correct/len(labels):.3f}')
        #print('> labels=', [x.item() for x in labels])
        #print('> predic=', [x.item() for x in output.max(1)[1]])

    # start attack
    # XXX: [switch-adbn]: start from AD or BN
    current_tag = os.getenv('TAG', None)
    if not current_tag:
        raise ValueError('please export a TAG for PGDT and continue')
    # [group 1] AD and BN
    print(f'Preparing <{current_tag}> adversarial examples')
    if current_tag == 'BN':
        xr, r = images, th.zeros_like(images)
    elif current_tag == 'AD':  # BIM
        xr, r = projectedGradientDescent(model, images, labels, eps=eps,
                alpha=2./255., maxiter=maxiter, verbose=verbose,
                device=device, targeted=False, unbound=False)
    elif current_tag == 'BIMl2':
    #    xr, r = projectedGradientDescent(model, images, labels, eps=eps,
    #            alpha=2./255., maxiter=maxiter, verbose=verbose,
    #            device=device, targeted=False, unbound=False, norm='l2')
        xr, r = BIMl2(model, images, labels, eps=eps,
                alpha=2./255., maxiter=maxiter, verbose=verbose)
    elif current_tag == 'FA':
        xr, r = projectedGradientDescent(model, images, labels, eps=eps,
                alpha=2./255., maxiter=maxiter, verbose=verbose,
                device=device, targeted=False, losstype='fa')
    elif current_tag == 'DLR':
        xr, r = projectedGradientDescent(model, images, labels, eps=eps,
                alpha=2./255., maxiter=maxiter, verbose=verbose,
                device=device, targeted=False, losstype='dlr')
    elif current_tag == 'PGDl8alt':
        xr, r = PGDl8alt(model, images, labels, eps=eps, alpha=2./255.,
                maxiter=maxiter, verbose=verbose, device=device)
    elif current_tag == 'APGD':
        xr, r = APGD(model, images, labels, eps=eps, alpha=2./255.,
                maxiter=maxiter, verbose=verbose)
    elif current_tag == 'APGDdlr':
        xr, r = APGDdlr(model, images, labels, eps=eps, alpha=2./255.,
                maxiter=maxiter, verbose=verbose)
    elif current_tag == 'FAB':
        xr, r = FAB(model, images, labels, eps=eps, alpha=2./255.,
                maxiter=maxiter, verbose=verbose, device=device)
    elif current_tag == 'AA':
        xr, r = AutoAttack(model, images, labels, eps=eps, alpha=2./255.,
                maxiter=maxiter, verbose=verbose, device=device)
    elif current_tag == 'FGSM':
        xr, r = projectedGradientDescent(model, images, labels, eps=eps,
                alpha=eps, maxiter=1, verbose=verbose,
                device=device, targeted=False, unbound=False)
    elif current_tag == 'CW':
        xr, r = CW(model, images, labels, eps=eps,
                verbose=verbose, device=device)
    elif current_tag == 'FMNl8':
        xr, r = FMNl8(model, images, labels, eps=eps,
                verbose=verbose, device=device)
    elif current_tag == 'NES':
        xr, r = NES(model, images, labels, eps=eps, verbose=verbose)
    elif current_tag == 'SPSA':
        xr, r = SPSA(model, images, labels, eps=eps, verbose=verbose)
    elif current_tag == 'PGDl8':
        xr, r = PGDl8(model, images, labels, eps=eps, alpha=alpha,
                maxiter=maxiter, verbose=verbose, device=device, random_start=True)
    elif current_tag == 'PGDl2':
        xr, r = PGDl2(model, images, labels, eps=eps, alpha=alpha,
                maxiter=maxiter, verbose=verbose, device=device, random_start=True)
    elif current_tag == 'MIM':
        xr, r = MIM(model, images, labels, eps=eps, alpha=alpha,
                maxiter=maxiter, verbose=verbose, device=device)
    elif current_tag == 'Square':
        xr, r = Square(model, images, labels, eps=eps, verbose=verbose, device=device)
    elif current_tag == 'DIFGSM':
        # [use model itself as transfer proxy]
        #xr, r = DIFGSM(model, images, labels, eps=eps, alpha=alpha,
        #        verbose=verbose, device=device)
        # [use real proxy]
        if images.shape[-1] == 32:
            # CIFAR10
            xr, r = DIFGSM_r18(images, labels, eps=eps, alpha=alpha,
                    verbose=verbose, device=device)
        elif images.shape[-1] == 224:
            xr, r = DIFGSM_r50(images, labels, eps=eps, alpha=alpha,
                    verbose=verbose, device=device)
        else:
            raise NotImplementedError
    elif current_tag == 'TIFGSM':
        # [this creates transferrable on model itself]
        #xr, r = TIFGSM(model, images, labels, eps=eps, alpha=alpha,
        #        verbose=verbose, device=device)
        # [this creates on proxy model]
        if images.shape[-1] == 32:
            # CIFAR10
            xr, r = TIFGSM_r18(images, labels, eps=eps, alpha=alpha,
                    verbose=verbose, device=device)
        elif images.shape[-1] == 224:
            # ImageNet
            xr, r = TIFGSM_r50(images, labels, eps=eps, alpha=alpha,
                    verbose=verbose, device=device)
        else:
            raise NotImplementedError
    elif current_tag == 'Jitter':
        xr, r = Jitter(model, images, labels, eps=eps, alpha=alpha, verbose=verbose)
    # [group 2] noise UN and GA
    elif current_tag == 'UN':
        xr, r = UniformNoise(model, images, labels, eps=eps, verbose=verbose)
    elif current_tag == 'GA':
        xr, r = GaussianNoise(model, images, labels, eps=eps, verbose=verbose)
    elif current_tag == 'LM':
        # logit matching as adaptive attack
        pm = th.randperm(len(loader.dataset))[:labels.nelement()]
        pmlab = th.tensor([loader.dataset[i][1] for i in pm]).to(device).view(-1)
        while any(labels == pmlab):
            pm = th.randperm(len(loader.dataset))[:labels.nelement()]
            pmlab = th.tensor([loader.dataset[i][1] for i in pm]).to(device).view(-1)
            # until label differs
        with th.no_grad():
            pmx = th.stack([loader.dataset[i][0] for i in pm])
            pmx = pmx.to(device)
            #print('pmx', pmx.shape)
            reference_logit = model.forward(pmx)
        xr, r = LogitMatchingl8(model, images, labels, eps=eps, verbose=verbose, logit=reference_logit)
    elif current_tag == 'IN':
        # interpolation as adaptive attack
        xr, r = Interpolate(model, images, labels, eps=eps, alpha=alpha, maxiter=maxiter, device=device, verbose=verbose)
    else:
        raise ValueError(f'TAG[{current_tag}] seems invalid.')
    # common part
    _trajmethod = 'pgdt'
    if os.getenv('ARC_TRAJ', ''):
        # options: pgdt, fa, gaussian, uniform
        _trajmethod = os.getenv('ARC_TRAJ')
    arc_data_split = os.getenv('ARC_DATA_SPLIT', 'test')
    if arc_data_split == 'train':
        labelleak = labels
    else:
        labelleak = None
    time_start_1 = time.time()
    thetraj, mlike, llike = exploitation_vectors(model, xr, device,
            trajmethod=_trajmethod, labelleak=labelleak)
    time_end_1 = time.time()
    tmp_accuracy = mlike.eq(labels).float().mean().item()
    ll_acc = llike.eq(labels).float().mean().item()
    c.print('* Adve Sample', f'accuracy= {tmp_accuracy:.3f}', f'llike.acc={ll_acc}')

    # adversarial: forward perturbed images
    model.eval()
    with th.no_grad():
        output, loss = model.loss(xr, labels, adv=False)
        adv_labels = output.max(1)[1].cpu().detach().numpy()
        adv_correct = output.max(1)[1].eq(labels).sum().item()
    #if verbose:
    #    cprint('* Adve Sample', 'yellow', 'on_magenta', end=' ')
    #    print(f'loss= {loss.item():.5f}', f'accuracy= {adv_correct/len(labels):.3f}')

    time_start_2 = time.time()
    arcM = compute_arc_matrix(model, thetraj, device)
    time_end_2 = time.time()
    time_elapsed_per_image = (time_end_2 - time_start_2 + time_end_1 - time_start_1) / images.shape[0]
    __TIME_PER_IMAGE__.append(time_elapsed_per_image)
    directory = os.getenv('DIR', '.')
    idx = save_arcm_file(directory, arcM)
    save_aux_information(directory, idx, labels.cpu().numpy(),
            mlike.cpu().numpy(), llike.cpu().numpy())
    if idx >= 4-1 and (images.shape[-1] == 28 or images.shape[-1] == 32):
        warnings.warn('Stopping early with 4 batches. please edit code for more data')
        exit()
    elif idx >= 4-1 and images.shape[-1] == 224:
        warnings.warn('Stopping early with 4 batches. please edit code for more data')
        exit()

    return xr, r, (output_orig, output), \
            (loss_orig, loss), (orig_correct, adv_correct)


def AttackNSSDetectBatch(model, images, labels, *, eps, alpha=2./255., maxiter=1,
        verbose=False, device='cpu', atype=False, traj=True):
    assert(type(images) == th.Tensor)

    images = images.to(device)
    images_orig = images.clone().detach()
    images.requires_grad = True
    labels = labels.to(device).view(-1)

    # baseline: forward original samples
    model.eval()
    with th.no_grad():
        output, loss = model.loss(images, labels, adv=False)
        orig_correct = output.max(1)[1].eq(labels).sum().item()
        output_orig = output.clone().detach()
        loss_orig = loss.clone().detach()
    if verbose:
        cprint('* Orig Sample', 'white', 'on_magenta', end=' ')
        print(f'loss= {loss.item():.5f}', f'accuracy= {orig_correct/len(labels):.3f}')
        #print('> labels=', [x.item() for x in labels])
        #print('> predic=', [x.item() for x in output.max(1)[1]])

    # start attack
    # XXX: [switch-adbn]: start from AD or BN
    current_tag = os.getenv('TAG', None)
    if not current_tag:
        raise ValueError('please export a TAG for PGDT and continue')
    # [group 1] AD and BN
    print(f'Preparing <{current_tag}> adversarial examples')
    if current_tag == 'BN':
        xr, r = images, th.zeros_like(images)
    elif current_tag == 'AD':  # BIM
        xr, r = projectedGradientDescent(model, images, labels, eps=eps,
                alpha=2./255., maxiter=maxiter, verbose=verbose,
                device=device, targeted=False, unbound=False)
    elif current_tag == 'APGD':
        xr, r = APGD(model, images, labels, eps=eps, alpha=2./255.,
                maxiter=maxiter, verbose=verbose)
    elif current_tag == 'AA':
        xr, r = AutoAttack(model, images, labels, eps=eps, alpha=2./255.,
                maxiter=maxiter, verbose=verbose, device=device)
    elif current_tag == 'PGDl8':
        xr, r = PGDl8(model, images, labels, eps=eps, alpha=alpha,
                maxiter=maxiter, verbose=verbose, device=device, random_start=True)
    elif current_tag == 'MIM':
        xr, r = MIM(model, images, labels, eps=eps, alpha=alpha,
                maxiter=maxiter, verbose=verbose, device=device)

    # adversarial: forward perturbed images
    model.eval()
    with th.no_grad():
        output, loss = model.loss(xr, labels, adv=False)
        adv_labels = output.max(1)[1].cpu().detach().numpy()
        adv_correct = output.max(1)[1].eq(labels).sum().item()
        cprint('* Adve Sample', 'yellow', 'on_magenta', end=' ')
        print(f'loss= {loss.item():.5f}', f'accuracy= {adv_correct/len(labels):.3f}')

    # common part
    directory = os.getenv('DIR', '.')

    time_start = time.time()
    feats1 = []
    feats2 = []
    feats3 = []
    for _images in xr:
        #print(_images_orig.shape)
        #img = V.transforms.ToPILImage()(_images_orig)
        img = denorm(_images).permute(1,2,0).detach().cpu().numpy()
        #print(img.shape, img.max())
        img = img * 255

        mscn_coef = nss.calculate_mscn_coefficients(img, 7, 7/6)
        mscn_histo = nss.mscn_histogram(mscn_coef, 81)
        feats1.append(mscn_histo)

        param2 = nss.calculate_ggd_aggd(img, 'GGD', kernel_size=7, sigma=7/6)
        feats2.append(param2)

        param3 = nss.calculate_brisque_features(img, kernel_size=7, sigma=7/6)
        feats3.append(param3)

    feats1 = np.vstack(feats1)
    feats2 = np.vstack(feats2)
    feats3 = np.vstack(feats3)
    time_end = time.time()
    __TIME_PER_IMAGE__.append( (time_end - time_start) / images.shape[0] )
    idx = save_arcm_file(directory, feats1, prefix=f'nss1')
    idx = save_arcm_file(directory, feats2, prefix=f'nss2')
    idx = save_arcm_file(directory, feats3, prefix=f'nss3')

    if idx >= 4-1 and (images.shape[-1] == 28 or images.shape[-1] == 32):
        warnings.warn('Stopping early with 4 batches. please edit code for more data')
        exit()
    elif idx >= 4-1 and images.shape[-1] == 224:
        warnings.warn('Stopping early with 4 batches. please edit code for more data')
        exit()

    return xr, r, (output_orig, output), \
            (loss_orig, loss), (orig_correct, adv_correct)
