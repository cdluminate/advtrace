'''
Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
Source code will be released under Apache-2.0 license if paper accepted.
'''
import matplotlib as mpl
# mpl.use('Agg')  # XXX: this is for headless mode.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch as th
import os
import sys
from .arcfeat import *
from .arcio import *
import rich
c = rich.get_console()

CMAP='jet'

def plot_arcm(bndir: str, fdump: str = None, *, verbose: bool=False):
    '''
    show mean cosine matrix over a directory
    '''
    # load data
    #cbn_ = arcfeat.load_cosine_data(bndir, 'bn')
    cbn_ = load_arcm_dir(bndir)
    n = int(np.sqrt(cbn_.shape[-1]))
    # calculate mean
    mcbn = cbn_.mean(axis=0).reshape(n, n)
    # start plotting
    plt.figure(figsize=(5,4))
    plt.pcolormesh(mcbn, cmap=CMAP, vmin=0., vmax=1.)
    plt.colorbar()
    plt.axis('square')
    plt.axis('off')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    if fdump:
        plt.savefig(fdump)
    else:
        plt.show()
    # surface plot 3d
    _x, _y = np.arange(n), np.arange(n)
    x, y = np.meshgrid(_x, _y)
    z = mcbn
    print('x', x.shape, 'y', y.shape, 'z', z.shape)

    from matplotlib import cm
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #ax.plot_surface(x, y, z, cmap=cm.coolwarm, antialiased=True)
    ax.bar3d(x.ravel(), y.ravel(), np.zeros_like(z.ravel()), 1, 1,
            z.ravel(), shade=True)
    plt.show()

    '''
    show scatter 2-d
    '''
    arcm = cbn_
    plt.figure()
    #pca = PCA(n_components=2)
    #red = pca.fit_transform(arcm.reshape(r, -1))
    arcv = arcm2v(arcm.reshape(-1, n, n))
    print(arcv.shape)
    plt.scatter(arcv[:,1], arcv[:,0])
    plt.tight_layout()
    plt.show()



def plot_pair_arcm(bndir: str, addir: str, fdump: str = None,
        *, verbose: bool = False):
    '''
    a pair of directory for comparison
    '''
    arcmb, rb, nb = load_arcm_dir(bndir, return_rc=True)
    marcb = arcmb.mean(axis=0).reshape(nb, nb)
    arcma, ra, na = load_arcm_dir(addir, return_rc=True)
    marca = arcma.mean(axis=0).reshape(na, na)
    plt.figure(figsize=(9,3))
    for (i, m) in enumerate([marcb, marca], 1):
        plt.subplot(1,3,i)
        plt.pcolormesh(m, cmap=CMAP, vmin=0., vmax=1.)
        plt.colorbar()
        plt.axis('square')
        plt.gca().invert_yaxis()
    plt.subplot(1,3,3)

    # aux info
    gt, ml, ll = load_aux_dir(addir)
    print('ml-Acc.', (gt == ml).mean(), 'll-Acc.', (gt==ll).mean())
    fail_mask = (gt == ml)
    print('num-fail', fail_mask.sum())
    #

    #pca = PCA(n_components=2)
    arcm = np.stack([arcmb, arcma])
    #red = pca.fit_transform(arcm.reshape(rb+ra, -1))
    red = arcm2v(arcm.reshape(rb+ra, nb, nb))
    label = [0]*rb + [1]*ra
    color = ['blue' if i == 0 else 'red' for i in label]
    print('ARCv:', red.shape)

    for i in np.argwhere(fail_mask).flatten():
        color[i] = 'lime'

    plt.scatter(red[:,1], red[:,0], color=color, s=3)

    if fdump:
        plt.savefig(fdump)
    else:
        plt.show()


def plot_quad_arcm(prefix: str, fpath: str = None, *, verbose=False, noscatter=False):
    '''
    show 4 arcm pcolormesh & scatters.
    '''
    if not noscatter:
        arc0, r0, _ = load_arcm_dir(prefix + '-e0',  return_rc=True, fmax=512)
    arc2, r2, n = load_arcm_dir(prefix + '-e2',  return_rc=True, fmax=512)
    arc4, r4, _ = load_arcm_dir(prefix + '-e4',  return_rc=True, fmax=512)
    arc8, r8, _ = load_arcm_dir(prefix + '-e8',  return_rc=True, fmax=512)
    arc6, r6, _ = load_arcm_dir(prefix + '-e16', return_rc=True, fmax=512)
    if noscatter:
        plt.figure(figsize=(8,2))
    else:
        plt.figure(figsize=(15,2))
    for (i, arc) in enumerate([arc2, arc4, arc8, arc6], 1):
        marc = arc.mean(axis=0).reshape(n, n)
        if noscatter:
            plt.subplot(1,4,i)
        else:
            plt.subplot(1,8,2*i-1)
        plt.pcolormesh(marc, cmap=CMAP, vmin=0., vmax=1.)
        #plt.colorbar(location='left')
        plt.axis('square')
        plt.gca().invert_yaxis()
        plt.axis('off')
        plt.title('ARCm $\\varepsilon='+f'{2**i}'+'/255$')

        if noscatter:
            continue
        else:
            plt.subplot(1, 8, 2*i)

        arcv = np.vstack([arcm2v(arc0.reshape(-1, n, n)),
                arcm2v(arc.reshape(-1, n, n))])
        label = [0] * r0 + [1] * [r2, r4, r8, r6][i-1]
        color = ['blue' if i == 0 else 'red' for i in label]
        perm = np.random.permutation(len(label))

        arcv = arcv[perm]
        label = np.array(label)[perm]
        color = ['blue' if i == 0 else 'red' for i in label]

        plt.scatter(arcv[:,1], arcv[:,0], color=color, s=1)
        plt.title('ARCv $\\varepsilon='+f'{2**i}'+'/255$')

    plt.tight_layout()
    if fpath:
        c.print(f'[bold yellow]dumping SVG to {fpath}')
        plt.savefig(fpath)
    else:
        c.print('showing directly since --save is not specified')
        plt.show()


def plot_polar_arcm(prefix: str, fpath: str = None, *, verbose=False):
    '''
    show 4 arcm pcolormesh & scatters.
    '''
    arc2, r2, n = load_arcm_dir(prefix + '-e2',  return_rc=True, fmax=512)
    arc6, r6, _ = load_arcm_dir(prefix + '-e16', return_rc=True, fmax=512)
    plt.figure(figsize=(4,2))
    for (i, arc) in enumerate([arc2, arc6], 1):
        marc = arc.mean(axis=0).reshape(n, n)
        plt.subplot(1,2,i)
        plt.pcolormesh(marc, cmap=CMAP, vmin=0., vmax=1.)
        #plt.colorbar(location='left')
        plt.axis('square')
        plt.gca().invert_yaxis()
        plt.axis('off')
        __p = {1: 1, 2: 4}[i]
        plt.title('ARCm $\\varepsilon='+f'{2**__p}'+'/255$')

    plt.tight_layout()
    if fpath:
        c.print(f'[bold yellow]dumping SVG to {fpath}')
        plt.savefig(fpath)
    else:
        c.print('showing directly since --save is not specified')
        plt.show()


def plot_loli(bndir: str, addir: str, fpath: str = None, *, verbose=False):
    '''
    show ARC features and scatter
    '''
    raise Exception('deprecated')
    cad_ = arcfeat.load_cosine_data(addir, 'ad')
    cbn_ = arcfeat.load_cosine_data(bndir, 'bn')
    r1, c = cad_.shape
    r2 = cbn_.shape[0]
    n = int(np.sqrt(c))
    print(f'r1 {r1} c {c} n {n}')
    mcad_ = cad_.mean(axis=0).reshape(n, n)
    mcbn_ = cbn_.mean(axis=0).reshape(n, n)

    # pcolor cosine for AD
    f, (a0, a1, a2) = plt.subplots(3,1,figsize=(3,7), gridspec_kw={'height_ratios': [3,1,3]})
    a0.pcolormesh(mcad_, cmap=CMAP, vmin=0., vmax=1.)
    #plt.colorbar()
    a0.axis('square')
    a0.set_ylim(a0.get_ylim()[::-1])
    a0.title.set_text('ARC Matrix')
    #a0.set_xlabel('Step')
    #a0.set_ylabel('Step')

    # stem + regression for AD
    ts = np.arange(n)
    odiag_ad = arcfeat.cm2arc(mcad_)
    odiag_bn = arcfeat.cm2arc(mcbn_)
    ml, sl, bl = a1.stem(ts, odiag_ad, linefmt='grey', markerfmt='ro', label='Adversarial')
    plt.setp(sl, 'linestyle', 'dotted')
    a1.plot(ts, odiag_bn, 'bx', label='Benign')
    # stem fit
    assert(len(odiag_ad) % 2 == 1)
    loc = int(len(odiag_ad) / 2.)
    lapad, coefad = arcfeat.laplace_func_fit(ts, odiag_ad, loc)
    print(f'LAPLACIAN[loc={loc}]', coefad)
    fitad = lapad(ts, *coefad)
    #a1.plot(ts, fitad, 'yellow')
    a1.set_ylabel('Cosine')
    a1.title.set_text('ARC Vector')
    a1.legend(prop={'size': 6}, loc='upper right')

    # scatter for AD BN
    diagbn = arcfeat.cm2arc(cbn_.reshape(r2, n, n))
    print('shape diagbn', diagbn.shape)
    diagad = arcfeat.cm2arc(cad_.reshape(r1, n, n))
    print('shape diagad', diagad.shape)

    no_center_stem = False
    if no_center_stem:
        diagbn = np.hstack([diagbn[:, :loc], diagbn[:, loc+2:]])
        diagad = np.hstack([diagad[:, :loc], diagad[:, loc+2:]])
        n = n - 2
        print('no_center_stem', diagbn.shape, diagad.shape)

    lapbn = arcfeat.arc2laplace(diagbn)
    lapad = arcfeat.arc2laplace(diagad)
    print('fit shape', lapbn.shape, lapad.shape)

    a2.scatter(lapad[:,1], lapad[:,0], color='r', s=4, label='Adversarial')
    a2.scatter(lapbn[:,1], lapbn[:,0], color='b', s=4, label='Benign')
    #a2.legend(loc='lower left')
    a2.legend()
    a2.set_ylabel('Laplacian: $\\alpha$')
    a2.set_xlabel('Laplacian: $\\sigma$')
    #a2.set_ylim([0.8, 1.1])
    #a2.set_xlim([0, 42])
    #a2.set_xlim([0, 18])
    a2.title.set_text('Laplacian Parameter')
    plt.tight_layout()

    if fpath:
        plt.savefig(fpath)
    else:
        plt.show()


def plot_roc(fpath: str = None):
    '''
    draw the ROC curve
    '''
    # [CIFAR10/ResNet18]
    raise Exception('deprecated')
    data_uad_ct = np.array([
        # weight, TPR FPR
        [0, 0.0, 0.0],
        [512, 0.4859, 0.0019],
        [256, 0.488037109375, 0.001953125],
        [128, 0.53271484375, 0.001953125],
        [64, 0.575439453125, 0.00390625],
        [32, 0.607666015625, 0.0078125],
        [16,  0.6407, 0.0097],
        [8, 0.6636962890625, 0.013671875],
        [4, 0.702880859375, 0.041015625],
        [2, 0.7646484375, 0.11328125],
        [1.5, 0.792236328125, 0.197265625],
        [1.25, 0.827880859375, 0.306640625],
        [1, 0.8751220703125, 0.515625],
        [0.75, 0.940673828125, 0.783203125],
        [0.5, 1.0, 1.0],
        [0, 1.0, 1.0],
        ])
    data = data_uad_ct
    print(data)
    plt.plot(data[:,2], data[:,1], 'ko-')
    plt.plot([0,1], [0,1], 'k:')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    #plt.title('ROC Curve')
    plt.axis('square')
    plt.show()
