'''
Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
Source code will be released under Apache-2.0 license if paper accepted.
'''
'''
ARC Feature :: Computation Module
'''
import os
import sys
import numpy as np
import torch as th
import itertools as it
import functools as ft
from scipy.optimize import curve_fit
from .pgdt import *
import gc
import fcntl
import rich
from rich.progress import track
c = rich.get_console()


def exploitation_vectors(model, images, device, *,
        trajmethod='pgdt', labelleak=None):
    '''
    Get a series of explitation vectors
    input(images): [B, C, H, W]
    output(thetraj): [maxiter_traj+1, B, C, H, W]

    design space: use what kind of attack to incur sequel attack effect.
    '''
    assert(trajmethod in ('pgdt', 'pgdtl2', 'pgdtl8', 'pgdtl8n', 'fa',
        'dlr', 'gaussian', 'uniform', 'cwt'))
    model.eval()
    eps_traj = 8./255.
    alpha_traj = 2./255.
    maxiter_traj = 6
    with th.no_grad():
        # we do not know the labels of the given image.
        # hence use the model decision.
        output = model.forward(images).detach()
        mostlike = output.max(1)[1]
        leastlike = output.min(1)[1]
    # dispatch the label to use
    if not os.getenv('ARC_LABEL', ''):
        uselabel = leastlike if labelleak is None else labelleak
    elif os.getenv('ARC_LABEL') == 'mlike':
        uselabel = mostlike
    elif os.getenv('ARC_LABEL') == 'llike':
        uselabel = leastlike
    elif os.getenv('ARC_LABEL') == 'rand':
        if images.shape[-1] == 224:
            uselabel = th.randint_like(mostlike, 0, 1000).long().to(mostlike.device)
        elif images.shape[-1] == 32:
            uselabel = th.randint_like(mostlike, 0, 10).long().to(mostlike.device)
        else:
            raise ValueError('do not know how to set label for such images.shape')
    else:
        raise ValueError('ARC_LABEL')
    # get trajectory
    if trajmethod == 'pgdt':
        # [note] losstype in {ce, fa, ...}
        xr, r, thetraj = PGDtraj(model, images, uselabel,
                eps=eps_traj, alpha=alpha_traj, maxiter=maxiter_traj,
                verbose=False, device=device, losstype='ce')
    elif trajmethod == 'pgdtl2':
        #xr, r, thetraj = PGDtraj(model, images, uselabel,
        #        eps=eps_traj, alpha=alpha_traj, maxiter=maxiter_traj,
        #        verbose=False, device=device, losstype='ce', norm='l2')
        xr, r, thetraj = BIMl2T(model, images, uselabel, maxiter=maxiter_traj)
    elif trajmethod == 'pgdtl8':
        # just for sanity check
        xr, r, thetraj = BIMl8T(model, images, uselabel, maxiter=maxiter_traj)
    elif trajmethod == 'pgdtl8n':
        # just for sanity check
        xr, r, thetraj = BIMl8Tn(model, images, uselabel, maxiter=maxiter_traj)
    elif trajmethod == 'fa':
        xr, r, thetraj = PGDtraj(model, images, uselabel,
                eps=eps_traj, alpha=alpha_traj, maxiter=maxiter_traj,
                verbose=False, device=device, losstype='fa')
    elif trajmethod == 'dlr':
        xr, r, thetraj = PGDtraj(model, images, uselabel,
                eps=eps_traj, alpha=alpha_traj, maxiter=maxiter_traj,
                verbose=False, device=device, losstype='dlr')
    elif trajmethod == 'cwt':
        xr, r, thetraj = CWTraj(model, images, uselabel)
    elif trajmethod in ('uniform', 'gaussian'):
        # XXX: no separation
        thetraj = []
        thetraj.append(images.clone().detach())
        for step in range(maxiter_traj):
            if trajmethod == 'uniform':
                r = alpha_traj * (2 * th.rand(images.shape).to(device) - 1)
            elif trajmethod == 'gaussian':
                r = th.clamp(th.randn(images.shape).to(device) * eps_traj / 3.0,
                        min=-eps_traj, max=eps_traj)
            else:
                raise NotImplementedError
            assert(images.min() < 0) # assume imagenet preprocessing
            xr = images + r/IMstd[:,None,None].to(device)
            xr = th.max(xr, renorm(th.zeros(images.shape, device=device)))
            xr = th.min(xr, renorm(th.ones(images.shape, device=device)))
            thetraj.append(xr.clone().detach())
        thetraj = th.stack(thetraj)
    else:
        raise Exception(trajmethod)
    #print('DEBUG', thetraj.shape)
    return thetraj, mostlike, leastlike


def compute_arc_matrix(model, thetraj, device):
    '''
    Compute arc matrix from the trajectory composed of exploitation vector
    then dump results to arcm-*.txt
    input(thetraj) shape [T+1, Batch, C, H, W]

    This part is feature-frozen [Apr. 10]
    '''
    model.eval()
    # data shape inference.
    assert(len(thetraj.shape) == 5)
    Nstep = thetraj.shape[0]
    Nbatch = thetraj.shape[1]
    if thetraj.shape[-1] == 32 and \
            thetraj.shape[-2] == 32 and \
            thetraj.shape[-3] == 3:
        # CIFAR10
        Nclass = 10
        Nflatten = 3*32*32
    elif thetraj.shape[-1] == 28 and \
            thetraj.shape[-2] == 28 and \
            thetraj.shape[-3] == 1:
        # MNIST
        Nclass = 10
        Nflatten = 1*28*28
    elif thetraj.shape[-1] == 224 and \
            thetraj.shape[-2] == 224 and \
            thetraj.shape[-3] == 3:
        # ImageNet
        Nclass = 1000
        Nflatten = 3*224*224
    else:
        raise NotImplementedError
    # start computing ARC Matrices for every sample in the batch
    arcM = []
    # [arc method 1: jacobian]
    def _compute_jx_from_isample_jacobian(isample):
        Jx = []
        for istep in range(thetraj.size(0)):
            gc.collect()
            if thetraj.shape[-1] == 224:
                print('isample', isample, 'istep', istep+1, '/', thetraj.size(0))
            xi = thetraj[istep, isample, :, :, :]  # [1, 1, C, H, W]
            if len(xi.shape) == 3:
                xi = xi.unsqueeze(0) # [1, C, H, W]
            jxi = th.autograd.functional.jacobian(model, xi, create_graph=False)
            # [D, 1, C, H, W]
            jxi = jxi.view(-1).clone().detach().cpu().to(th.float32)
            # print('jxi.nbytes', jxi.nbytes) (3*224*224)*1000*17*2
            Jx.append(jxi) # D*C*H*W
        return Jx

    # start computing arc
    for isample in track(range(thetraj.size(1)), description="ARC-M::isample"):
        gc.collect()
        #batch = thetraj[:, isample, :, :, :]  # [steps, 1, C, H, W]
        #print('batch.shape', batch.shape)
        #with th.no_grad():
        #    trajoutput = model.forward(batch)
        #    trajpred = trajoutput.max(1)[1].cpu().detach().numpy()
        #print('isample', isample)
        #print('TrajPred:', trajpred)
        Jx = _compute_jx_from_isample_jacobian(isample)

#        # [BEGIN METHOD 2]
#        # acquire lock to avoid oom
#        lock = open('/dev/shm/X-arcfeat-matrix-compute-oom.lock', 'w')
#        fcntl.lockf(lock, fcntl.LOCK_EX)
#        with th.no_grad():
#            # [the following th.stack costs lots of memory]                     
#            #Jx = th.stack(Jx).view(Nstep, -1)
#            #Jxn = th.nn.functional.normalize(Jx)
#            #cos = th.mm(Jxn, Jxn.T).detach()
#            # [this version should be able to save some memory]
#            cos = th.zeros(len(Jx), len(Jx))
#            for i in range(len(Jx)):
#                Jx[i] = th.nn.functional.normalize(Jx[i].view(1, -1))
#            for i, j in it.product(range(len(Jx)), range(len(Jx))):
#                cos[i,j] = th.dot(Jx[i].view(-1), Jx[j].view(-1))
#            # [end two versions]
#            arcM.append(cos)
#        # release lock
#        fcntl.lockf(lock, fcntl.LOCK_UN)
#        lock.close()
#        # [END METHOD 2]
#        continue
        # [BEGIN METHOD 1]
        # acquire lock to avoid oom
        lock = open('/dev/shm/X-arcfeat-matrix-compute-oom.lock', 'w')
        fcntl.lockf(lock, fcntl.LOCK_EX)

        with th.no_grad():
            # [the following th.stack costs lots of memory]                     
            #Jx = th.stack(Jx) # [steps, D*C*H*W], or [steps, Nclass * Nflatten 
            #print('Jx', Jx.shape) # partial y / partial x, Jacobian flattend.  
            # [step, Nclass * Nflatten]                                         
            #plt.figure()                                                       
            tmp = []
            for k in range(Nclass):
                # Jxk [steps, Nflatten] for each class                          
                # Jxk = Jx[:, k*Nflatten:(k+1)*Nflatten]  # Jx -- k-th class column group
                Jxk = th.stack([x[k*Nflatten:(k+1)*Nflatten] for x in Jx])  # save memory
                Jxkn = th.nn.functional.normalize(Jxk)
                cos = th.mm(Jxkn, Jxkn.T)  # this class, cross step             
                tmp.append(cos.clone().detach().cpu())  # cross step cos for this class
                # cos shape [steps, steps]                                      
            # tmp shape [Nclass, steps, steps]                                      
            #plt.show()                                                             
            if True:
                # take max instead of all                                       
                sums = [x.sum().item() for x in tmp]
                argsort = np.argsort(sums)[::-1]
                tmp = tmp[argsort[0]]
                tmp = [tmp]
        # mean across class
        arcM.append(th.stack(tmp).mean(0).detach())

        # release lock
        fcntl.lockf(lock, fcntl.LOCK_UN)
        lock.close()
        # [END METHOD 1]

    arcM = np.vstack([x.cpu().numpy().reshape(1, -1) for x in arcM])
    arcM = arcM.reshape(arcM.shape[0], -1)
    return arcM


def remove_diagonal(rows: np.ndarray) -> np.ndarray:
    '''
    shape (r, c), a square matrix per row, column = mat size ^ 2
    https://www.codegrepper.com/code-examples/python/pytorch+get+non+diag+element
    '''
    assert(len(rows.shape) == 2)
    r, c = rows.shape
    n = int(np.sqrt(c))
    #ret = []
    #for x in rows:
    #    n = x.shape[0]
    #    y = x.flatten()[1:].reshape(n-1, n+1)[:,:-1]
    #    ret.append(y)
    #return np.array([x.flatten()[1:].reshape(x.shape[1]-1,x.shape[1]+1)[:,:-1]
    #        for x in rows])
    return rows[:, 1:].reshape(r, n-1, n+1)[:,:,:-1].reshape(r, n*(n-1))


def remove_2diag(arcm: np.ndarray) -> np.ndarray:
    '''
    remove 0-diag and 1-diag
    '''
    assert(len(arcm.shape) == 2)
    assert(arcm.shape[0] == arcm.shape[1])
    #a = arcm.flatten()[1:].reshape(n-1, n+1)[:,:-1]
    l = np.tril(arcm, k=-1)[1:,:-1]
    u = np.triu(arcm, k=2)[:-1,1:]
    return l+u


def arcm2v(arcm: np.ndarray, diagonly: bool=False) -> np.ndarray:
    '''
    convert arc matrix to arc vector (2d)

    This part is feature-frozen. [Apr. 10]
    '''
    def laplace(x, *p):
        A, scale = p
        return A * np.exp(-np.abs(x/scale))
    if len(arcm.shape) == 2:
        assert(arcm.shape[0] == arcm.shape[1])
        if False:
            arcm = remove_2diag(arcm)
        n = arcm.shape[0]
        mr = np.arange(n).reshape(n,1).repeat(n, axis=1)
        mc = np.arange(n).reshape(1,n).repeat(n, axis=0)
        x = np.abs(mr - mc).reshape(-1)
        y = arcm.reshape(-1)
        if diagonly:
            x = np.diag(np.fliplr(np.abs(mr-mc)))
            y = np.diag(np.fliplr(arcm))
        coef, var_matrix = curve_fit(laplace,
                x, y, p0=[1.0, 1.0], method='lm')
        return coef
    elif len(arcm.shape) == 3:
        batch = arcm.shape[0]
        coefs = np.stack([arcm2v(m) for m in arcm])
        return coefs
    else:
        raise ValueError(f'arcm size irregular', arcm.shape)
