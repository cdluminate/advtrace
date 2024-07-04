'''
Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
Source code will be released under Apache-2.0 license if paper accepted.
'''
import torch as th
from . import ct_res18
from . import il_r50
import torchattacks as atk
from .utils import *
import rich
c = rich.get_console()

__proxy_name = None
__proxy_model = None

def __check_load_proxy(mname: str, device: str = 'cpu'):
    '''
    check and load proxy model for transfer attack
    only load once on the first call
    '''
    global __proxy_name
    global __proxy_model
    proxies = {
        'res18': 'trained/ct_res18.transfer.sdth',
        'res50': 'trained/il_r50.transfer.sdth',
        }
    assert(mname in proxies.keys())
    if __proxy_model is not None:
        return
    if mname == 'res18':
        mpath = proxies[mname]
        model = ct_res18.Model().to(device)
        model.load_state_dict(th.load(mpath))
        __proxy_model = model
        __proxy_name = mname
    if mname == 'res50':
        mpath = proxies[mname]
        model = il_r50.Model().to(device)
        model.load_state_dict(th.load(mpath))
        __proxy_model = model
        __proxy_name = mname
    print(f'{__name__}: loaded {mpath} on the first call')


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


class ModelwPreprocess(th.nn.Module):
    def __init__(self, model):
        super(ModelwPreprocess, self).__init__()
        self.model = model
    def forward(self, input):
        return self.model(renorm(input))


def __DIFGSM(model, images, labels, *, eps=8./255., alpha=2./255.,
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


def DIFGSM_r18(images, labels, *, eps=8./255., alpha=2./255.,
        maxiter=20, verbose=False, device='cpu'):
    '''
    use res18 proxy model
    '''
    global __proxy_model
    __check_load_proxy('res18', device=device)
    return __DIFGSM(__proxy_model, images, labels, eps=eps, alpha=alpha,
            maxiter=maxiter, verbose=verbose, device=device)


def DIFGSM_r50(images, labels, *, eps=8./255., alpha=2./255.,
        maxiter=20, verbose=False, device='cpu'):
    '''
    use res50 proxy model
    '''
    global __proxy_model
    __check_load_proxy('res50', device=device)
    return __DIFGSM(__proxy_model, images, labels, eps=eps, alpha=alpha,
            maxiter=maxiter, verbose=verbose, device=device)


def __TIFGSM(model, images, labels, *, eps=8./255., alpha=2./255.,
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


def TIFGSM_r18(images, labels, *, eps=8./255., alpha=2./255.,
        maxiter=20, verbose=False, device='cpu'):
    '''
    create adv image from res18
    note, different from __TIFGSM, the first model param has been dropped.
    '''
    global __proxy_model
    __check_load_proxy('res18', device=device)
    return __TIFGSM(__proxy_model, images, labels, eps=eps, alpha=alpha,
            maxiter=maxiter, verbose=verbose, device=device)


def TIFGSM_r50(images, labels, *, eps=8./255., alpha=2./255.,
        maxiter=20, verbose=False, device='cpu'):
    '''
    create adv image from res50
    '''
    global __proxy_model
    __check_load_proxy('res50', device=device)
    return __TIFGSM(__proxy_model, images, labels, eps=eps, alpha=alpha,
            maxiter=maxiter, verbose=verbose, device=device)

