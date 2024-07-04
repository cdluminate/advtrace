'''
Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
Source code will be released under Apache-2.0 license if paper accepted.
'''
'''
ARC Features :: Input / Output Module
'''
import os
import sys
import numpy as np
import glob
from . import arcfeat
import rich
from rich.progress import track
c = rich.get_console()


def save_arcm_file(directory: str, arcM: np.ndarray, *, prefix='arcm') -> int:
    '''
    arcM = np.vstack([x.cpu().numpy().reshape(1, -1) for x in arcM])
    arcM = arcM.reshape(arcM.shape[0], -1)

    the returned idx is the currently using index number for saving file.
    '''
    if not os.path.exists(directory):
        os.mkdir(directory)
    # automatically probe existing arcm-*.txt files and bump the index number
    idx = 0
    for ii in range(99999):
        fpath = os.path.join(directory, f'{prefix}-{ii}.txt')
        if not os.path.exists(fpath):
            idx = ii
            break
    print('Save> arcM shape', arcM.shape)
    fpath = os.path.join(directory, f'{prefix}-{idx}.txt')
    np.savetxt(fpath, arcM)
    c.rule(f'ARC-M (idx:{idx}) Dumped at {fpath}.')
    return idx


def save_aux_information(directory: str, idx: int,
        labels: np.ndarray, mlike: np.ndarray, llike: np.ndarray):
    '''
    save auxiliary information.
    use in conjunction with save_arcm_file(...).
    labels: ground truth
    mlike: most likely prediction
    llike: least likely prediction
    '''
    fpath = os.path.join(directory, f'labels-{idx}.txt')
    np.savetxt(fpath, labels.flatten())
    fpath = os.path.join(directory, f'mlike-{idx}.txt')
    np.savetxt(fpath, mlike.flatten())
    fpath = os.path.join(directory, f'llike-{idx}.txt')
    np.savetxt(fpath, llike.flatten())


def load_arcm_dir(directory: str, *, return_rc:bool=False, fmax:int=None, verbose=False) -> np.ndarray:
    '''
    load txt matrices from directory/arcm-\d+.txt
    '''
    # sanity
    if not os.path.exists(directory):
        raise FileNotFoundError(f'{directory} does not exist')
    # start reading
    flist = sorted(glob.glob(os.path.join(directory, 'arcm-*.txt')))
    allc = []
    for fpath in flist:
        arcm = np.loadtxt(fpath)
        if len(arcm.shape) == 1:
            arcm = arcm.reshape(1, -1)
        allc.extend(arcm.tolist())
    if fmax is not None:
        allc = allc[:fmax]
    allc = np.array(allc)
    if verbose:
        c.print(f'[bold]ARCio:: Load data from {directory} <{allc.shape}>')
    if not return_rc:
        return allc
    else:
        r = allc.shape[0]
        n = int(np.sqrt(allc.shape[-1]))
        return allc, r, n


def load_arcv_dir(directory: str, *, prefix='arcv', fmax:int=None, verbose=False) -> np.ndarray:
    '''
    load txt vectors from directory/arcv-\d+.txt
    '''
    flist = sorted(glob.glob(os.path.join(directory, f'{prefix}-*.txt')))
    allc = []
    for fpath in flist:
        arcv = np.loadtxt(fpath)
        if len(arcv.shape) == 1:
            arcv = arcv.reshape(1, -1)
        allc.extend(arcv.tolist())
    if fmax is not None:
        allc = allc[:fmax]
    allc = np.array(allc)
    if verbose:
        c.print(f'ARCio:: Load ARCv from {directory} <{allc.shape}>')
    return allc


def load_aux_dir(directory: str, fmax:int=None, verbose=False) -> tuple:
    '''
    load auxiliary information files
    '''
    labels = []
    mlikes = []
    llikes = []
    for idx in range(99999):
        fpath = os.path.join(directory, f'labels-{idx}.txt')
        if not os.path.exists(fpath):
            break
        labels.append(np.loadtxt(fpath))
        fpath = os.path.join(directory, f'mlike-{idx}.txt')
        mlikes.append(np.loadtxt(fpath))
        fpath = os.path.join(directory, f'llike-{idx}.txt')
        llikes.append(np.loadtxt(fpath))
    labels = np.hstack(labels).astype(np.int)
    mlikes = np.hstack(mlikes).astype(np.int)
    llikes = np.hstack(llikes).astype(np.int)
    if fmax is not None:
        labels = labels[:fmax]
        mlikes = mlikes[:fmax]
        llikes = llikes[:fmax]
    if verbose:
        print('Aux>', labels.shape, mlikes.shape, llikes.shape)
    return labels, mlikes, llikes


def load_data_series(prefix: str, es:list= [0,2,4,8,16], fmax=None, verbose=False):
    '''
    load data for uninformed attack detection
    '''
    LAP = []
    LABEL = []
    for e in track(es, description='LOAD_SERIES_DATA'):
        dirname = f'{prefix}-e{e}'
        cachename = f'{dirname}.cache'
        if os.path.exists(cachename + '.npz'):
            lap = np.load(cachename + '.npz')['lap']
            LAP.append(lap)
            r = lap.shape[0]
            label = np.ones(r) * (e if e < 1 else np.log2(e))
            LABEL.append(label.astype(np.int))
            if verbose:
                print(f'{__name__}> loading cache {cachename + ".npz"}')
            continue
        #print(f'loading {dirname} for uninformed det ...')
        assert(os.path.exists(dirname))
        cos, r, n = load_arcm_dir(dirname, return_rc=True, fmax=fmax, verbose=verbose)
        lap = arcfeat.arcm2v(cos.reshape(r, n, n))
        #lap = lap[lap[:,1] < 1e3]
        #r = lap.shape[0]
        LAP.append(lap)
        label = np.ones(r) * (e if e < 1 else np.log2(e))
        LABEL.append(label.astype(np.int))
        if not os.path.exists(cachename):
            np.savez(cachename, lap=lap)
            print(f'{__name__}> written cache', cachename, 'shape', lap.shape)
    LAP = np.vstack(LAP)
    LABEL = np.hstack(LABEL)
    if verbose:
        print(f'{__name__}> Shape', LAP.shape, LABEL.shape)
    return LAP, LABEL


def load_aux_series(prefix: str, es:list=[0,2,4,8,16], fmax:int=None, verbose=False):
    '''
    load aux information
    '''
    LABEL, MLIKE, LLIKE = [], [], []
    for e in es:
        dirname = f'{prefix}-e{e}'
        lab, ml, ll = load_aux_dir(dirname, fmax=fmax, verbose=verbose)
        LABEL.append(lab)
        MLIKE.append(ml)
        LLIKE.append(ll)
    label = np.hstack(LABEL)
    mlike = np.hstack(MLIKE)
    llike = np.hstack(LLIKE)
    if verbose:
        print('{__name__}> AUX Shape', label.shape, mlike.shape, llike.shape)
    return label, mlike, llike
