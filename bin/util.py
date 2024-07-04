'''
Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
Source code will be released under Apache-2.0 license if paper accepted.
'''
import numpy as np
import torch as th
import os
import sys
sys.path.extend(['.', '..'])
import path
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
import functools as ft
import rich, rich.progress
c = rich.get_console()
import arc

if __name__ == "__main__":
    ag = argparse.ArgumentParser()
    ag.add_argument('action', type=str)
    ag.add_argument('-b', '--bndir', type=str, default=None)
    ag.add_argument('-a', '--addir', type=str, default=None)
    ag.add_argument('-p', '--prefix', type=str, default=None)
    ag.add_argument('-s', '--save', type=str, default=None)
    ag.add_argument('-l', '--load', type=str, default=None)
    ag.add_argument('-v', '--verbose', action='store_true')
    ag = ag.parse_args()

    if ag.action == 'arcm':
        arc.plot_arcm(ag.bndir, ag.save, verbose=ag.verbose)
    elif ag.action == 'p-arcm':
        arc.plot_pair_arcm(ag.bndir, ag.addir, ag.save, verbose=ag.verbose)
    elif ag.action == 'q-arcm':
        arc.plot_quad_arcm(ag.prefix, ag.save, verbose=ag.verbose)
    elif ag.action == 'qx-arcm':
        arc.plot_quad_arcm(ag.prefix, ag.save, verbose=ag.verbose, noscatter=True)
    elif ag.action == 'polar-arcm':
        arc.plot_polar_arcm(ag.prefix, ag.save, verbose=ag.verbose)
    elif ag.action == 'loli':
        arc.plot_loli(ag.bndir, ag.addir, ag.save, verbose=ag.verbose)
    elif ag.action == 'train':
        arc.train_svm(ag.bndir, ag.addir, ag.save, verbose=ag.verbose)
    elif ag.action == 'val':
        arc.validate_svm(ag.bndir, ag.addir, ag.load, verbose=ag.verbose)
    elif any(ag.action.startswith(key) for key in ('svm', 'svr', 'zs')):
        # e.g. train and test BIM svm-ct-ad
        #      test APGD zsm-ct-apgd
        arc.detect(ag.action, verbose=ag.verbose)
    elif ag.action == 'roc':
        arc.plot_roc(ag.save)
    elif ag.action == 'test':
        arcv, labels = arc.load_data_series('data/trn-ct')
    else:
        raise ValueError(f'unknown action {ag.action}')
