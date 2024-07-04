# Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
# Source code will be released under Apache-2.0 license if paper accepted.
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
from sklearn.svm import SVC
import rich
c = rich.get_console()

if __name__ == "__main__":


    # according to NSS paper, method 3 is best performing. we use method 3
    _nss = '3'
    arcv0  = arc.load_arcv_dir('datanss/trn-ct-e0',  prefix=f'nss{_nss}', fmax=50)
    l0 = np.zeros(arcv0.shape[0])
    arcv2  = arc.load_arcv_dir('datanss/trn-ct-e2',  prefix=f'nss{_nss}', fmax=50)
    l2 = np.ones(arcv2.shape[0]) * 1
    arcv4  = arc.load_arcv_dir('datanss/trn-ct-e4',  prefix=f'nss{_nss}', fmax=50)
    l4 = np.ones(arcv4.shape[0]) * 2
    arcv8  = arc.load_arcv_dir('datanss/trn-ct-e8',  prefix=f'nss{_nss}', fmax=50)
    l8 = np.ones(arcv8.shape[0]) * 3
    arcv16 = arc.load_arcv_dir('datanss/trn-ct-e16', prefix=f'nss{_nss}', fmax=50)
    l16 = np.ones(arcv16.shape[0]) * 4
    ARCv = [arcv2, arcv4, arcv8, arcv16]
    Ls = [l2, l4, l8, l16]
    allV = [arcv0, *ARCv]
    allL = [l0, *Ls]

    xarcv0  = arc.load_arcv_dir('datanss/val-ct-e0',  prefix=f'nss{_nss}')
    xl0 = np.zeros(xarcv0.shape[0])
    xarcv2  = arc.load_arcv_dir('datanss/val-ct-e2',  prefix=f'nss{_nss}')
    xl2 = np.ones(xarcv2.shape[0]) * 1
    xarcv4  = arc.load_arcv_dir('datanss/val-ct-e4',  prefix=f'nss{_nss}')
    xl4 = np.ones(xarcv4.shape[0]) * 2
    xarcv8  = arc.load_arcv_dir('datanss/val-ct-e8',  prefix=f'nss{_nss}')
    xl8 = np.ones(xarcv8.shape[0]) * 3
    xarcv16 = arc.load_arcv_dir('datanss/val-ct-e16', prefix=f'nss{_nss}')
    xl16 = np.ones(xarcv16.shape[0]) * 4
    XARCv = [xarcv2, xarcv4, xarcv8, xarcv16]
    XLs = [xl2, xl4, xl8, xl16]

    SVM = []
    for (i, arcv) in enumerate(ARCv):
        c.print('[white bold on purple]>_<', 'ARCv', 'k', i+1)
        svm = SVC(kernel='sigmoid', gamma='auto', verbose=False)

        X = np.vstack([arcv0, arcv])
        Y = np.hstack([l0, Ls[i] > 0])

        # cifar10 res18 weight
        ws = np.array([1.01, 1.05, 1.11, 1.12])
        # imagenet res152 weight 
        #ws = np.array([1.05, 1.16, 1.39, 1.88])
        # imagenet swin weight
        #ws = np.array([1.05, 1.15, 1.38, 1.92])

        # assign weights
        weights = np.ones(len(Y))
        weights[np.argwhere(Y == 0)] = ws[i]

        # train
        svm.fit(X, Y, sample_weight=weights)
        SVM.append(svm)
        #pred = svm.predict(X)
        #print('Train : SVM accuracy', (pred == Y).mean())
        #tpr = ((pred>0)==(Y>0))[np.argwhere(Y>0)].mean()
        #fpr = (pred>0)[np.argwhere(Y==0)].mean()
        #print('Train : TPR / FPR', tpr, fpr)
        xX = np.vstack([xarcv0, XARCv[i]])
        xY = np.hstack([xl0, XLs[i]])
        pred = svm.predict(xX)
        tpr = ((pred>0)==(xY>0))[np.argwhere(xY>0)].mean()
        fpr = (pred>0)[np.argwhere(xY==0)].mean()
        print('Test : TPR / FPR', tpr, fpr)

    c.print('[white bold on purple]>_<', 'Ordinal regression')
    aX = np.vstack([xarcv0, *XARCv])
    aL = np.vstack([xl0, *XLs]).flatten()
    print(aX.shape, aL.shape)
    pred1 = SVM[0].predict(aX)
    pred2 = SVM[1].predict(aX)
    pred3 = SVM[2].predict(aX)
    pred4 = SVM[3].predict(aX)
    khat = pred1 + pred2 + pred3 + pred4
    print(khat.shape)
    mae = np.abs(khat - aL).mean()
    tpr = ((khat>0)==(aL>0))[np.argwhere(aL>0)].mean()
    fpr = (khat>0)[np.argwhere(aL==0)].mean()
    print('Test<OrdReg>: TPR / FPR', tpr, fpr)
