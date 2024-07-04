'''
Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
Source code will be released under Apache-2.0 license if paper accepted.
'''
import os
import sys
from .arcfeat import *
from .arcio import *
import numpy as np
from sklearn import svm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle as pkl
from typing import *
import rich
from rich.progress import track
from rich.table import Table
c = rich.get_console()


def train_svm(bndir: str, addir: str, save:str = None, *, verbose=False):
    '''
    train SVM classifier, binary.
    https://scikit-learn.org/stable/modules/svm.html#classification
    '''
    assert(os.path.exists(bndir))
    assert(os.path.exists(addir))
    cbn = arc.load_cosine_data(bndir, 'bn')
    cad = arc.load_cosine_data(addir, 'ad')
    assert(len(cad.shape) == 2)
    r1, c = cad.shape
    n = int(np.sqrt(c))
    r2, _ = cbn.shape

    lapA = arc.cm2laplace(cad.reshape(r1, n, n))
    lapA = lapA[lapA[:,1]<1e3] # remove outlier
    #lapA = lapA[lapA[:,1]>1e1]
    r1 = lapA.shape[0]
    lapB = arc.cm2laplace(cbn.reshape(r2, n, n))
    X = np.vstack([lapA, lapB])
    #print('outlier', np.argwhere(X[:,1]>1e3))
    print('lapA<1>', lapA.shape, 'lapB<0>', lapB.shape, 'X', X.shape)

    label = np.concatenate([np.ones(r1), np.zeros(r2)])
    #print('labels', label, label.shape)

    #clf = svm.SVC()
    clf = svm.SVC(kernel='linear', verbose=False)  # better.
    clf.fit(X, label)
    print('support vectors', clf.n_support_)

    # predict on training set
    yhat = clf.predict(X)
    print('training Accuracy', (yhat==label).mean())
    print('training TPR', (yhat==label)[np.argwhere(label==1)].mean())
    print('training FPR', (yhat!=label)[np.argwhere(label==0)].mean())

    # https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane_unbalanced.html
    plt.scatter(X[:, 1], X[:, 0], c=label, cmap=plt.cm.Paired, edgecolors='k')
    #ax = plt.gca()
    #xlim = ax.get_xlim()
    #ylim = ax.get_ylim()
    #xx = np.linspace(xlim[0], xlim[1], 30)
    #yy = np.linspace(ylim[0], ylim[1], 30)
    #YY, XX = np.meshgrid(yy, xx)
    #xy = np.vstack([XX.ravel(), YY.ravel()]).T
    #Z = clf.decision_function(xy).reshape(XX.shape)
    #a = ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"])
    if verbose:
        plt.show()

    # confidence score
    dec = clf.decision_function(X)
    avgscore = (dec * (label*2 - 1)).mean()
    print('avg conf score', avgscore, dec[yhat==0].mean(), dec[yhat==1].mean())
    plt.stem(dec)
    #if verbose:
    #    plt.show()

    # roc curve
    from sklearn.metrics import RocCurveDisplay
    # this requires sklearn > 1.0.0
    RocCurveDisplay.from_predictions(label, yhat)
    #if verbose:
    #    plt.show()

    if save:
        with open(save, 'wb') as f:
            s = pkl.dump(clf, f)
        print(f'PKL: written SVM to {save}')
    return clf

def validate_svm(bndir: str, addir: str, load: str = None, *, verbose=False):
    '''
    load trained SVM and validate.
    '''
    assert(load)
    assert(os.path.exists(bndir))
    assert(os.path.exists(addir))
    cbn = arc.load_cosine_data(bndir, 'bn')
    cad = arc.load_cosine_data(addir, 'ad')
    assert(len(cad.shape) == 2)
    r1, c = cad.shape
    n = int(np.sqrt(c))
    r2, _ = cbn.shape

    # go to laplace param space
    lapA = arc.cm2laplace(cad.reshape(r1, n, n))
    lapA = lapA[lapA[:,1]<1e3]
    r1 = lapA.shape[0]
    lapB = arc.cm2laplace(cbn.reshape(r2, n, n))
    X = np.vstack([lapA, lapB])
    print('lapA<1>', lapA.shape, 'lapB<0>', lapB.shape, 'X', X.shape)

    label = np.concatenate([np.ones(r1), np.zeros(r2)])
    #print('labels', label, label.shape)

    # load SVM
    with open(load, 'rb') as f:
        clf = pkl.load(f)
    print('[SVM] support vectors', clf.n_support_)

    # predict on training set
    yhat = clf.predict(X)
    print('val Accuracy', (yhat==label).mean())
    print('val TPR', (yhat==label)[np.argwhere(label==1)].mean())
    print('val FPR', (yhat!=label)[np.argwhere(label==0)].mean())

    # confidence score
    dec = clf.decision_function(X)
    #avgscore = (dec * (label*2 - 1)).mean()
    avgscore = np.abs(dec).mean()
    print('avg abs conf', avgscore, dec[yhat==0].mean(), dec[yhat==1].mean())
    return None

def load_ordreg(prefix: str, emax: int = 16, *, verbose=False) -> List:
    '''
    load trained svms and turn them into ordinal regression model
    '''
    # existence check
    for i in range(1, emax+1):
        if not os.path.exists(prefix + f'{i}.pkl'):
            raise FileNotFoundError(prefix + f'{i}.pkl')
    # list : h_1, h_2, ..., h_16
    hs = []
    for i in range(1, emax+1):
        fpath = prefix + f'{i}.pkl'
        with open(fpath, 'rb') as f:
            clf = pkl.load(f)
            assert(clf.n_support_[0] > 0)
        if verbose:
            print(f'[OrdReg] Loading {fpath} ...', clf.n_support_)
        hs.append(clf)
    #
    return hs

class OrdReg(object):
    '''
    assemble a list of hs: h_1, h_2, h_3, h_4 into an ord reg model

    We partly follow sklearn estimator API definition.
    https://scikit-learn.org/stable/developers/develop.html
    '''

    def __init__(self, hs: list, weight: np.ndarray):
        '''
        initialize with hs
        '''
        # we support dispatch based on type
        if isinstance(hs, list):
            # regard as a list of trained models
            self.hs = hs
        elif isinstance(hs, str):
            # regard as a prefix
            hs_ = load_ordreg(hs)
            self.hs = hs_
        elif hs is None:
            self.hs = []
        else:
            raise TypeError(hs)
        print(f'[OrdReg] initialized with {len(self.hs)} binary classifiers')
        self.weight = weight

    def fit(self, data, targets, numhs:int=4):
        hs = []
        for e in track(range(1, numhs+1), description='TRAIN OrdReg'):
            #[1] all
            # high FPR
            #clf = svm.SVC(kernel='linear', verbose=True)  # better.
            #clf = svm.SVC(kernel='rbf', verbose=False)  # better.
            #label = (targets >= e).astype(np.int)
            #clf.fit(data, label)

            #[2] select
            #clf = svm.LinearSVC(verbose=False)  # better.
            #mask = np.argwhere(np.logical_or(targets== 0, targets>= e)).reshape(-1)
            #print('SVM DATA SHAPE', data[mask].shape, targets[mask].shape)
            #clf.fit(data[mask], targets[mask]>0)

            #[3] all weighted: bad with any kernel. DR ~0.58,FPR 0.03
            #clf = svm.LinearSVC(verbose=False)
            #clf = svm.SVC(kernel='poly', verbose=False)
            #mask = np.argwhere(targets == 0).reshape(-1)
            #weight = np.ones(len(targets))
            #weight[mask] = 512
            #label = (targets >= e).astype(np.int)
            #clf.fit(data, label, sample_weight=weight)

            #[4] select weighted:
            # good with rbf weight 1 high FPR
            #               weight 128 0.616 0.007
            #               weight 16  0.718 0.048
            #clf = svm.LinearSVC(verbose=False)
            #clf = svm.SVC(kernel='rbf')
            #select = np.argwhere(np.logical_or(targets== 0, targets>= e)).reshape(-1)
            #benign = np.argwhere(targets == 0).reshape(-1)
            #weight = np.ones(len(targets))
            #weight[benign] = 16
            #clf.fit(data[select], targets[select]>0, sample_weight=weight[select])

            #[5] binary weighted
            # linear: no converge. weight 1: high FPR weight 128: 0.68 0.02
            # rbf: converges. weight 1 high FPR. weight 16: 0.64 0.009
            #                 weight 8 0.66 0.01 weight 512 0.48 0.0019
            #                 weight 1024: same as 512
            #clf = svm.LinearSVC(verbose=False)
            clf = svm.SVC(kernel='rbf')
            select = np.argwhere(np.logical_or(targets== 0, targets == e)).reshape(-1)
            benign = np.argwhere(targets == 0).reshape(-1)
            weight = np.ones(len(targets))
            weight[benign] = self.weight[e-1]
            clf.fit(data[select], targets[select]>0, sample_weight=weight[select])

            #[common]
            hs.append(clf)
        self.hs = hs

    def predict(self, data):
        preds = [h.predict(data) for h in self.hs]
        return np.vstack(preds).sum(axis=0).reshape(-1).astype(np.float32)

def detect(task: str, *, verbose=False):
    '''
    perform pre-defined detection
    '''
    if task.startswith('svm') or task.startswith('zsm'):

        if task.endswith('ct-ad'):
            assert('svm' in task)
            valdata, vallabel = load_data_series('data/val-ct')
            vlabel, vmlike, vllike = load_aux_series('data/val-ct')
            trndata, trnlabel = load_data_series('data/trn-ct', fmax=50)
            org = OrdReg(hs=None, weight=
                    np.array([2, 2, 8, 4])
                    )
        elif 'ct-' in task:
            assert('svm' not in task)
            task = '-'.join(task.split('-')[1:])
            valdata, vallabel = load_data_series(f'data/val-{task}')
            vlabel, vmlike, vllike = load_aux_series(f'data/val-{task}')
            with open('svm-ct-ad.cache', 'rb') as f:
                org = pkl.load(f)
            print(org)

        elif task.endswith('il-ad'):
            assert('svm' in task)
            valdata, vallabel = load_data_series('data/val-il')
            vlabel, vmlike, vllike = load_aux_series('data/val-il')
            trndata, trnlabel = load_data_series('data/trn-il', fmax=50)
            org = OrdReg(hs=None, weight=
                    np.array([2, 3.2, 8, 3])
                    )
        elif 'il-' in task:
            assert('svm' not in task)
            task = '-'.join(task.split('-')[1:])
            valdata, vallabel = load_data_series(f'data/val-{task}')
            vlabel, vmlike, vllike = load_aux_series(f'data/val-{task}')
            with open('svm-il-ad.cache', 'rb') as f:
                org = pkl.load(f)
            print(org)

        elif task.endswith('sw-ad'):
            assert('svm' in task)
            valdata, vallabel = load_data_series('data/val-sw')
            vlabel, vmlike, vllike = load_aux_series('data/val-sw')
            trndata, trnlabel = load_data_series('data/trn-sw', fmax=50)
            org = OrdReg(hs=None, weight=
                    np.array([8, 128, 128, 1])
                    )
        elif 'sw-' in task:
            assert('svm' not in task)
            task = '-'.join(task.split('-')[1:])
            valdata, vallabel = load_data_series(f'data/val-{task}')
            vlabel, vmlike, vllike = load_aux_series(f'data/val-{task}')
            with open('svm-sw-ad.cache', 'rb') as f:
                org = pkl.load(f)
            print(org)

        elif task.endswith('m4-ad'):
            assert('svm' in task)
            valdata, vallabel = load_data_series('data/val-m4')
            vlabel, vmlike, vllike = load_aux_series('data/val-m4')
            trndata, trnlabel = load_data_series('data/trn-m4')
            org = OrdReg(hs=None, weight=128)
        elif 'm4-' in task:
            assert('svm' not in task)
            task = '-'.join(task.split('-')[1:])
            valdata, vallabel = load_data_series(f'data/val-{task}')
            vlabel, vmlike, vllike = load_aux_series(f'data/val-{task}')
            with open('svm-m4-ad.cache', 'rb') as f:
                org = pkl.load(f)
            print(org)

        elif task.endswith('m8-ad'):
            assert('svm' in task)
            valdata, vallabel = load_data_series('data/val-m8')
            vlabel, vmlike, vllike = load_aux_series('data/val-m8')
            trndata, trnlabel = load_data_series('data/trn-m8')
            org = OrdReg(hs=None, weight=2)
        elif 'm8-' in task:
            assert('svm' not in task)
            task = '-'.join(task.split('-')[1:])
            valdata, vallabel = load_data_series(f'data/val-{task}')
            vlabel, vmlike, vllike = load_aux_series(f'data/val-{task}')
            with open('svm-m8-ad.cache', 'rb') as f:
                org = pkl.load(f)
            print(org)

        else:
            raise ValueErorr(task)
        if task.endswith('ad'):
            org.fit(trndata, trnlabel)
        else:
            pass

        plt.plot(np.arange(len(vallabel)), vallabel, 'ro')
        pred = org.predict(valdata)
        print('PRED', pred.shape)
        plt.scatter(np.arange(len(vallabel)), pred, c='b', marker='.')

        # predict on val set
        mask_benign = np.argwhere(vallabel == 0).flatten()
        mask_adversarial = np.argwhere(vallabel != 0).flatten()
        print('Model Benign Image Classification Accuracy',
                accuracy_score(vlabel[mask_benign], vmlike[mask_benign]))
        print('Ruthless AdvImage L-Like Guess Accuracy',
                accuracy_score(vlabel[mask_adversarial], vllike[mask_adversarial]))
        print('UAD Cls Acc.', (vlabel == vmlike).mean())
        print('UAD Cls Cor Acc.', (vlabel == np.where(pred>0, vllike, vmlike)).mean())
        print('UAD Det Acc.', ((vallabel>0)==(pred>0)).mean())
        print('UAD Prec.', precision_score(vallabel>0, pred>0))
        print('UAD AP', average_precision_score(vallabel>0, pred>0))
        print('UAD TPR', (((pred>0)==(vallabel>0))[np.argwhere(vallabel>0)]).mean())
        print('UAD FPR', ((pred>0)[np.argwhere(vallabel==0)]).mean(), len(np.argwhere(vallabel==0)))
        #dec = clf.decision_function(X)
        #avgscore = np.abs(dec).mean()
        #print('UAD Conf', avgscore, dec[yhat==0].mean(), dec[yhat==1].mean())
        print('UAD MAE', np.mean(np.abs(pred - vallabel)))
        print('UAD MAE', mean_absolute_error(vallabel, pred))
        c.print(classification_report(vallabel > 0, pred > 0))

        # draw table
        table = Table(title=task)
        table.add_column('Epsilon', justify='right')
        #table.add_column('Det-Acc. %', justify='right')
        table.add_column('TPR %', justify='right')
        table.add_column('FPR %', justify='right')
        #table.add_column('F1', justify='right')
        table.add_column('MAE', justify='right')
        table.add_column('Cls-Acc. %', justify='right')
        table.add_column('Cls-Acc* %', justify='right')
        for (e, hs) in track(enumerate(org.hs, 1), total=len(org.hs)):
            row = [str(e)]
            where = np.argwhere(np.logical_or(vallabel == 0,
                    vallabel == e)).reshape(-1)
            wheread = np.argwhere(vallabel == e).reshape(-1)
            #xval = valdata[where].view(-1, 2)
            yval = vallabel[where].reshape(-1) > 0
            #c.print(f'selecting hs {e}', xval.shape, yval.shape)
            #yhat = pred[where].reshape(-1) > 0  # no, no ORDREG pred
            yhat = org.hs[e-1].predict(valdata[where]).reshape(-1)
            yhatad = org.hs[e-1].predict(valdata[wheread]).reshape(-1)
            #
            acc = accuracy_score(yval, yhat)
            #row.append(f'{100*acc:.1f}')
            tpr = (yval == yhat)[np.argwhere(yval > 0)].mean()
            row.append(f'{100*tpr:.1f}')
            fpr = yhat[np.argwhere(yval == 0)].mean()
            row.append(f'{100*fpr:.1f}')
            f1 = f1_score(yval, yhat)
            #row.append(f'{f1:.2f}')
            mae = '-'
            row.append(mae)
            clsacc = (vlabel[where] == vmlike[where]).mean() * 100
            clsaccad = (vlabel[wheread] == vmlike[wheread]).mean() * 100
            row.append(f'{clsacc:.1f} ({clsaccad:.1f})')
            clsaccs = (vlabel[where] == np.where(yhat>0, vllike[where], vmlike[where])).mean() * 100
            clsaccsad = (vlabel[wheread] == np.where(yhatad>0, vllike[wheread], vmlike[wheread])).mean() * 100
            row.append(f'{clsaccs:.1f} ({clsaccsad:.1f})')
            table.add_row(*row)
        oradvcacc = '%.1f' % (100*accuracy_score(vlabel[mask_adversarial], vmlike[mask_adversarial]))
        oradvcaccs = '%.1f' % (100*accuracy_score(vlabel[mask_adversarial],
            np.where(pred[mask_adversarial]>0, vllike[mask_adversarial], vmlike[mask_adversarial])))
        table.add_row('UAD',
                #"%.1f" % (((vallabel>0)==(pred>0)).mean() * 100),
                "%.1f" % ((((pred>0)==(vallabel>0))[np.argwhere(vallabel>0)]).mean() * 100),
                "%.1f" % (((pred>0)[np.argwhere(vallabel==0)]).mean() * 100),
                #"%.2f" % f1_score(vallabel>0, pred>0),
                "%.2f" % mean_absolute_error(vallabel, pred),
                ("%.1f" % ((vlabel == vmlike).mean() * 100)) + f" ({oradvcacc})",
                ("%.1f" % ((vlabel == np.where(pred>0, vllike, vmlike)).mean() * 100)) + f" ({oradvcaccs})",
        )
        c.print(table)

        if task.startswith('svm'):
            save = f'{task}.cache'
            with open(save, 'wb') as f:
                pkl.dump(org, f)
                print('OrgReg model written to', save)
        if verbose:
            plt.show()

    if task.startswith('svr') or task.startswith('zsr'):
        if task.endswith('ct-ad'):
            trndata, trnlabel = load_data_series('data/trn-ct')
            valdata, vallabel = load_data_series('data/val-ct')
            weight_ = 64
        elif task.endswith('il-ad'):
            trndata, trnlabel = load_data_series('data/trn-il')
            valdata, vallabel = load_data_series('data/val-il')
            weight_ = 128
        elif task.endswith('sw-ad'):
            trndata, trnlabel = load_data_series('data/trn-sw')
            valdata, vallabel = load_data_series('data/val-sw')
            weight_ = 256
        else:
            raise ValueError(task)
        # [1] high FPR
        #reg = svm.SVR(verbose=True)
        #reg.fit(trndata, trnlabel)
        # [2] w 16 FPR 24
        #     w 128 0.69 0.03
        #     w 512 0.65 0.0097
        reg = svm.SVR(verbose=True)
        benign = np.argwhere(trnlabel == 0).reshape(-1)
        weight = np.ones(len(trnlabel))
        weight[benign] = weight_
        reg.fit(trndata, trnlabel, weight)
        # [3] w 16 0.87 0.49
        #reg = svm.LinearSVR(verbose=True)
        #benign = np.argwhere(trnlabel == 0).reshape(-1)
        #weight = np.ones(len(trnlabel))
        #weight[benign] = 512
        #reg.fit(trndata, trnlabel, weight)
        # [end]
        pred = reg.predict(valdata)
        xs = np.arange(len(vallabel))
        plt.plot(xs, vallabel, 'ro')
        plt.scatter(xs, pred, c='b', marker='.')

        print('SVR Acc.', ((vallabel>0) == (pred.round()>0)).mean())
        print('SVR TPR', ((pred.round()>0)==(vallabel>0))[
            np.argwhere(vallabel>0)].mean())
        print('SVR FPR', (pred.round()>0)[
            np.argwhere(vallabel==0)].mean(), len(np.argwhere(vallabel==0)))
        print('SVR MAE', mean_absolute_error(vallabel, pred))
        c.print(classification_report(vallabel > 0, pred.round() > 0))
        if verbose:
            plt.show()

    if task == 'ct-fgsm':
        # load ct-uad model
        fpath = 'ct-uad.org.cache'
        if not os.path.exists(fpath):
            raise Exception('please train ct-uad first')
        with open(fpath, 'rb') as f:
            org = pkl.load(f)
        print(org)
        # load ct-fgsm validation data
        # you may symlink data/ct-e0 to data/ct-fgsm-e0, as they equivalent
        valdata, vallabel = load_series_data('data/ct-fgsm')

        xs = np.arange(len(vallabel))
        plt.plot(xs, vallabel, 'ro')
        pred = org.predict(valdata)
        plt.scatter(xs, pred, c='b', marker='.')

        # metrics
        print('ZeroShot Acc.', accuracy_score(vallabel>0, pred>0))
        print('ZeroShot TPR', (((pred>0)==(vallabel>0))[np.argwhere(vallabel>0)]).mean())
        print('ZeroShot FPR', ((pred>0)[np.argwhere(vallabel==0)]).mean(),
                'numFalse', (vallabel==0).sum())
        print('ZeroShot F1', f1_score(vallabel>0, pred>0))
        print('zeroshot MAE', mean_absolute_error(vallabel, pred))
        c.print(classification_report(vallabel>0, pred>0))
        if verbose:
            plt.show()

