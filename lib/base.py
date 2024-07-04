'''
Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
Source code will be released under Apache-2.0 license if paper accepted.
'''
from scipy import stats
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
import traceback
import json
from .utils import IMmean, IMstd, renorm, denorm, xdnorm

#######################################################################

class Model(th.nn.Module):
    '''
    Abstract model. Similar to the pytorch-lightning API style.
    '''
    def forward(self, x):
        '''
        Purely input -> output, no loss function
        '''
        raise NotImplementedError

    def loss(self, x, y, device='cpu', *, adv: bool=False):
        '''
        Combination: input -> output -> loss
        Boolean variable adv toggles adversarial training

        Adversarial training: replace normal example with adv example
        https://github.com/MadryLab/mnist_challenge/blob/master/train.py
        '''
        raise NotImplementedError

    def report(self, epoch, iteration, total, output, labels, loss):
        '''
        Given the (output, loss) combination, report current stat
        '''
        raise NotImplementedError

    def validate(self, dataloader, device='cpu'):
        '''
        Run validation on the given dataset
        '''
        raise NotImplementedError

    def getloader(self, kind:str='train', batchsize:int=1):
        '''
        Load the specific dataset for the model
        '''
        raise NotImplementedError
