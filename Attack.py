#!/usr/bin/env python3
'''
Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
Source code will be released under Apache-2.0 license if paper accepted.
'''
import os
import sys
import yaml
import re
import numpy as np
import argparse
import torch as th
from termcolor import cprint
#
import lib


def Attack(argv):
    '''
    Attack a pre-trained model
    '''
    ag = argparse.ArgumentParser()
    ag.add_argument('-D', '--device', type=str,
            default='cuda' if th.cuda.is_available() else 'cpu')
    ag.add_argument('-A', '--attack', type=str, required=True,
            choices=[
                # untargeted attacks
                'UT:FGSM', 'UT:PGD', 'UT:UPGD', 
                'UT:CW', 'UT:PGDl8', 'UT:PGDl2', 'UT:NES', 'UT:SPSA',
                'UT:GA', 'UT:UN', 'UT:MIM',
                # ARC Feature
                'UT:PGDT', 'ARC',
                # NSS feature
                'NSS',
                ])
    ag.add_argument('-e', '--epsilon', default=8,
            type=float, help='hyper-param epsilon, will be divided by 255.')
    ag.add_argument('-M', '--model', type=str, required=True)
    ag.add_argument('-v', '--verbose', action='store_true', help='verbose?')
    ag.add_argument('--maxiter', type=int, default=100)
    ag.add_argument('--vv', action='store_true', help='more verbose')
    ag.add_argument('-X', '--dataparallel', action='store_true')
    ag = ag.parse_args(argv)

    ## mangle parameters
    ag.epsilon = ag.epsilon/255.
    print('>>> Processing epsilon', ag.epsilon, '<-', ag.epsilon * 255.)

    print('>>> Parsing arguments and configuration file')
    for x in yaml.dump(vars(ag)).split('\n'): cprint(x, 'green')
    if ag.vv: ag.verbose = True
    config = yaml.load(open('config.yml', 'r').read(), Loader=yaml.SafeLoader)
    cprint(f'Attacking method is {ag.attack} \u03b5={ag.epsilon}', 'white',
            'on_magenta')

    # Load the white-box attacking target model
    if re.match('\S+:\S+', ag.model):
        Mname, Mpath = re.match('(\S+):(\S+)', ag.model).groups()
    else:
        Mname, Mpath = ag.model, 'trained/' + ag.model + '.sdth'
    print(f'>>> Loading white-box target {Mname} model from:', Mpath)
    if ag.dataparallel:
        model = getattr(lib, Mname).Model(dataparallel=ag.dataparallel).to(ag.device)
    else:
        model = getattr(lib, Mname).Model().to(ag.device)
    model.load_state_dict(th.load(Mpath))

    print('>>> Loading dataset ...', end=' ')
    arc_data_split = os.getenv('ARC_DATA_SPLIT', 'test')
    assert arc_data_split in ('train', 'test'), "USE_SPLIT must be either train or test"
    if not ag.vv:
        loader_test = \
            model.getloader(arc_data_split, config[Mname]['batchsize_atk'])
    elif ag.vv:
        loader_test = model.getloader(arc_data_split, 1)
        print('| overriden batchsize to 1', end=' ')
    print('| Testing dataset size =', len(loader_test.dataset))

    dconf = {'epsilon': ag.epsilon, 'maxiter': ag.maxiter, }
    print('>>> Start Attacking ...')
    model.attack(ag.attack, loader_test, dconf=dconf, verbose=ag.verbose)


if __name__ == '__main__':
    Attack(sys.argv[1:])
