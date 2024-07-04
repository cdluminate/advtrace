#!/usr/bin/env python3
'''
Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
Source code will be released under Apache-2.0 license if paper accepted.
'''
import sys, os, yaml
import numpy as np
import torch as th, torch.utils.data
import argparse, collections
import lib
from termcolor import cprint, colored


def Train(argv):
    '''
    Train the Neural Network
    '''
    ag = argparse.ArgumentParser()
    ag.add_argument('-D', '--device',
            default='cuda' if th.cuda.is_available() else 'cpu',
            type=str, help='computational device')
    ag.add_argument('-M', '--model', type=str, required=True)
    ag.add_argument('--overfit', action='store_true')
    ag.add_argument('--report', type=int, default=10)
    ag.add_argument('--validate', action='store_true')
    ag.add_argument('-X', '--dataparallel', action='store_true')
    ag = ag.parse_args(argv)

    print('>>> Parsing arguments')
    for x in yaml.dump(vars(ag)).split('\n'): print(colored(x, 'green'))
    config = yaml.load(open('config.yml', 'r').read(), Loader=yaml.SafeLoader)

    if ag.validate:
        sdpath = 'trained/' + ag.model + '.sdth'
        print('>>> Loading model from', sdpath)
        model = getattr(lib, ag.model).Model().to(ag.device)
        model.load_state_dict(th.load(sdpath))
        print(model)
        print('>>> Loading datasets')
        loader_test = model.getloader('test', config[ag.model]['batchsize'])
        print(len(loader_test.dataset))
        print(colored('Validate '+str(model.validate(loader_test)), 'white', 'on_magenta'))
        exit(0)

    print('>>> Setting up model and optimizer')
    if ag.dataparallel:
        model = getattr(lib, ag.model).Model(dataparallel=ag.dataparallel).to(ag.device)
    else:
        model = getattr(lib, ag.model).Model().to(ag.device)
    if config[ag.model].get('momentum', -1.0) > 0.0:
        optim = th.optim.SGD(model.parameters(),
                lr=config[ag.model]['lr'],
                weight_decay=float(config[ag.model]['weight_decay']) if config[ag.model].get('weight_decay', None) is not None else 1e-7,
                momentum=config[ag.model]['momentum'])
    else:
        optim = th.optim.Adam(model.parameters(),
                lr=config[ag.model]['lr'],
                weight_decay=float(config[ag.model]['weight_decay']) if config[ag.model].get('weight_decay', None) is not None else 1e-7,
                )
    print(model); print(optim)

    print('>>> Loading datasets')
    loader_train = model.getloader('train', config[ag.model]['batchsize'])
    loader_test  = model.getloader('test', config[ag.model]['batchsize'])
    print(len(loader_train.dataset), len(loader_test.dataset))

    print('>>> Start training')
    print(colored('Validate[-1] '+str(model.validate(loader_test)), 'white', 'on_magenta'))
    for epoch in range(config[ag.model]['epoch']):

        # dynamic learning rate (step policy)
        lrstep = list(config[ag.model].get('lrstep', []))
        lrfactor = float(config[ag.model].get('lrfactor', 0.1))
        for (pw, ep) in reversed(list(enumerate(lrstep, 1))):
            if epoch >= ep:
                lrn = config[ag.model]['lr']
                lrn = lrn * (lrfactor ** pw)
                for param_group in optim.param_groups:
                    param_group['lr'] = lrn
                break

        # Do the normal training process
        for iteration, (images, labels) in enumerate(loader_train):
            model.train()
            output, loss = model.loss(images, labels) # NOTE: don't specify adv=*. here.
            optim.zero_grad()
            loss.backward()
            optim.step()

            if (iteration % ag.report == 0) or ag.overfit:
                model.report(epoch, iteration, len(loader_train),
                        output, labels, loss)
            if ag.overfit:
                break

        # save a snapshot
        cprint(f'Validate[{epoch}] '+str(model.validate(loader_test)), 'white', 'on_magenta')
        if not os.path.exists('trained/'):
            os.mkdir('trained')
        th.save(model.state_dict(), 'trained/'+ag.model+'+snapshot.sdth')

    print('>>> Saving the network to:', 'trained/' + ag.model + '.sdth')
    th.save(model.cpu().state_dict(), 'trained/' + ag.model + '.sdth')


if __name__ == '__main__':
    Train(sys.argv[1:])
