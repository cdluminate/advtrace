'''
Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
Source code will be released under Apache-2.0 license if paper accepted.
'''
import os
import torch as th
import torchvision as V
from collections import OrderedDict

# convert torchvision resnet50 state dictionary

if not os.path.exists('trained/imagenet_linf_4.pt'):
    raise FileNotFoundError('run bin/download-madry.sh first')


def convert_madry4():
    model = V.models.resnet50(pretrained=False)
    # load madry state dictionary
    sd = th.load('trained/imagenet_linf_4.pt')['model']
    print(type(sd))
    # convert to torchvision standard state dictionary and test load
    stdsd = OrderedDict()
    for (key, value) in sd.items():
        if 'module.attacker.model' not in key:
            continue
        else:
            stdsd[key.replace('module.attacker.model.', '')] = value
    model.load_state_dict(stdsd)
    # convert to our state dictionary
    sd = model.cpu().state_dict()
    newsd = OrderedDict()
    for (key, value) in sd.items():
        newsd['net.' + key] = value
    th.save(newsd, 'trained/il_madry4.sdth')
    print('madry4 (resnet50) conversion complete')

def convert_madry8():
    model = V.models.resnet50(pretrained=False)
    # load madry state dictionary
    sd = th.load('trained/imagenet_linf_8.pt')['model']
    print(type(sd))
    # convert to torchvision standard state dictionary and test load
    stdsd = OrderedDict()
    for (key, value) in sd.items():
        if 'module.attacker.model' not in key:
            continue
        else:
            stdsd[key.replace('module.attacker.model.', '')] = value
    model.load_state_dict(stdsd)
    # convert to our state dictionary
    sd = model.cpu().state_dict()
    newsd = OrderedDict()
    for (key, value) in sd.items():
        newsd['net.' + key] = value
    th.save(newsd, 'trained/il_madry8.sdth')
    print('madry8 (resnet50) conversion complete')


if __name__ == '__main__':
    convert_madry4()
    convert_madry8()
