'''
Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
Source code will be released under Apache-2.0 license if paper accepted.
'''
import torch as th
import torchvision as V
from collections import OrderedDict

# https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth
sd = th.load('trained/swin_base_patch4_window7_224.pth')['model']
print(sd)
newsd = OrderedDict()

for (key, value) in sd.items():
    newsd['net.' + key] = value

th.save(newsd, 'trained/il_swin.sdth')
print('swin-transformer conversion complete')
print(newsd.keys())
