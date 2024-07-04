'''
Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
Source code will be released under Apache-2.0 license if paper accepted.
'''
import torch as th
import torchvision as V
from collections import OrderedDict

# convert torchvision resnet152 state dictionary

model = V.models.resnet152(True)
#print(model)

sd = model.cpu().state_dict()
print(type(sd))
newsd = OrderedDict()

for (key, value) in sd.items():
    newsd['net.' + key] = value

th.save(newsd, 'trained/il_r152.sdth')
print('resnet152 conversion complete')
