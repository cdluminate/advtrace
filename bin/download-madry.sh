#!/bin/bash
# Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
# Source code will be released under Apache-2.0 license if paper accepted.
set -e

# https://github.com/MadryLab/robustness

echo download madry m4 resnet50
wget -c 'https://www.dropbox.com/s/axfuary2w1cnyrg/imagenet_linf_4.pt?dl=0' -O trained/imagenet_linf_4.pt

echo download madry m8 resnet50
wget -c 'https://www.dropbox.com/s/yxn15a9zklz3s8q/imagenet_linf_8.pt?dl=0' -O trained/imagenet_linf_8.pt

echo done
