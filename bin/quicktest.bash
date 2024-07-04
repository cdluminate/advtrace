#!/bin/bash
# Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
# Source code will be released under Apache-2.0 license if paper accepted.
set -e
set -x

export TAG=BN
export DIR=testbn
if ! test -d ${DIR}; then
python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 0
fi

export TAG=AD
export DIR=testad
if ! test -d ${DIR}; then
python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 8
fi

python3 bin/util.py p-arcm -b testbn -a testad
