#!/bin/bash
# Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
# Source code will be released under Apache-2.0 license if paper accepted.
set -e
set -x

export TAG=BN
export DIR=data/pgdtl8-e0
if ! test -d ${DIR}; then
ARC_TRAJ=pgdtl8 python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 0
fi

export TAG=AD
export DIR=data/pgdtl8-e2
if ! test -d ${DIR}; then
ARC_TRAJ=pgdtl8 python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 2
fi

export TAG=AD
export DIR=data/pgdtl8-e4
if ! test -d ${DIR}; then
ARC_TRAJ=pgdtl8 python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 4
fi

export TAG=AD
export DIR=data/pgdtl8-e8
if ! test -d ${DIR}; then
ARC_TRAJ=pgdtl8 python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 8
fi

export TAG=AD
export DIR=data/pgdtl8-e16
if ! test -d ${DIR}; then
ARC_TRAJ=pgdtl8 python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 16
fi

python3 bin/util.py q-arcm -p data/pgdtl8
