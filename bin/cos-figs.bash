#!/bin/bash
# Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
# Source code will be released under Apache-2.0 license if paper accepted.
set -e
set -x

# 1. first edit lib/classify.py, and change maxiter_traj to 48
#    then edit the for loop in bin/create_data.bash and invoke the
#    following shell in that script
#        create_bn ct_res18 data/cos-ct 0
#    as a result, data will be populated in data/cos-ct-e0/.
#    The data is not used anywhere else and is only for demonstration.

# 2. then we create plot from the data
export TAG=BN
export DIR=data/cos-ct-e16
export SAVE=cos-ct-res18.svg
if ! test -e ${SAVE}; then
	python3 bin/util.py cos -b ${DIR} -s ${SAVE}
fi
