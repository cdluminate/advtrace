# Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
# Source code will be released under Apache-2.0 license if paper accepted.
mkdir -p datanss

# part 1: following instructions in create-data.bash
DIR=datanss/trn-ct-e0  TAG=BN python3 Attack.py -M ct_res18 -A NSS -v -e 0
DIR=datanss/trn-ct-e2  TAG=AD python3 Attack.py -M ct_res18 -A NSS -v -e 2
DIR=datanss/trn-ct-e4  TAG=AD python3 Attack.py -M ct_res18 -A NSS -v -e 4
DIR=datanss/trn-ct-e8  TAG=AD python3 Attack.py -M ct_res18 -A NSS -v -e 8
DIR=datanss/trn-ct-e16 TAG=AD python3 Attack.py -M ct_res18 -A NSS -v -e 16

# part 2: clean git repo, remove idx check from lib/classfy.py
DIR=datanss/val-ct-e0  TAG=BN python3 Attack.py -M ct_res18 -A NSS -v -e 0
DIR=datanss/val-ct-e2  TAG=AD python3 Attack.py -M ct_res18 -A NSS -v -e 2
DIR=datanss/val-ct-e4  TAG=AD python3 Attack.py -M ct_res18 -A NSS -v -e 4
DIR=datanss/val-ct-e8  TAG=AD python3 Attack.py -M ct_res18 -A NSS -v -e 8
DIR=datanss/val-ct-e16 TAG=AD python3 Attack.py -M ct_res18 -A NSS -v -e 16
