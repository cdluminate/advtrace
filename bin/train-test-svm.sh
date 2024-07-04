# Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
# Source code will be released under Apache-2.0 license if paper accepted.
mkdir -p logs
# [cifar10 / res18]
# train and test
python3 bin/util.py svm-ct-ad    | tee logs/svm-ct-ad.log
# test
python3 bin/util.py zsm-ct-pgdl8 | tee logs/zsm-ct-pgdl8.log
python3 bin/util.py zsm-ct-mim   | tee logs/zsm-ct-mim.log
python3 bin/util.py zsm-ct-apgd  | tee logs/zsm-ct-apgd.log
python3 bin/util.py zsm-ct-aa    | tee logs/zsm-ct-aa.log

# [imagenet/ res152]
# train and test
python3 bin/util.py svm-il-ad    | tee logs/svm-il-ad.log
# test
python3 bin/util.py zsm-il-pgdl8 | tee logs/zsm-il-pgdl8.log
python3 bin/util.py zsm-il-mim   | tee logs/zsm-il-mim.log
python3 bin/util.py zsm-il-apgd  | tee logs/zsm-il-apgd.log
python3 bin/util.py zsm-il-aa    | tee logs/zsm-il-aa.log

# [imagenet / swin]
# train and test
python3 bin/util.py svm-sw-ad    | tee logs/svm-sw-ad.log
# test
python3 bin/util.py zsm-sw-pgdl8 | tee logs/zsm-sw-pgdl8.log
python3 bin/util.py zsm-sw-mim   | tee logs/zsm-sw-mim.log
python3 bin/util.py zsm-sw-apgd  | tee logs/zsm-sw-apgd.log
python3 bin/util.py zsm-sw-aa    | tee logs/zsm-sw-aa.log
