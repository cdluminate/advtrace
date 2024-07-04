# Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
# Source code will be released under Apache-2.0 license if paper accepted.
mkdir -p logs
# [cifar10 / res18]
# train and test
python3 bin/util.py svm-ct-ad
# test
python3 bin/util.py zsm-ct-biml2  | tee logs/zsm-ct-biml2.log
python3 bin/util.py zsm-ct-dlr    | tee logs/zsm-ct-dlr.log
python3 bin/util.py zsm-ct-fgsm   | tee logs/zsm-ct-fgsm.log
python3 bin/util.py zsm-ct-nes    | tee logs/zsm-ct-nes.log
python3 bin/util.py zsm-ct-spsa   | tee logs/zsm-ct-spsa.log
python3 bin/util.py zsm-ct-cw     | tee logs/zsm-ct-cw.log
python3 bin/util.py zsm-ct-fab    | tee logs/zsm-ct-fab.log
python3 bin/util.py zsm-ct-fmn    | tee logs/zsm-ct-fmn.log
python3 bin/util.py zsm-ct-ga     | tee logs/zsm-ct-ga.log
python3 bin/util.py zsm-ct-un     | tee logs/zsm-ct-un.log
python3 bin/util.py zsm-ct-square | tee logs/zsm-ct-square.log
python3 bin/util.py zsm-ct-difgsm | tee logs/zsm-ct-difgsm.log
python3 bin/util.py zsm-ct-tifgsm | tee logs/zsm-ct-tifgsm.log

# [imagenet/ res152]
# train and test
python3 bin/util.py svm-il-ad
# test
python3 bin/util.py zsm-il-biml2  | tee logs/zsm-il-biml2.log
python3 bin/util.py zsm-il-dlr    | tee logs/zsm-il-dlr.log
python3 bin/util.py zsm-il-fgsm   | tee logs/zsm-il-fgsm.log
python3 bin/util.py zsm-il-nes    | tee logs/zsm-il-nes.log
python3 bin/util.py zsm-il-spsa   | tee logs/zsm-il-spsa.log
python3 bin/util.py zsm-il-cw     | tee logs/zsm-il-cw.log
python3 bin/util.py zsm-il-fab    | tee logs/zsm-il-fab.log
python3 bin/util.py zsm-il-fmn    | tee logs/zsm-il-fmn.log
python3 bin/util.py zsm-il-ga     | tee logs/zsm-il-ga.log
python3 bin/util.py zsm-il-un     | tee logs/zsm-il-un.log
python3 bin/util.py zsm-il-square | tee logs/zsm-il-square.log
python3 bin/util.py zsm-il-difgsm | tee logs/zsm-il-difgsm.log
python3 bin/util.py zsm-il-tifgsm | tee logs/zsm-il-tifgsm.log

# [imagenet / swin]
# train and test
python3 bin/util.py svm-sw-ad
# test
python3 bin/util.py zsm-sw-biml2  | tee logs/zsm-sw-biml2.log
python3 bin/util.py zsm-sw-dlr    | tee logs/zsm-sw-dlr.log
python3 bin/util.py zsm-sw-fgsm   | tee logs/zsm-sw-fgsm.log
python3 bin/util.py zsm-sw-nes    | tee logs/zsm-sw-nes.log
python3 bin/util.py zsm-sw-spsa   | tee logs/zsm-sw-spsa.log
python3 bin/util.py zsm-sw-cw     | tee logs/zsm-sw-cw.log
python3 bin/util.py zsm-sw-fab    | tee logs/zsm-sw-fab.log
python3 bin/util.py zsm-sw-fmn    | tee logs/zsm-sw-fmn.log
python3 bin/util.py zsm-sw-ga     | tee logs/zsm-sw-ga.log
python3 bin/util.py zsm-sw-un     | tee logs/zsm-sw-un.log
python3 bin/util.py zsm-sw-square | tee logs/zsm-sw-square.log
python3 bin/util.py zsm-sw-difgsm | tee logs/zsm-sw-difgsm.log
python3 bin/util.py zsm-sw-tifgsm | tee logs/zsm-sw-tifgsm.log
