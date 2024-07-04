AdvTrace: On Trace of PGD-Like Adversarial Attacks -- python code usage
===

## 1. Prepration: Dataset and pretrained models

### 1.1 Dataset

1. CIFAR-10: download cifar 10 (python verison) and put it to `~/.torch/`.
Details can be found in `lib/cifar10.py`. Namely, there should be a
`cifar-10-batches-py` directory under `~/.torch/`.

2. ImageNet. Download the kaggle version of image net. And put it to ~/.torch/`.
Details can be found in `lib/ilsvrc.py`.

### 1.2 Pretrained models

For cifar10, we train by ourselves.
```shell
python3 Train.py -M ct_res18
```

For ResNet152, please download pytorch version using `bin/convert-r152.py`.
For SwinTransformer, download the model using the URL in `bin/convert-swint.py`
to `trained/`. Then run the script with `python3 bin/convert-swint.py`.

## 2. Feature extraction

First create directory `data` under this source tree.

Then let's create the first batch of BIM examples, with k=0,1,2,3,4

```shell
export MODEL=ct_res18
DIR=data/val-ct-e0  TAG=BN python3 Attack.py -M ${model} -A ARC -v -e 0
DIR=data/val-ct-e2  TAG=BN python3 Attack.py -M ${model} -A ARC -v -e 2
DIR=data/val-ct-e4  TAG=BN python3 Attack.py -M ${model} -A ARC -v -e 4
DIR=data/val-ct-e8  TAG=BN python3 Attack.py -M ${model} -A ARC -v -e 8
DIR=data/val-ct-e16 TAG=BN python3 Attack.py -M ${model} -A ARC -v -e 16
```

Note, you may need to remove or edit the `if idx >=...: exit()` part if you
want to go throuh more data. Remove it to traverse the whole validation dataset.
The values in the code are merely for demonstration purpose -- it will only
go through several data batches for quick and brief experiment.

If you want to create data on the training set, you need to modify the code
following `bin/create_data.sh`. And put training data to `data/trn-ct-e{0,2,4,8,16}`.

For ResNet152, replace `ct` (c_ifar t_en) into `il` (i_l_svrc).
For SwinT-B-IN1K, replace `ct` with `sw`.

Now we want to create data for e.g. MIM attack. For all attacks other than BIM,
we only have to go through the validation/test dataset and omit the training set.
```
export MODEL=ct_res18
cd data; ln -s val-ct-e0 val-ct-mim-e0  # reuse e=0 data. Identical even if we run again.
DIR=data/val-ct-mim-e2  TAG=MIM python3 Attack.py -M ${model} -A ARC -v -e 2
DIR=data/val-ct-mim-e4  TAG=MIM python3 Attack.py -M ${model} -A ARC -v -e 4
DIR=data/val-ct-mim-e8  TAG=MIM python3 Attack.py -M ${model} -A ARC -v -e 8
DIR=data/val-ct-mim-e16 TAG=MIM python3 Attack.py -M ${model} -A ARC -v -e 16
```

You can also do this using `bin/create-data.sh`. See `tmux-template.sh` for example
in how to invoke that script. In some scripts the old name `UT:PGDT` of ARC will
be used instead of the new name `ARC`.

Use `data/stat.py` to see how many feature data we have created.

## 3. Prediction

When we have collected training data `data/trn-ct-e{...}` and validation data
`data/val-ct-e{...}`. We first train and evaluate for BIM attack
```shell
python3 bin/util.py svm-ct-ad
```

Then we evaluate for MIM attack, for instance
```
python3 bin/util.py zsm-ct-mim
```
The program will automatically grab the training data and report performance
on the validation data.

The reference for full evaluation can be found in `bin/train-test-svm.sh`
(PGD-like attacks), and `bin/test-svm-nonpgd.sh` for non-pgd-like attacks.

## A. File Tree explained

Important files are annotated. Files without annotation are rather auxiliary
and may be not really useful.


```
.
├── arc                        ARC feature
│   ├── arcfeat.py             ARCm and ARCv calculation
│   ├── arcio.py               read/write ARC feature files
│   ├── detect.py              train and evaluate SVMs
│   ├── __init__.py
│   ├── pgdt.py                exploitation vector calculation
│   ├── plots.py               visualize ARC feature
│   └── utils.py
├── bin
│   ├── convert-madry.py
│   ├── convert-r152.py        download pytorch resnet152 model and convert
│   ├── convert-r50.py
│   ├── convert-swint.py       download swin transformer and convert
│   ├── cos-figs.bash
│   ├── create-data.bash       helper script for extracting features
│   ├── download-madry.sh
│   ├── figure2.sh
│   ├── figure3.sh
│   ├── figure4.sh
│   ├── figure5.sh
│   ├── figure6.sh
│   ├── informed.bash
│   ├── nss-data.sh            extract features using NSS
│   ├── nss-detect.py          train SVMs using NSS
│   ├── nss-vis.py             visualize NSS feature
│   ├── pgdtl8.sh
│   ├── quicktest.bash
│   ├── roc-ct.py
│   ├── roc-il.py
│   ├── roc-sw.py
│   ├── test-svm-nonpgd.sh
│   ├── train-test-svm.sh
│   └── util.py                train SVM using ARC or visualize ARC
├── config.yml
├── data
│   └── stat.py
├── lib
│   ├── attacks.py             Adversarial attacks
│   ├── base.py
│   ├── cifar10.py
│   ├── classify.py            Main evaluation loop that traverses dataset
│   ├── ct_res18d.py
│   ├── ct_res18.py            ResNet18 model
│   ├── fa_c2f2d.py
│   ├── fa_c2f2.py
│   ├── fa_mlpd.py
│   ├── fa_mlp.py
│   ├── fashion.py
│   ├── il_madry4.py
│   ├── il_madry8.py
│   ├── il_r152.py             ResNet152 model
│   ├── il_r50.py
│   ├── ilsvrc.py              ILSVRC dataset abstraction
│   ├── il_swin.py             SwinTransformer model
│   ├── __init__.py
│   ├── nss                    The code of NSS from the author. (author did not specify license)
│   │   ├── __init__.py
│   │   └── README.md          The URL of the NSS source code.
│   ├── swint                  SwinTransformer official code copy
│   │   └── LICENSE            Their code is MIT-licensed.
│   ├── transfer.py            transferrability attack
│   └── utils.py
├── Makefile
├── README.md                  THIS DOCUMENTATION
├── requirements.txt           python dependency
├── template-ct.sh
├── template-il.sh
├── template-sw.sh
├── tmux-template.sh           tmux script as example bin/create-data.sh usage
├── trained
│   └── ct_res18.sdth          trained ResNet18 for CIFAR10 is put here
├── Attack.py                  start adversarial attack
└── Train.py                   train a model (only for cifar10)
```

