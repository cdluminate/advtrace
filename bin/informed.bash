#!/bin/bash
# Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
# Source code will be released under Apache-2.0 license if paper accepted.
set -e
set -x

trainval_ct () {
echo ==========================================================================
echo === Training =============================================================
echo ==========================================================================
for e in 2 4 8 16; do
    if test -e ct-h${e}.pkl; then
        true #continue
    fi
    python3 bin/util.py train --bndir data/trn-ct-e0 --addir data/trn-ct-e${e} --save ct-h${e}.pkl
done
echo ==========================================================================
echo === Validating ===========================================================
echo ==========================================================================
for e in 2 4 8 16; do
    python3 bin/util.py val --bndir data/val-ct-e0 --addir data/val-ct-e${e} --load ct-h${e}.pkl
done
}
trainall_ct () {
for e in $(seq 1 16); do
    python3 bin/util.py train --bndir data/trn-ct-e0 --addir data/trn-ct-e${e} --save ct-h${e}.pkl
    python3 bin/util.py val   --bndir data/ct-e0 --addir data/ct-e${e} --load ct-h${e}.pkl
done
}

trainval_il () {
echo ==========================================================================
echo === Training =============================================================
echo ==========================================================================
for e in 2 4 8 16; do
    if test -e il-h${e}.pkl; then
        true #continue
    fi
    python3 bin/util.py train --bndir il-e0 --addir il-e${e} --save il-h${e}.pkl
done
echo ==========================================================================
echo === Validating ===========================================================
echo ==========================================================================
for e in 2 4 8 16; do
    python3 bin/util.py val --bndir val-il-e0 --addir val-il-e${e} --load il-h${e}.pkl
done
}

trainval_sw () {
echo ==========================================================================
echo === Training =============================================================
echo ==========================================================================
for e in 2 4 8 16; do
    if test -e sw-h${e}.pkl; then
        true #continue
    fi
    python3 bin/util.py train --bndir sw-e0 --addir sw-e${e} --save sw-h${e}.pkl
done
echo ==========================================================================
echo === Validating ===========================================================
echo ==========================================================================
for e in 2 4 8 16; do
    python3 bin/util.py val --bndir val-sw-e0 --addir val-sw-e${e} --load sw-h${e}.pkl
done
}

#trainval_ct
#trainval_il
#trainval_sw

trainall_ct
