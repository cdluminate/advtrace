# Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
# Source code will be released under Apache-2.0 license if paper accepted.
python3 bin/util.py qx-arcm -p data/val-ct-pgdl8 --save ct-pgdl8.svg
python3 bin/util.py qx-arcm -p data/val-il-pgdl8 --save il-pgdl8.svg
python3 bin/util.py qx-arcm -p data/val-sw-pgdl8 --save sw-pgdl8.svg

python3 bin/util.py qx-arcm -p data/val-ct-mim --save ct-mim.svg
python3 bin/util.py qx-arcm -p data/val-il-mim --save il-mim.svg
python3 bin/util.py qx-arcm -p data/val-sw-mim --save sw-mim.svg

python3 bin/util.py qx-arcm -p data/val-ct-apgd --save ct-apgd.svg
python3 bin/util.py qx-arcm -p data/val-il-apgd --save il-apgd.svg
python3 bin/util.py qx-arcm -p data/val-sw-apgd --save sw-apgd.svg
