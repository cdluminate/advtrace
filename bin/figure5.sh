# Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
# Source code will be released under Apache-2.0 license if paper accepted.
python3 bin/util.py qx-arcm -p data/val-i5 --save val-i5.pdf
pdfcrop val-i5.pdf val-i5.pdf

python3 bin/util.py qx-arcm -p data/val-m4 --save val-m4.pdf
pdfcrop val-m4.pdf val-m4.pdf

python3 bin/util.py qx-arcm -p data/val-m8 --save val-m8.pdf
pdfcrop val-m8.pdf val-m8.pdf
