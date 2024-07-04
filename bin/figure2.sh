# Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
# Source code will be released under Apache-2.0 license if paper accepted.
python3 bin/util.py q-arcm -p data/val-ct --save val-ct-ex.svg
inkscape -o val-ct-ex.pdf val-ct-ex.svg
pdfcrop val-ct-ex.pdf val-ct-ex.pdf
python3 bin/util.py q-arcm -p data/val-il --save val-il-ex.svg
inkscape -o val-il-ex.pdf val-il-ex.svg
pdfcrop val-il-ex.pdf val-il-ex.pdf
python3 bin/util.py q-arcm -p data/val-sw --save val-sw-ex.svg
inkscape -o val-sw-ex.pdf val-sw-ex.svg
pdfcrop val-sw-ex.pdf val-sw-ex.pdf
