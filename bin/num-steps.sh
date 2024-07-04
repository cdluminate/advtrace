#!/bin/bash
set -x
# s100
#│       4 │  79.2 │   1.1 │    - │  43.6 (0.0) │ 74.3 (62.4) │
DIR=val-ct-e16-s50 TAG=AD python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 16 --maxiter=50
# s50
#│       4 │  75.0 │   1.1 │    - │  43.6 (0.0) │ 72.1 (58.1) │
DIR=val-ct-e16-s25 TAG=AD python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 16 --maxiter=25
# s25
#│       4 │  64.1 │   1.1 │    - │  43.6 (0.0) │ 66.7 (47.3) │
DIR=val-ct-e16-s15 TAG=AD python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 16 --maxiter=15
# s15
#│       4 │  49.3 │   1.1 │    - │  43.6 (0.0) │ 59.9 (33.5) │
DIR=val-ct-e16-s10 TAG=AD python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 16 --maxiter=10
# s10
#│       4 │  33.1 │   1.1 │    - │  43.7 (0.2) │ 53.1 (20.1) │
DIR=val-ct-e16-s08 TAG=AD python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 16 --maxiter=8
# s08
#│       4 │  22.4 │   1.1 │    - │  44.0 (0.7) │ 49.2 (12.2) │
DIR=val-ct-e16-s05 TAG=AD python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 16 --maxiter=5
# s05
#│       4 │   7.1 │   1.1 │    - │  45.5 (3.7) │  45.9 (5.5) │
