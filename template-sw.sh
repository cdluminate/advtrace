# Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
# Source code will be released under Apache-2.0 license if paper accepted.
source bin/create-data.bash
create_$1 il_swin data/val-sw-$1 $2
