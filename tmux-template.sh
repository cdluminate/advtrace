# Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
# Source code will be released under Apache-2.0 license if paper accepted.
tmux new-window -n 'sw-02' 'CUDA_VISIBLE_DEVICES=8 bash template-sw.sh mim  2; sh'
tmux new-window -n 'sw-04' 'CUDA_VISIBLE_DEVICES=9 bash template-sw.sh mim  4; sh'
tmux new-window -n 'sw-08' 'CUDA_VISIBLE_DEVICES=6 bash template-sw.sh mim  8; sh'
tmux new-window -n 'sw-16' 'CUDA_VISIBLE_DEVICES=7 bash template-sw.sh mim 16; sh'

tmux new-window -n 'il-02' 'CUDA_VISIBLE_DEVICES=8 bash template-il.sh mim  2; sh'
tmux new-window -n 'il-04' 'CUDA_VISIBLE_DEVICES=9 bash template-il.sh mim  4; sh'
tmux new-window -n 'il-08' 'CUDA_VISIBLE_DEVICES=6 bash template-il.sh mim  8; sh'
tmux new-window -n 'il-16' 'CUDA_VISIBLE_DEVICES=7 bash template-il.sh mim 16; sh'

tmux new-window -n 'ct-02' 'CUDA_VISIBLE_DEVICES=8 bash template-ct.sh mim  2; sh'
tmux new-window -n 'ct-04' 'CUDA_VISIBLE_DEVICES=9 bash template-ct.sh mim  4; sh'
tmux new-window -n 'ct-08' 'CUDA_VISIBLE_DEVICES=6 bash template-ct.sh mim  8; sh'
tmux new-window -n 'ct-16' 'CUDA_VISIBLE_DEVICES=7 bash template-ct.sh mim 16; sh'

