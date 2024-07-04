#!/bin/bash
# Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
# Source code will be released under Apache-2.0 license if paper accepted.
set -e
set -x

tmux new-window 'CUDA_VISIBLE_DEVICES=6 DIR=discuss/ct-cwt-e2  TAG=CW ARC_TRAJ=cwt python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 2 ; sh'
tmux new-window 'CUDA_VISIBLE_DEVICES=7 DIR=discuss/ct-cwt-e4  TAG=CW ARC_TRAJ=cwt python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 4 ; sh'
tmux new-window 'CUDA_VISIBLE_DEVICES=8 DIR=discuss/ct-cwt-e8  TAG=CW ARC_TRAJ=cwt python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 8 ; sh'
tmux new-window 'CUDA_VISIBLE_DEVICES=9 DIR=discuss/ct-cwt-e16 TAG=CW ARC_TRAJ=cwt python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 16; sh'

tmux new-window 'CUDA_VISIBLE_DEVICES=6 DIR=discuss/il-cwt-e2  TAG=CW ARC_TRAJ=cwt python3 Attack.py -M il_r152 -A UT:PGDT -v -e 2  ; sh'
tmux new-window 'CUDA_VISIBLE_DEVICES=7 DIR=discuss/il-cwt-e4  TAG=CW ARC_TRAJ=cwt python3 Attack.py -M il_r152 -A UT:PGDT -v -e 4  ; sh'
tmux new-window 'CUDA_VISIBLE_DEVICES=8 DIR=discuss/il-cwt-e8  TAG=CW ARC_TRAJ=cwt python3 Attack.py -M il_r152 -A UT:PGDT -v -e 8  ; sh'
tmux new-window 'CUDA_VISIBLE_DEVICES=9 DIR=discuss/il-cwt-e16 TAG=CW ARC_TRAJ=cwt python3 Attack.py -M il_r152 -A UT:PGDT -v -e 16 ; sh'

tmux new-window 'CUDA_VISIBLE_DEVICES=6 DIR=discuss/sw-cwt-e2  TAG=CW ARC_TRAJ=cwt python3 Attack.py -M il_swin -A UT:PGDT -v -e 2  ; sh'
tmux new-window 'CUDA_VISIBLE_DEVICES=7 DIR=discuss/sw-cwt-e4  TAG=CW ARC_TRAJ=cwt python3 Attack.py -M il_swin -A UT:PGDT -v -e 4  ; sh'
tmux new-window 'CUDA_VISIBLE_DEVICES=8 DIR=discuss/sw-cwt-e8  TAG=CW ARC_TRAJ=cwt python3 Attack.py -M il_swin -A UT:PGDT -v -e 8  ; sh'
tmux new-window 'CUDA_VISIBLE_DEVICES=9 DIR=discuss/sw-cwt-e16 TAG=CW ARC_TRAJ=cwt python3 Attack.py -M il_swin -A UT:PGDT -v -e 16 ; sh'
