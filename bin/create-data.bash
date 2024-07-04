#!/bin/bash
# Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
# Source code will be released under Apache-2.0 license if paper accepted.
set -e
set -x

# NOTE: (1) to create training dataset, first edit Attack.py and change
#    model.getloader('test', ...)
# into loading the training dataset
#    model.getloader('train', ...)
# (2) and then edit the dataset loader python file, and switch the training dataset
# transformation into that for the test dataset for reproducible result.
# (3) edit the dataset loader and turn off shuffle for training set for reproducibility
# (4) edit lib/classify.py, and pass the ground truth labels to exploitation_vectors(...,
# labelleak=labels).
# (5) optionally break to avoid going through whole dataset.
# append e.g. `if idx>=127: exit()` in lib/classify.py

# export CUDA_VISIBLE_DEVICES=0
# export SKIP=96
# export SKIP=4

create_ () {
	local model=$1
	local prefix=$2
	local e=$3
	local tag=$4
	local dir=${prefix}-e${e}
	if test -d ${dir} && test -z "${SKIP}"; then
		echo skipping ${dir} because it already exists
		return
	fi
	mkdir -p ${dir}
	DIR=${dir} TAG=${tag} python3 Attack.py -M ${model} -A UT:PGDT -v -e ${e}
}

create_bn ()            { create_ ${1} ${2} ${3} BN; }
create_ad ()            { create_ ${1} ${2} ${3} AD; }
create_un ()            { create_ ${1} ${2} ${3} UN; }
create_ga ()            { create_ ${1} ${2} ${3} GA; }
create_fgsm ()          { create_ ${1} ${2} ${3} FGSM; }
create_aa ()            { create_ ${1} ${2} ${3} AA; }
create_apgd ()          { create_ ${1} ${2} ${3} APGD; }
create_cw ()            { create_ ${1} ${2} ${3} CW; }
create_nes ()           { create_ ${1} ${2} ${3} NES; }
create_spsa ()          { create_ ${1} ${2} ${3} SPSA; }
create_mim ()           { create_ ${1} ${2} ${3} MIM; }
create_pgdl8()          { create_ ${1} ${2} ${3} PGDl8; }
create_pgdl2()          { create_ ${1} ${2} ${3} PGDl2; }
create_square()         { create_ ${1} ${2} ${3} Square; }
create_fab ()           { create_ ${1} ${2} ${3} FAB; }
create_fmnl8()          { create_ ${1} ${2} ${3} FMNl8; }
create_difgsm()         { create_ ${1} ${2} ${3} DIFGSM; }
create_tifgsm()         { create_ ${1} ${2} ${3} TIFGSM; }
create_jitter()         { create_ ${1} ${2} ${3} Jitter; }
create_apgddlr()        { create_ ${1} ${2} ${3} APGDdlr; }
create_fa()             { create_ ${1} ${2} ${3} FA; }
create_fa_arcfa()       { ARC_TRAJ=fa create_ ${1} ${2} ${3} FA; }
create_dlr()            { create_ ${1} ${2} ${3} DLR; }
create_biml2()          { create_ ${1} ${2} ${3} BIMl2; }
create_lm()             { create_ ${1} ${2} ${3} LM; }
create_in()             { create_ ${1} ${2} ${3} IN; }

main () {
model=$1
prefix=$2

create_bn ${model} ${prefix} 0
for i in 2 4 8 16; do
#for i in 2 4 8 16; do
#for i in 1 3 5 6    7 9 10 11    12 13 14 15; do
	create_ad ${model} ${prefix} ${i}
done
}

echo PLEASE edit this file and choose the function you would like to invoke
#main $1 $2

#main ct_res18 data/val-ct

#for e in 2 4 8 16; do
#create_un ct_res18 data/test-junk 2
#done

#export SKIP=96
#main il_r152 val-il
#main il_swin val-sw
#export SKIP=4
#main ct_res18 val-ct
