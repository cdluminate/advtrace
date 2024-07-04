# Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
# Source code will be released under Apache-2.0 license if paper accepted.
f1_data(){
	echo use bin/create_data.sh
}
f1(){
	python3 bin/util.py polar-arcm -p data/val-ct --save figure4-f1.svg
}
f2_data(){
	export TAG=AD
	export DIR=data/val-ct-arcga-e2
	if ! test -d ${DIR}; then
	ARC_TRAJ='gaussian' python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 2
	fi
	export TAG=AD
	export DIR=data/val-ct-arcga-e16
	if ! test -d ${DIR}; then
	ARC_TRAJ='gaussian' python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 16
	fi
}
f2(){
	python3 bin/util.py polar-arcm -p data/val-ct-arcga --save figure4-f2.svg
}
f3_data(){
	export TAG=AD
	export DIR=data/val-ct-arcun-e2
	if ! test -d ${DIR}; then
	ARC_TRAJ='uniform' python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 2
	fi
	export TAG=AD
	export DIR=data/val-ct-arcun-e16
	if ! test -d ${DIR}; then
	ARC_TRAJ='uniform' python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 16
	fi
}
f3(){
	python3 bin/util.py polar-arcm -p data/val-ct-arcun --save figure4-f3.svg
}
f4_data(){
	export TAG=BIMl2
	export DIR=data/fig-ct-biml2-e2
	if ! test -d ${DIR}; then
	python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 2
	fi
	export TAG=BIMl2
	export DIR=data/fig-ct-biml2-e16
	if ! test -d ${DIR}; then
	python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 16
	fi
}
f4(){
	python3 bin/util.py polar-arcm -p data/fig-ct-biml2 --save figure4-f4.svg
}
f5_data(){
	export TAG=AD
	export DIR=data/val-ct-arcl2-e2
	if ! test -d ${DIR}; then
	ARC_TRAJ=pgdtl2 python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 2
	fi
	export TAG=AD
	export DIR=data/val-ct-arcl2-e16
	if ! test -d ${DIR}; then
	ARC_TRAJ=pgdtl2 python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 16
	fi
}
f5(){
	python3 bin/util.py polar-arcm -p data/val-ct-arcl2 --save figure4-f5.svg
}
f6_data(){
	export TAG=BIMl2
	export DIR=data/val-ct-biml2-arcl2-e2
	if ! test -d ${DIR}; then
	ARC_TRAJ=pgdtl2 python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 2
	fi
	export TAG=BIMl2
	export DIR=data/val-ct-biml2-arcl2-e16
	if ! test -d ${DIR}; then
	ARC_TRAJ=pgdtl2 python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 16
	fi
}
f6(){
	python3 bin/util.py polar-arcm -p data/val-ct-biml2-arcl2 --save figure4-f6.svg
}
f7(){
	python3 bin/util.py polar-arcm -p data/val-ct-fa --save figure4-f7.svg
}
f8_data(){
	export TAG=AD
	export DIR=data/val-ct-arcfa-e2
	if ! test -d ${DIR}; then
	ARC_TRAJ='fa' python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 2
	fi
	export TAG=AD
	export DIR=data/val-ct-arcfa-e16
	if ! test -d ${DIR}; then
	ARC_TRAJ='fa' python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 16
	fi
}
f8(){
	python3 bin/util.py polar-arcm -p data/val-ct-arcfa --save figure4-f8.svg
}
f9_data(){
	export TAG=FA
	export DIR=data/val-ct-fa-arcfa-e2
	if ! test -d ${DIR}; then
	ARC_TRAJ='fa' python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 2
	fi
	export TAG=FA
	export DIR=data/val-ct-fa-arcfa-e16
	if ! test -d ${DIR}; then
	ARC_TRAJ='fa' python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 16
	fi
}
f9(){
	python3 bin/util.py polar-arcm -p data/val-ct-fa-arcfa --save figure4-f9.svg
}
f10_data(){
	export TAG=DLR
	export DIR=data/fig-ct-dlr-e2
	if ! test -d ${DIR}; then
	python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 2
	fi
	export TAG=DLR
	export DIR=data/fig-ct-dlr-e16
	if ! test -d ${DIR}; then
	python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 16
	fi
}
f10(){
	python3 bin/util.py polar-arcm -p data/fig-ct-dlr --save figure4-f10.svg
}
f11_data(){
	export TAG=AD
	export DIR=data/val-ct-arcdlr-e2
	if ! test -d ${DIR}; then
	ARC_TRAJ='dlr' python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 2
	fi
	export TAG=AD
	export DIR=data/val-ct-arcdlr-e16
	if ! test -d ${DIR}; then
	ARC_TRAJ='dlr' python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 16
	fi
}
f11(){
	python3 bin/util.py polar-arcm -p data/val-ct-arcdlr --save figure4-f11.svg
}
f12_data(){
	export TAG=DLR
	export DIR=data/val-ct-dlr-arcdlr-e2
	if ! test -d ${DIR}; then
	ARC_TRAJ='dlr' python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 2
	fi
	export TAG=DLR
	export DIR=data/val-ct-dlr-arcdlr-e16
	if ! test -d ${DIR}; then
	ARC_TRAJ='dlr' python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 16
	fi
}
f12(){
	python3 bin/util.py polar-arcm -p data/val-ct-dlr-arcdlr --save figure4-f12.svg
}
f13(){
	python3 bin/util.py polar-arcm -p data/val-ct-fgsm --save figure4-f13.svg
}
f14(){
	python3 bin/util.py polar-arcm -p data/val-ct-nes --save figure4-f14.svg
}
f15(){
	python3 bin/util.py polar-arcm -p data/val-ct-spsa --save figure4-f15.svg
}
f16(){
	python3 bin/util.py polar-arcm -p data/val-ct-ga --save figure4-f16.svg
}
f17_data(){
	export TAG=AD
	export DIR=data/val-ct-arcmlike-e2
	if ! test -d ${DIR}; then
	ARC_LABEL='mlike' python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 2
	fi
	export TAG=AD
	export DIR=data/val-ct-arcmlike-e16
	if ! test -d ${DIR}; then
	ARC_LABEL='mlike' python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 16
	fi
}
f17(){
	python3 bin/util.py polar-arcm -p data/val-ct-arcmlike --save figure4-f17.svg
}
f18_data(){
	export TAG=AD
	export DIR=data/val-ct-arcrand-e2
	if ! test -d ${DIR}; then
	ARC_LABEL='rand' python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 2
	fi
	export TAG=AD
	export DIR=data/val-ct-arcrand-e16
	if ! test -d ${DIR}; then
	ARC_LABEL='rand' python3 Attack.py -M ct_res18 -A UT:PGDT -v -e 16
	fi
}
f18(){
	python3 bin/util.py polar-arcm -p data/val-ct-arcrand --save figure4-f18.svg
}



for i in $(seq 1 18); do
	eval "f${i}"
	inkscape -o figure4-f${i}.pdf figure4-f${i}.svg
	pdfcrop figure4-f${i}.pdf figure4-f${i}.pdf
done
