# Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
# Source code will be released under Apache-2.0 license if paper accepted.
stat:
	python3 data/stat.py

create-cifar-trn:
	bash -c "export ARC_DATA_SPLIT=train; source bin/create-data.bash; create_bn ct_res18 data/trn-ct  0;"
	bash -c "export ARC_DATA_SPLIT=train; source bin/create-data.bash; create_ad ct_res18 data/trn-ct  2;"
	bash -c "export ARC_DATA_SPLIT=train; source bin/create-data.bash; create_ad ct_res18 data/trn-ct  4;"
	bash -c "export ARC_DATA_SPLIT=train; source bin/create-data.bash; create_ad ct_res18 data/trn-ct  8;"
	bash -c "export ARC_DATA_SPLIT=train; source bin/create-data.bash; create_ad ct_res18 data/trn-ct 16;"

create-cifar-val:
	bash -c "source bin/create-data.bash; create_bn ct_res18 data/val-ct  0;"
	bash -c "source bin/create-data.bash; create_ad ct_res18 data/val-ct  2;"
	bash -c "source bin/create-data.bash; create_ad ct_res18 data/val-ct  4;"
	bash -c "source bin/create-data.bash; create_ad ct_res18 data/val-ct  8;"
	bash -c "source bin/create-data.bash; create_ad ct_res18 data/val-ct 16;"
