'''
Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
Source code will be released under Apache-2.0 license if paper accepted.
'''
# This file imports models that can be specified by "Train.py -M"

# fashion-mnist (fa)
from . import fa_mlp
from . import fa_mlpd
from . import fa_c2f2  # vanilla
from . import fa_c2f2d # defense (madry)

# cifar10 (ct)
from . import ct_res18
from . import ct_res18d

# imagenet (il)
from . import il_r152
from . import il_swin
from . import il_r50
from . import il_madry4
from . import il_madry8

#from . import attacks
#from . import transfer
