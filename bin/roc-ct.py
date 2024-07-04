'''
Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
Source code will be released under Apache-2.0 license if paper accepted.
'''
import os
import numpy as np
import matplotlib.pyplot as plt
FS=(4.2,4.0)

def foo(s):
    return [float(x) for x in s.split()]
arc_bim_dr = foo('100 81 75 62 54 30 0')
arc_bim_fpr = foo('100 48 37 20 12 2 0')
nss_bim_dr = foo('100 93 78 61 31 2 1 0')
nss_bim_fpr = foo('100 96 87 75 52 11 2 0')

arc_mim_dr = foo('100 82 76 64 57 33 0')
arc_mim_fpr = foo('100 48 37 20 12 2 0')
nss_mim_dr = foo('100 91 75 58 30 5 0')
nss_mim_fpr = foo('100 96 87 75 52 2 0')

arc_aa_dr = foo('100 78 72 60 52 31 0')
arc_aa_fpr = foo('100 48 37 20 12 2 0')
nss_aa_dr = foo('100 93 78 61 33 4 1 0')
nss_aa_fpr = foo('100 96 87 75 52 13 2 0')


fig = plt.figure(figsize=FS)

plt.plot(nss_bim_fpr, nss_bim_dr, 'x--', label='BIM/NSS')
plt.plot(arc_bim_fpr, arc_bim_dr,  '.-', label='BIM/ARC')

plt.plot(nss_mim_fpr, nss_mim_dr, 'x--', label='MIM/NSS')
plt.plot(arc_mim_fpr, arc_mim_dr,  '.-', label='MIM/ARC')

plt.plot(nss_aa_fpr, nss_aa_dr, 'x--', label='AA/NSS')
plt.plot(arc_aa_fpr, arc_aa_dr,  '.-', label='AA/ARC')

plt.axis('square')
plt.plot([0, 100], [0,100], '--', color='grey')
plt.grid('on')
plt.xlim([0,100])
plt.ylim([0,100])
#plt.legend(['BIM', 'PGD', 'MIM', 'APGD', 'AA'], loc='lower right')
plt.legend(loc='lower right')
plt.title('ROC Curve for $\\varepsilon{=}?$ on ResNet-18')
plt.xlabel('FPR (%)')
plt.ylabel('DR (%)')
#plt.show()
plt.savefig('roc-ct.pdf')
os.system('pdfcrop roc-ct.pdf roc-ct.pdf')
