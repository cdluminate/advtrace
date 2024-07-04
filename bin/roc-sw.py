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
arc_bim_dr = foo('100 97.1 87 79.2 73 63.3 54.7 49.1 0 ')
arc_bim_fpr = foo('100 75.8 41 23.0 15 6.1 3.3 2.0 0')
nss_bim_dr = foo('100 99 98 96 94 85 76 44 0')
nss_bim_fpr = foo('100 94 65 45 35 22 14 2 0')

arc_mim_dr = foo('100 95 88 82 78 75 68 43 0')
arc_mim_fpr = foo('100 75 56 41 32 26 15 2 0')
nss_mim_dr = foo('100 68 66 61 57 49 44 28 0')
nss_mim_fpr = foo('100 94 65 45 35 23 13 2 0')

arc_aa_dr = foo('100 91 81 75 66 58 52 29 0')
arc_aa_fpr = foo('100 75 56 45 32 23 16 2 0')
nss_aa_dr = foo('100 98 91 83 78 66 58 20 0')
nss_aa_fpr = foo('100 94 65 45 35 21 13 2 0')


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
plt.title('ROC Curve for $\\varepsilon{=}?$ on SwinT-B-IN1K')
plt.xlabel('FPR (%)')
plt.ylabel('DR (%)')
#plt.show()
plt.savefig('roc-sw.pdf')
os.system('pdfcrop roc-sw.pdf roc-sw.pdf')
