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
arc_bim_dr = foo('100 90 76 73 64 49 31 0')
arc_bim_fpr = foo('100 77 45 36 24 10 1 0')
nss_bim_dr = foo('100 99 99 97 94 85 76 41 0')
nss_bim_fpr = foo('100 94 63 47 35 20 12 2 0')

arc_mim_dr = foo('100 85 70 64 53 40 23 0')
arc_mim_fpr = foo('100 77 45 36 24 10 1 0')
nss_mim_dr = foo('100 85 83 78 73 62 53 32 0')
nss_mim_fpr = foo('100 86 67 47 35 20 12 2 0')

arc_aa_dr = foo('100 84 63 56 46 34 19 0')
arc_aa_fpr = foo('100 77 45 36 24 10 1 0')
nss_aa_dr = foo('100 98 91 83 78 63 53 28 0')
nss_aa_fpr = foo('100 94 67 47 38 20 12 2 0')


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
plt.title('ROC Curve for $\\varepsilon{=}?$ on ResNet-152')
plt.xlabel('FPR (%)')
plt.ylabel('DR (%)')
#plt.show()
plt.savefig('roc-il.pdf')
os.system('pdfcrop roc-il.pdf roc-il.pdf')
