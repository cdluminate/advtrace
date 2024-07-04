# Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
# Source code will be released under Apache-2.0 license if paper accepted.
import numpy as np
import matplotlib.pyplot as plt

v0 = np.vstack([
        np.loadtxt('datanss/val-ct-e0/nss2-0.txt'),
        np.loadtxt('datanss/val-ct-e0/nss2-1.txt'),
        np.loadtxt('datanss/val-ct-e0/nss2-2.txt'),
        np.loadtxt('datanss/val-ct-e0/nss2-3.txt'),
        np.loadtxt('datanss/val-ct-e0/nss2-4.txt'),
        np.loadtxt('datanss/val-ct-e0/nss2-5.txt'),
        np.loadtxt('datanss/val-ct-e0/nss2-6.txt'),
        np.loadtxt('datanss/val-ct-e0/nss2-7.txt'),
        np.loadtxt('datanss/val-ct-e0/nss2-8.txt'),
        np.loadtxt('datanss/val-ct-e0/nss2-9.txt'),
        ])
v6 = np.vstack([
        np.loadtxt('datanss/val-ct-e16/nss2-0.txt'),
        np.loadtxt('datanss/val-ct-e16/nss2-1.txt'),
        np.loadtxt('datanss/val-ct-e16/nss2-2.txt'),
        np.loadtxt('datanss/val-ct-e16/nss2-3.txt'),
        np.loadtxt('datanss/val-ct-e16/nss2-4.txt'),
        np.loadtxt('datanss/val-ct-e16/nss2-5.txt'),
        np.loadtxt('datanss/val-ct-e16/nss2-6.txt'),
        np.loadtxt('datanss/val-ct-e16/nss2-7.txt'),
        np.loadtxt('datanss/val-ct-e16/nss2-8.txt'),
        np.loadtxt('datanss/val-ct-e16/nss2-9.txt'),
        ])

plt.figure()
plt.scatter(v6[:,0], v6[:,1], color='red')
plt.scatter(v0[:,0], v0[:,1], color='blue')
plt.show()
