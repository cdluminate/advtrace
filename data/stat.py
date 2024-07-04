'''
Copyright (C) 2021-2023 Mo Zhou <cdluminate@gmail.com>.
Source code will be released under Apache-2.0 license if paper accepted.
'''
import os
import sys
import glob
import rich
from rich.table import Table
c = rich.get_console()

table = Table(title='Dataset Statistics')
table.add_column('Dataset/Model')
DM = ['ct', 'il', 'sw', 'm4', 'm8']
table.add_column('Split')
SP = ['trn', 'val']
table.add_column('Attack')
ATK = ['bn', 'ad', 'fgsm', 'apgd', 'aa', 'cw', 'nes', 'spsa', 'ga', 'un',
        'pgdl8', 'pgdl2', 'mim', 'fab', 'square', 'jitter', 'difgsm', 'tifgsm',
        'apgddlr', 'fmnl8', 'fa']
ATK_PGD = ['bn', 'ad', 'apgd', 'aa', 'pgdl8', 'mim']
table.add_column('Eps-0')
table.add_column('Eps-2')
table.add_column('Eps-4')
table.add_column('Eps-8')
table.add_column('Eps-16')

for dm in DM:
    for sp in SP:
        if dm in ('ct', 'il', 'sw'):
            atks = ATK
        else:
            atks = ATK_PGD
        for atk in atks:
            prefix = f'data/{sp}-{dm}'
            if atk not in ('bn', 'ad'):
                prefix += f'-{atk}'
            if sp == 'trn' and atk not in ('bn', 'ad'):
                    continue
            #print('glob', prefix + '/*.txt')
            es = [len(glob.glob(prefix + f'-e{e}/*.txt'))//4
                    for e in (0,2,4,8,16)]
            #print('found', dm, sp, atk, *es)
            table.add_row(dm,
                    f'[bold red]{sp}[/bold red]' if sp == 'trn' else f'[bold green]{sp}[/bold green]',
                    atk,
                    *(str(x) for x in es))
    c.print(table)

    table = Table(title='Dataset Statistics')
    table.add_column('Dataset/Model')
    table.add_column('Split')
    table.add_column('Attack')
    table.add_column('Eps-0')
    table.add_column('Eps-2')
    table.add_column('Eps-4')
    table.add_column('Eps-8')
    table.add_column('Eps-16')
