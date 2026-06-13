#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch fix fillna(method=) issues"""

from pathlib import Path

count = 0
for fpath in sorted(Path('.').glob('get_trading_signal_*.py')):
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'fillna(method=' not in content:
        continue
    
    new_content = content.replace(
        "df = df.fillna(method='bfill').fillna(method='ffill')",
        "df = df.bfill().ffill()"
    )
    
    if new_content != content:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'Fixed: {fpath.name}')
        count += 1

print(f'Total: {count} files fixed')
