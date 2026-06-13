#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch fix for hardcoded model paths"""

import os
import re
from pathlib import Path

fixed = 0
for fpath in Path('.').glob('get_trading_signal_*.py'):
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'Users\\Silvi' not in content:
        continue
        
    pattern = r'model_path\s*=\s*r?"C:\\Users\\Silvi\\Projects\\trading-bot\\([^"]+)"'
    matches = re.findall(pattern, content)
    
    if not matches:
        continue
    
    model_name = matches[0]
    old_line = f'model_path = r"C:\\Users\\Silvi\\Projects\\trading-bot\\{model_name}"'
    
    new_code = f'''import os
    model_filename = "{model_name}"
    possible_paths = [os.path.join(os.getcwd(), model_filename), model_filename]
    model_path = None
    for path in possible_paths:
        if os.path.exists(f"{{path}}.zip") or os.path.exists(path):
            model_path = path
            break
    if not model_path:
        model_path = model_filename'''
    
    new_content = content.replace(old_line, new_code)
    
    if new_content != content:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'Fixed: {fpath.name}')
        fixed += 1

print(f'\nTotal: {fixed} files fixed')
