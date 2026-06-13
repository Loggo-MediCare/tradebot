"""Replace model_type references with safe fallback in all signal scripts."""
import os, glob, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
targets  = glob.glob(os.path.join(BASE_DIR, 'get_trading_signal_*.py'))

fixed = []
for fpath in sorted(targets):
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()

    changed = False

    # Fix: replace bare 'model_type' in predict block with safe locals() fallback
    old = "    if model_type == 'PPO':"
    new = "    _mtype = locals().get('model_type', 'DQN')\n    if _mtype == 'PPO':"
    if old in content and '_mtype' not in content:
        content = content.replace(old, new)
        changed = True

    if changed:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)
        fixed.append(os.path.basename(fpath))

print(f'Fixed {len(fixed)} files:')
for f in fixed:
    print(f'  {f}')
