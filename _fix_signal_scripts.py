"""Fix two bugs in auto-generated signal scripts:
1. get_best_model_type() returns 5 values but scripts unpack 3
2. NTNT$ double-prefix in 8064 script
"""
import os, glob, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Files to fix (auto-generated DQN scripts)
targets = glob.glob(os.path.join(BASE_DIR, 'get_trading_signal_*.py'))

fixed = []
for fpath in sorted(targets):
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()

    changed = False

    # Fix 1: unpack 3 → unpack 5
    old = 'best_type, score_ppo, score_dqn = get_best_model_type(TICKER)'
    new = 'best_type, score_ppo, score_dqn, score_xgb, score_hyb = get_best_model_type(TICKER)'
    if old in content:
        content = content.replace(old, new)
        changed = True

    # Fix 2: NTNT$ → NT$
    if 'NTNT$' in content:
        content = content.replace('NTNT$', 'NT$')
        changed = True

    # Fix 3: model.predict() call - DQN vs PPO
    old3 = '    q_values = model.predict(state.reshape(1, -1), verbose=0)[0]\n    action   = int(np.argmax(q_values))'
    new3 = '''    _mtype = locals().get('model_type', 'DQN')
    if _mtype == 'PPO':
        action_raw, _ = model.predict(state, deterministic=True)
        av = float(action_raw[0]) if hasattr(action_raw, '__len__') else float(action_raw)
        action = 2 if av > 0.1 else (0 if av < -0.1 else 1)
        q_values = np.array([0.0, 0.0, 0.0]); q_values[action] = 1.0
    else:
        q_values = model.predict(state.reshape(1, -1), verbose=0)[0]
        action   = int(np.argmax(q_values))'''
    if old3 in content:
        content = content.replace(old3, new3)
        changed = True

    if changed:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)
        fixed.append(os.path.basename(fpath))

print(f'Fixed {len(fixed)} files:')
for f in fixed:
    print(f'  {f}')
