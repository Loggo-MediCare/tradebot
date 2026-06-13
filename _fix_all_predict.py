"""Final comprehensive fix: replace all raw verbose=0 predict calls."""
import os, glob, sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
targets  = glob.glob(os.path.join(BASE_DIR, 'get_trading_signal_*.py'))

PPO_BLOCK = """    _mtype = locals().get('model_type', 'DQN')
    if _mtype == 'PPO':
        _row = df_feat.iloc[-1] if 'df_feat' in dir() else df.iloc[-1]
        _p   = current_price
        _ppo_obs = np.array([
            0.0, 10000.0, _p,
            float(_row.get('sma_10', _row.get('SMA_10', _p))),
            float(_row.get('sma_30', _row.get('SMA_20', _p))),
            float(_row.get('sma_50', _row.get('SMA_50', _p))),
            float(_row.get('rsi', _row.get('RSI', 50))),
            float(_row.get('macd', _row.get('MACD', 0))),
            float(_row.get('macd_signal', _row.get('Signal_Line', 0))),
            float(_row.get('bb_upper', _p * 1.05)),
            float(_row.get('bb_lower', _p * 0.95)),
            float(df['Volume'].iloc[-1] if 'df' in dir() else 0),
            0.0, 0.0, 1.0,
        ], dtype=np.float32)
        action_raw, _ = model.predict(_ppo_obs, deterministic=True)
        av = float(action_raw[0]) if hasattr(action_raw, '__len__') else float(action_raw)
        action = 1 if av > 0.1 else (2 if av < -0.1 else 0)
        q_values = np.array([0.0, 0.0, 0.0]); q_values[action] = 1.0
    else:
        q_values = model.predict(state.reshape(1, -1), verbose=0)[0]
        action   = int(np.argmax(q_values))"""

fixed = []
for fpath in sorted(targets):
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find any remaining raw verbose=0 predict without _mtype guard
    if 'verbose=0' in content and '_mtype' not in content:
        # Replace the raw predict block
        pattern = r'    q_values = model\.predict\(state\.reshape\(1, -1\), verbose=0\)\[0\]\s*\n    action\s*=\s*int\(np\.argmax\(q_values\)\)'
        if re.search(pattern, content):
            content = re.sub(pattern, PPO_BLOCK, content)
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(content)
            fixed.append(os.path.basename(fpath))

print(f'Fixed {len(fixed)} files:')
for f in fixed:
    print(f'  {f}')
