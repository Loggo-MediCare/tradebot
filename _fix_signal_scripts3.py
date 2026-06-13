"""Fix PPO observation shape mismatch in auto-generated signal scripts."""
import os, glob, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
targets  = glob.glob(os.path.join(BASE_DIR, 'get_trading_signal_*.py'))

# The PPO predict block to replace (with wrong shape)
OLD = """    _mtype = locals().get('model_type', 'DQN')
    if _mtype == 'PPO':
        action_raw, _ = model.predict(state, deterministic=True)
        action = int(action_raw[0]) if hasattr(action_raw, '__len__') else int(action_raw)
        # Map PPO continuous [-1,1] to discrete: <-0.1=SELL, >0.1=BUY, else HOLD
        av = float(action_raw[0]) if hasattr(action_raw, '__len__') else float(action_raw)
        action = 2 if av > 0.1 else (0 if av < -0.1 else 1)
        q_values = np.array([0.0, 0.0, 0.0]); q_values[action] = 1.0
    else:
        q_values = model.predict(state.reshape(1, -1), verbose=0)[0]
        action   = int(np.argmax(q_values))"""

# Fix: build 15-dim PPO obs from df for PPO, use DQN state for DQN
NEW = """    _mtype = locals().get('model_type', 'DQN')
    if _mtype == 'PPO':
        # PPO expects 15-dim obs: [shares, bal, close, sma10, sma30, sma50, rsi, macd, macd_sig, bb_up, bb_lo, vol, profit, stock_ratio, cash_ratio]
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
    if OLD in content:
        content = content.replace(OLD, NEW)
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)
        fixed.append(os.path.basename(fpath))

print(f'Fixed {len(fixed)} files:')
for f in fixed:
    print(f'  {f}')
