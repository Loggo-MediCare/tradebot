"""
Shared PPO backtest ROI calculator with JSON caching.
Used by all trading signal scripts to show P&L vs buy-and-hold.

Usage in signal scripts:
    from ppo_backtest_cache import format_ppo_roi_line
    print(format_ppo_roi_line(CODE, TICKER, PPO_MODEL, df, ppo_action))
"""
import os, json, warnings
import numpy as np
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

CACHE_DAYS = 7          # recompute every 7 days
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _run_ppo_backtest(ppo_model_path, df_full):
    """Run PPO through test set (last 20% of df_full). Returns (ppo_return%, bah_return%)."""
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO as PPOModel

    split = int(len(df_full) * 0.8)
    te = df_full.iloc[split:].copy().reset_index(drop=True)
    if len(te) < 10:
        return None, None

    # Standard 15-dim TradingEnv matching all TW signal scripts
    class _Env(gym.Env):
        def __init__(self, df):
            super().__init__(); self.df = df.reset_index(drop=True)
            self.action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
            self.reset()
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.i = 0; self.bal = 10000.; self.sh = 0; self.profit = 0.
            return self._obs(), {}
        def _obs(self):
            r = self.df.iloc[self.i]; p = float(r['close']); tv = self.bal + self.sh * p
            return np.array([float(self.sh), float(self.bal), p,
                float(r.get('sma_10',0)), float(r.get('sma_30',0)), float(r.get('sma_50',0)),
                float(r.get('rsi',50)),   float(r.get('macd',0)),   float(r.get('macd_signal',0)),
                float(r.get('bb_u',0)),   float(r.get('bb_l',0)),   float(r.get('volume',0)),
                float(self.profit),
                (self.sh*p)/tv if tv>0 else 0,
                self.bal/tv if tv>0 else 1], dtype=np.float32)
        def step(self, action):
            a = float(action[0]) if isinstance(action, np.ndarray) else float(action)
            a = np.clip(a, -1, 1); p = float(self.df.iloc[self.i]['close'])
            if a < -0.1:
                s = int(self.sh * abs(a))
                if s > 0: self.bal += s*p; self.sh -= s
            elif a > 0.1:
                s = int((self.bal // p) * a)
                if s > 0: self.bal -= s*p; self.sh += s
            self.profit = (self.bal + self.sh*p) - 10000.
            self.i += 1; done = self.i >= len(self.df) - 1
            return self._obs(), self.profit/10000. + (0.01 if abs(a)>0.1 else 0), done, False, {}

    pm = PPOModel.load(ppo_model_path)
    env = _Env(te)
    obs, _ = env.reset()
    done = False
    while not done:
        act, _ = pm.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(act)

    ppo_r = round(env.profit / 10000. * 100, 2)
    bah_r = round((float(te['close'].iloc[-1]) - float(te['close'].iloc[0])) / float(te['close'].iloc[0]) * 100, 2)
    return ppo_r, bah_r


def get_ppo_backtest_roi(code, ticker, ppo_model_path, df_full):
    """
    Returns (ppo_return_pct, bah_return_pct).
    Caches result in ppo_backtest_{code}.json, recomputes every CACHE_DAYS.
    df_full must have 'close' column and at least 50 rows.
    """
    cache_file = os.path.join(BASE_DIR, f'ppo_backtest_{code}.json')

    # Try cache first
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            age = datetime.now() - datetime.strptime(cached['computed_at'], '%Y-%m-%d %H:%M:%S')
            if age < timedelta(days=CACHE_DAYS):
                return cached['ppo_return'], cached['bah_return']
        except Exception:
            pass

    # Compute backtest
    try:
        full_path = os.path.join(BASE_DIR, ppo_model_path) if not os.path.isabs(ppo_model_path) else ppo_model_path
        ppo_r, bah_r = _run_ppo_backtest(full_path, df_full)
        if ppo_r is None:
            return None, None
        # Determine test period dates
        split = int(len(df_full) * 0.8)
        te = df_full.iloc[split:]
        try:
            t_start = str(te['Date'].iloc[0])[:10]
            t_end   = str(te['Date'].iloc[-1])[:10]
        except Exception:
            t_start = t_end = ''
        result = {
            'code': code, 'ticker': ticker,
            'ppo_return': ppo_r, 'bah_return': bah_r,
            'test_start': t_start, 'test_end': t_end,
            'computed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return ppo_r, bah_r
    except Exception:
        return None, None


def format_ppo_roi_line(code, ticker, ppo_model_path, df_full, ppo_action):
    """
    Returns formatted multi-line string:
      PPO動作:+0.2803  🟢 看多 (已持有→續抱 | 未持有→可進場)
         ℹ️  PPO回測+37.73% vs 買入持有+36.30% (+1.43%)
    """
    if ppo_action > 0.1:
        direction = '🟢 看多 (已持有→續抱 | 未持有→可進場)'
    elif ppo_action < -0.1:
        direction = '🔴 看空 (已持有→考慮減倉 | 未持有→勿追)'
    else:
        direction = '🟡 中性觀望'

    line1 = f"PPO動作:{ppo_action:+.4f}  {direction}"

    ppo_r, bah_r = get_ppo_backtest_roi(code, ticker, ppo_model_path, df_full)
    if ppo_r is not None:
        diff = ppo_r - bah_r
        sign = '+' if diff >= 0 else ''
        line2 = f"   ℹ️  PPO回測{'+' if ppo_r>=0 else ''}{ppo_r:.2f}% vs 買入持有{'+' if bah_r>=0 else ''}{bah_r:.2f}% ({sign}{diff:.2f}%)"
        return line1 + '\n' + line2
    return line1
