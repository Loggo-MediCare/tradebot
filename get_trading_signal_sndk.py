"""
SNDK (SanDisk) PPO 交易信號生成器 — PPO 58.97% beats XGBoost 42.22%
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np, pandas as pd, yfinance as yf
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from datetime import datetime, time
from zoneinfo import ZoneInfo
from model_accuracy_tracker import get_model_accuracy_display
from candlestick_patterns import analyze_candlestick_patterns, format_pattern_output, get_pattern_score_adjustment

TICKER     = 'SNDK'
MODEL_FILE = 'ppo_sndk_improved'
NAME       = 'SanDisk'


class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0; self.balance = self.initial_balance
        self.shares_held = 0; self.total_profit = 0
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        price = float(row['close'])
        total_value = self.balance + self.shares_held * price
        return np.array([
            float(self.shares_held), float(self.balance), price,
            float(row.get('sma_10', 0)), float(row.get('sma_30', 0)),
            float(row.get('sma_50', 0)), float(row.get('rsi', 50)),
            float(row.get('macd', 0)), float(row.get('macd_signal', 0)),
            float(row.get('bb_upper', 0)), float(row.get('bb_lower', 0)),
            float(row.get('volume', 0)), float(self.total_profit),
            (self.shares_held * price) / total_value if total_value > 0 else 0,
            self.balance / total_value if total_value > 0 else 1,
        ], dtype=np.float32)

    def step(self, action):
        action = np.clip(float(action[0]) if isinstance(action, np.ndarray) else float(action), -1.0, 1.0)
        price = float(self.df.iloc[self.current_step]['close'])
        if action < -0.1:
            s = int(self.shares_held * abs(action))
            if s > 0: self.balance += s * price; self.shares_held -= s
        elif action > 0.1:
            s = int((self.balance // price) * action)
            if s > 0: self.balance -= s * price; self.shares_held += s
        self.total_profit = (self.balance + self.shares_held * price) - self.initial_balance
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        return self._get_observation(), self.total_profit / self.initial_balance, done, False, {}


def add_technical_indicators(df):
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100).fillna(50)
    lo14 = df['low'].rolling(14).min(); hi14 = df['high'].rolling(14).max()
    df['K'] = ((df['close'] - lo14) / (hi14 - lo14) * 100).fillna(50)
    df['D'] = df['K'].rolling(3).mean()
    return df.bfill().ffill()


def get_trading_signal():
    accuracy_display = get_model_accuracy_display(TICKER)
    print(f"🤖 {TICKER} ({NAME}) | 準確度: {accuracy_display} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    try:
        model = PPO.load(MODEL_FILE)
        print(f"✅ PPO 模型加載成功: {MODEL_FILE}.zip")
    except Exception as e:
        print(f"❌ 模型加載失敗: {e}"); return None

    print(f"\n📊 下載 {TICKER} 最新數據...")
    try:
        df = yf.download(TICKER, period='300d', progress=False, auto_adjust=True)
        if df.empty: print("❌ 無法獲取數據"); return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        df = df.rename(columns={'Close':'close','Volume':'volume','Open':'open','High':'high','Low':'low'}).reset_index()
        print(f"✅ 成功下載 {len(df)} 天數據")
    except Exception as e:
        print(f"❌ 數據下載失敗: {e}"); return None

    df = add_technical_indicators(df)
    latest = df.iloc[-1]
    prev_close = float(df['close'].iloc[-2])
    current_price = float(latest['close'])
    price_change_pct = (current_price - prev_close) / prev_close * 100

    is_intraday = False
    try:
        now_et = datetime.now(ZoneInfo('America/New_York'))
        latest_date = pd.to_datetime(latest['Date']).date()
        if latest_date == now_et.date() and now_et.weekday() < 5 and time(9, 30) <= now_et.time() < time(16, 0):
            is_intraday = True
    except Exception:
        pass

    env = TradingEnv(df); env.current_step = len(df) - 1
    obs = env._get_observation()

    print("\n🧠 AI 模型分析中...")
    action, _ = model.predict(obs, deterministic=True)
    action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)

    rsi = float(latest['rsi']); macd = float(latest['macd']); ms = float(latest['macd_signal'])
    s10 = float(latest['sma_10']); s30 = float(latest['sma_30'])
    bb_upper = float(latest['bb_upper']); bb_lower = float(latest['bb_lower'])
    avg_vol = float(df['volume'].tail(20).mean())
    vol_ratio = float(latest['volume']) / avg_vol if avg_vol > 0 else 1.0
    candle_dir = 'up' if current_price > prev_close else 'down' if current_price < prev_close else 'flat'

    print("\n" + "=" * 80 + "\n📊 技術指標\n" + "=" * 80)
    print(f"當前價格:        ${current_price:.2f}")
    print(f"今日漲跌幅:      {price_change_pct:+.2f}%")
    print(f"RSI: {rsi:.2f}  {'[超買]' if rsi > 70 else '[超賣]' if rsi < 30 else '[中性]'}")
    print(f"MACD: {macd:.4f}  {'[金叉]' if macd > ms else '[死叉]'}")
    print(f"均線: SMA10={s10:.2f}, SMA30={s30:.2f}  {'[多頭]' if s10 > s30 else '[空頭]'}")
    print(f"布林帶位置:      {float(latest['bb_position']):.1f}%")
    print(f"KD: K={float(latest['K']):.1f}, D={float(latest['D']):.1f}")
    if is_intraday:
        vol_status = "(盤中量比未定)"
        volume_direction = "盤中量價方向尚未定型"
    else:
        vol_status = '[放量]' if vol_ratio > 1.5 else '[縮量]' if vol_ratio < 0.7 else '[正常]'
        volume_direction = '價漲量增' if candle_dir == 'up' and vol_ratio >= 1.2 else '價跌量增' if candle_dir == 'down' and vol_ratio >= 1.2 else '中性'

    print(f"量比:            {vol_ratio:.2f}x  {vol_status}")
    print(f"量價方向:        {volume_direction}")

    try:
        patterns = analyze_candlestick_patterns(df, days=5)
        print(format_pattern_output(patterns))
        print(f"\n型態評分調整: {get_pattern_score_adjustment(patterns):+.1f} 分")
    except Exception: pass

    # ── Overbought / low-volume chase guard ─────────────────────────────────
    K_val   = float(latest['K'])
    D_val   = float(latest['D'])
    bb_pos  = float(latest['bb_position'])

    ob_flags = 0
    ob_notes = []
    if rsi > 75:
        ob_flags += 1; ob_notes.append(f"RSI 極端超買 ({rsi:.1f})")
    elif rsi > 70:
        ob_flags += 1; ob_notes.append(f"RSI 超買 ({rsi:.1f})")

    if K_val > 90:
        ob_flags += 1; ob_notes.append(f"KD 極端超買 (K={K_val:.1f})")
    elif K_val > 80:
        ob_flags += 1; ob_notes.append(f"KD 超買 (K={K_val:.1f})")

    if bb_pos > 100:
        ob_flags += 1; ob_notes.append(f"布林上軌外延 ({bb_pos:.1f}%)")
    elif bb_pos > 90:
        ob_flags += 1; ob_notes.append(f"布林接近上軌 ({bb_pos:.1f}%)")

    low_vol = (not is_intraday) and vol_ratio < 0.8
    if low_vol:
        ob_notes.append(f"量比偏低 ({vol_ratio:.2f}x)，缺乏成交量確認")

    # Graduated penalty: each overbought flag trims buy strength
    penalty = 0.0
    if action_value > 0.1 and ob_flags >= 2:
        if ob_flags >= 3:
            penalty = 0.70
        else:
            penalty = 0.45
        if low_vol:
            penalty = min(penalty + 0.20, 0.90)

    adjusted_action = action_value * (1 - penalty) if penalty > 0 else action_value
    is_chasing = penalty > 0
    # ────────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 80 + "\n🎯 交易信號 (PPO)\n" + "=" * 80)
    print(f"模型原始動作值: {action_value:+.4f}")
    if is_chasing:
        print(f"⚠️  追價風險調整: -{penalty*100:.0f}%  →  調整後動作值: {adjusted_action:+.4f}")
        for n in ob_notes:
            print(f"   • {n}")

    if adjusted_action > 0.1:
        print("🟢 買入信號 (BUY)")
        print(f"   AI 強度: {adjusted_action:.2f} / 1.00  {'(原始 1.00，已降級)' if is_chasing else ''}")
        print(f"   建議買入價格: ${current_price*0.995:.2f} - ${current_price*1.000:.2f}")
        print(f"   止損參考: ${current_price*0.95:.2f} (-5%)")
        if is_chasing:
            print(f"   ⚠️  高位追價，建議僅小量試單，等回測支撐再加倉")
    elif adjusted_action < -0.1:
        print("🔴 賣出/觀望 (SELL/WAIT)")
        print(f"   AI 強度: {abs(adjusted_action):.2f} / 1.00")
    else:
        resist = current_price * 1.05 if current_price > bb_upper else bb_upper
        support = bb_upper if current_price > bb_upper else bb_lower
        print("🟡 持有觀望 (HOLD)")
        if is_chasing:
            print(f"   ⚠️  PPO 原始信號強買，但技術指標過熱 + 量能不足 → 降級為觀望")
            for n in ob_notes:
                print(f"   • {n}")
        print(f"   關注支撐位: ${support:.2f}{' (前上軌→支撐)' if current_price > bb_upper else ''}")
        print(f"   關注壓力位: ${resist:.2f}{' ↑突破後估算' if current_price > bb_upper else ''}")

    print("=" * 80)
    final_signal = 'BUY' if adjusted_action > 0.1 else 'SELL' if adjusted_action < -0.1 else 'HOLD'
    return {'ticker': TICKER, 'price': current_price,
            'signal': final_signal, 'action_value': adjusted_action,
            'raw_action': action_value, 'rsi': rsi, 'macd': macd}


if __name__ == "__main__":
    get_trading_signal()
