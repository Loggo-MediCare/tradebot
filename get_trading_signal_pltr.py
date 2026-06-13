"""
PLTR (Palantir) 雙模型交易信號生成器
XGBoost (55.87%) + PPO (58.55%) — 加權組合信號
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from datetime import datetime
from model_accuracy_tracker import get_model_accuracy_display
from candlestick_patterns import analyze_candlestick_patterns, format_pattern_output, get_pattern_score_adjustment

TICKER     = 'PLTR'
XGB_FILE   = 'xgb_pltr_model.pkl'
PPO_FILE   = 'ppo_pltr_improved'
XGB_ACC    = 55.87
PPO_ACC    = 58.55

class TradingEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.reset()
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.i = 0; self.bal = 10_000.0; self.sh = 0; self.profit = 0.0
        return self._obs(), {}
    def _obs(self):
        r = self.df.iloc[self.i]; p = float(r['close']); tv = self.bal + self.sh * p
        return np.array([float(self.sh), float(self.bal), p,
                         float(r.get('sma_10',0)), float(r.get('sma_30',0)), float(r.get('sma_50',0)),
                         float(r.get('rsi',50)),   float(r.get('macd',0)),   float(r.get('macd_signal',0)),
                         float(r.get('bb_upper',0)), float(r.get('bb_lower',0)), float(r.get('volume',0)),
                         float(self.profit),
                         (self.sh * p) / tv if tv > 0 else 0,
                         self.bal / tv if tv > 0 else 1], dtype=np.float32)
    def step(self, action): pass  # inference only

def add_technical_indicators(df):
    """添加技術指標"""
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100).fillna(50)

    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['K'] = ((df['close'] - low_14) / (high_14 - low_14) * 100).fillna(50)
    df['D'] = df['K'].rolling(3).mean()

    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ma20'] = df['obv'].rolling(20).mean()

    df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()

    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()

    df['price_change_5d'] = df['close'].pct_change(5) * 100
    df['price_change_10d'] = df['close'].pct_change(10) * 100
    df['price_change_20d'] = df['close'].pct_change(20) * 100

    df['ma50_slope'] = df['sma_50'].diff(5) / df['sma_50'].shift(5) * 100

    return df.bfill().ffill()

def get_trading_signal():
    """生成今日交易信號"""
    # 壓縮標題區塊
    accuracy_display = get_model_accuracy_display(TICKER)
    print(f"🤖 {TICKER} (Palantir) | 準確度: {accuracy_display} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    print("📊 模型升級: XGBoost (66.95%) 取代 PPO (63.5%) | 提升 +3.45%")
    print("=" * 80)

    # 1. 加載兩個模型
    xgb_model = ppo_model = None
    try:
        xgb_model = joblib.load(XGB_FILE)
        print(f"✅ XGBoost 加載成功 (acc={XGB_ACC}%)")
    except Exception as e:
        print(f"⚠️  XGBoost 加載失敗: {e}")
    try:
        ppo_model = PPO.load(PPO_FILE)
        print(f"✅ PPO     加載成功 (acc={PPO_ACC}%)")
    except Exception as e:
        print(f"⚠️  PPO 加載失敗: {e}")
    if not xgb_model and not ppo_model:
        print("❌ 兩個模型均加載失敗"); return None

    # 2. 下載最新數據
    print(f"\n📊 下載 {TICKER} 最新數據...")
    try:
        df = yf.download(TICKER, period='1y', progress=False)
        if df.empty:
            print("❌ 無法獲取數據")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df.rename(columns={
            'Close': 'close', 'Volume': 'volume',
            'Open': 'open', 'High': 'high', 'Low': 'low'
        }).reset_index()

        print(f"✅ 成功下載 {len(df)} 天數據")
    except Exception as e:
        print(f"❌ 數據下載失敗: {e}")
        return None

    # 3. 添加技術指標
    df = add_technical_indicators(df)

    # 4. 獲取最新數據
    latest = df.iloc[-1]
    current_price = float(latest['close'])

    feature_columns = [
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_position', 'K', 'D', 'obv', 'obv_ma20',
        'sma_10', 'sma_30', 'sma_50', 'sma_200',
        'volatility', 'atr',
        'price_change_5d', 'price_change_10d', 'price_change_20d',
        'ma50_slope'
    ]

    X = latest[feature_columns].values.reshape(1, -1)

    # 5. 雙模型預測
    print("\n🧠 雙模型 AI 分析中...")

    xgb_action = 0.0
    ppo_action  = 0.0
    buy_prob    = 50.0

    # XGBoost
    if xgb_model:
        proba = xgb_model.predict_proba(X)[0]
        pred  = xgb_model.predict(X)[0]
        buy_prob = proba[1] * 100
        xgb_action = buy_prob / 100 * 2 - 1   # maps [0,1] → [-1,1]
        xgb_signal = '🟢 BUY'  if buy_prob >= 60 else ('🟡 WEAK BUY' if buy_prob >= 52 else '🔴 HOLD/WAIT')
        print(f"  XGBoost ({XGB_ACC}%): 今日買入機率 P(buy)={buy_prob:.1f}%（P(not buy): {100-buy_prob:.1f}%）  →  {xgb_signal}")

    # PPO
    if ppo_model:
        env_tmp = TradingEnv(df)
        env_tmp.i = len(df) - 1
        obs = env_tmp._obs()
        act, _ = ppo_model.predict(obs, deterministic=True)
        ppo_action = float(act[0]) if isinstance(act, np.ndarray) else float(act)
        ppo_signal = '🟢 BUY' if ppo_action > 0.1 else ('🔴 SELL' if ppo_action < -0.1 else '🟡 HOLD')
        from ppo_backtest_cache import format_ppo_roi_line
        print(format_ppo_roi_line('PLTR', 'PLTR', 'ppo_pltr_improved', df, ppo_action))

    # Weighted combined action
    total_w = (XGB_ACC if xgb_model else 0) + (PPO_ACC if ppo_model else 0)
    combined_action = ((XGB_ACC * xgb_action if xgb_model else 0) +
                       (PPO_ACC * ppo_action  if ppo_model else 0)) / (total_w or 1)

    # Detect model conflict (one BUY, one SELL)
    models_conflict = (xgb_model and ppo_model and
                       xgb_action > 0.1 and ppo_action < -0.1 or
                       xgb_action < -0.1 and ppo_action > 0.1)
    print(f"\n  Combined action (weighted): {combined_action:+.3f}")

    # 6. 技術指標分析
    print("\n" + "=" * 80)
    print("📊 技術指標")
    print("=" * 80)
    print(f"當前價格: ${current_price:.2f}")
    print(f"RSI: {float(latest['rsi']):.2f} {'[超買]' if float(latest['rsi']) > 70 else '[超賣]' if float(latest['rsi']) < 30 else '[中性]'}")
    print(f"MACD: {float(latest['macd']):.4f} {'[金叉]' if float(latest['macd']) > float(latest['macd_signal']) else '[死叉]'}")
    print(f"均線: SMA10=${float(latest['sma_10']):.2f}, SMA30=${float(latest['sma_30']):.2f} {'[多頭]' if float(latest['sma_10']) > float(latest['sma_30']) else '[空頭]'}")
    print(f"布林帶位置: {float(latest['bb_position']):.1f}%")
    print(f"KD: K={float(latest['K']):.1f}, D={float(latest['D']):.1f}")

    # 6.5 蠟燭圖型態分析
    try:
        patterns = analyze_candlestick_patterns(df, days=5)
        print(format_pattern_output(patterns))

        # 獲取型態評分調整
        pattern_adjustment = get_pattern_score_adjustment(patterns)
        print(f"\n型態評分調整: {pattern_adjustment:+.1f} 分")
    except Exception as e:
        print(f"\n   ⚠️  型態分析失敗: {e}")
        pattern_adjustment = 0

    # 7. 交易建議
    print("\n" + "=" * 80)
    print("🎯 交易信號")
    print("=" * 80)

    if models_conflict:
        print(f"🟠 ⚠️  模型衝突 (BUYSELL)")
        print(f"   XGBoost ({XGB_ACC}%) 看{'多 ↑' if xgb_action>0 else '空 ↓'}  ↔  PPO ({PPO_ACC}%) 看{'多 ↑' if ppo_action>0 else '空 ↓'}")
        print(f"   加權動作值: {combined_action:+.3f}  →  {'偏多' if combined_action>0 else '偏空'}")
        # Quick tie-breaker using available indicators
        rsi_v   = float(latest['rsi'])
        macd_v  = float(latest['macd'])
        mac_sig = float(latest['macd_signal'])
        sma10   = float(latest['sma_10']); sma30 = float(latest['sma_30'])
        tb = (1 if macd_v > mac_sig else -1) + (1 if sma10 > sma30 else -1) + \
             (-1 if rsi_v > 70 else (1 if rsi_v < 30 else 0))
        print(f"\n  🔍 TIE-BREAKER 技術面")
        print(f"  MACD:  {'金叉 ✅' if macd_v>mac_sig else '死叉 ❌'}")
        print(f"  均線:  {'多頭 ✅' if sma10>sma30 else '空頭 ❌'}")
        print(f"  RSI:   {rsi_v:.1f}  {'超買⚠️' if rsi_v>70 else ('超賣✅' if rsi_v<30 else '中性')}")
        tb_v = '多方勝 → 偏向 BUY' if tb>0 else ('空方勝 → 偏向 SELL' if tb<0 else '平手 → 依加權值')
        print(f"  裁決:  得分 {tb:+d}  →  {tb_v}")
        print()  # blank line before final signal
    # ── Final signal (always runs — tie-breaker already printed above) ──────
    if combined_action > 0.1:
        print(f"🟢 {'[衝突解決] ' if models_conflict else ''}買入信號 (BUY)")
        print(f"   綜合動作值: {combined_action:+.3f}")
        print(f"   建議買入價格: ${current_price * 0.995:.2f} - ${current_price * 1.005:.2f}")
        print(f"   止損: ${current_price * 0.95:.2f} (-5%)")
    elif combined_action < -0.1:
        print(f"🔴 賣出/觀望 (SELL/WAIT)")
        print(f"   綜合動作值: {combined_action:+.3f}")
    else:
        print(f"🟡 持有觀望 (HOLD/WAIT)")
        print(f"   綜合動作值: {combined_action:+.3f}")

    print(f"\n   XGBoost {XGB_ACC}% | PPO {PPO_ACC}% | 加權平均")

    print("=" * 80)

    return {
        'ticker': TICKER,
        'price': current_price,
        'prediction': prediction if 'prediction' in dir() else 0,
        'buy_probability': buy_prob,
        'rsi': float(latest['rsi']),
        'macd': float(latest['macd'])
    }

if __name__ == "__main__":
    get_trading_signal()
