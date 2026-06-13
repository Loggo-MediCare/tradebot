"""
AAOI (Applied Optoelectronics) Hybrid PPO + XGBoost 交易信號生成器
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np, pandas as pd, yfinance as yf, joblib
from datetime import datetime
from model_accuracy_tracker import get_model_accuracy_display
from candlestick_patterns import analyze_candlestick_patterns, format_pattern_output, get_pattern_score_adjustment

TICKER     = 'AAOI'
MODEL_FILE = 'xgb_aaoi_model.pkl'
PPO_FILE   = 'ppo_aaoi_improved'
NAME       = 'Applied Optoelectronics'

FEATURE_COLUMNS = [
    'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_position', 'K', 'D',
    'obv', 'obv_ma20', 'sma_10', 'sma_30', 'sma_50', 'sma_200',
    'volatility', 'atr', 'price_change_5d', 'price_change_10d', 'price_change_20d', 'ma50_slope'
]

def add_technical_indicators(df):
    df['sma_10']  = df['close'].rolling(10).mean()
    df['sma_30']  = df['close'].rolling(30).mean()
    df['sma_50']  = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['ema_12']  = df['close'].ewm(span=12).mean()
    df['ema_26']  = df['close'].ewm(span=26).mean()
    d = df['close'].diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    l = (-d.where(d < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + g / (l + 1e-10)))
    df['macd']        = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']
    df['bb_middle']   = df['close'].rolling(20).mean()
    df['bb_std']      = df['close'].rolling(20).std()
    df['bb_upper']    = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower']    = df['bb_middle'] - 2 * df['bb_std']
    df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100).fillna(50)
    lo14 = df['low'].rolling(14).min(); hi14 = df['high'].rolling(14).max()
    df['K'] = ((df['close'] - lo14) / (hi14 - lo14) * 100).fillna(50)
    df['D'] = df['K'].rolling(3).mean()
    df['obv']      = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ma20'] = df['obv'].rolling(20).mean()
    df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
    hl = df['high'] - df['low']
    hc = np.abs(df['high'] - df['close'].shift())
    lc = np.abs(df['low']  - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['price_change_5d']  = df['close'].pct_change(5)  * 100
    df['price_change_10d'] = df['close'].pct_change(10) * 100
    df['price_change_20d'] = df['close'].pct_change(20) * 100
    df['ma50_slope'] = df['sma_50'].diff(5) / df['sma_50'].shift(5) * 100
    return df.bfill().ffill()

def get_trading_signal():
    accuracy_display = get_model_accuracy_display(TICKER)
    print(f"🤖 {TICKER} ({NAME}) | 準確度: {accuracy_display} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    try:
        model = joblib.load(MODEL_FILE)
        print(f"✅ 模型加載成功: {MODEL_FILE}")
    except Exception as e:
        print(f"❌ 模型加載失敗: {e}"); return None

    print(f"\n📊 下載 {TICKER} 最新數據...")
    try:
        df = yf.download(TICKER, period='1y', progress=False)
        if df.empty: print("❌ 無法獲取數據"); return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        df = df.rename(columns={'Close':'close','Volume':'volume','Open':'open','High':'high','Low':'low'}).reset_index()
        print(f"✅ 成功下載 {len(df)} 天數據")
    except Exception as e:
        print(f"❌ 數據下載失敗: {e}"); return None

    df = add_technical_indicators(df)
    latest = df.iloc[-1]
    prev_close    = float(df['close'].iloc[-2])
    current_price = float(latest['close'])
    price_change_pct = (current_price - prev_close) / prev_close * 100

    X = latest[FEATURE_COLUMNS].values.reshape(1, -1)
    print("\n🧠 AI 模型分析中...")
    proba = model.predict_proba(X)[0]
    pred  = model.predict(X)[0]
    buy_prob = proba[1] * 100

    # ── PPO second opinion ───────────────────────────────────────────────────
    ppo_model = None
    ppo_action = 0.0
    ppo_signal_label = 'N/A'
    try:
        from stable_baselines3 import PPO as _PPO
        ppo_model = _PPO.load(PPO_FILE)
        p_price = current_price
        obs = __import__('numpy').array([
            0, 100000, p_price,
            float(latest.get('sma_10', p_price)),
            float(latest.get('sma_30', p_price)),
            float(latest.get('sma_50', p_price)),
            float(latest.get('rsi',   50)),
            float(latest.get('macd',   0)),
            float(latest.get('macd_signal', 0)),
            float(latest.get('bb_upper', p_price)),
            float(latest.get('bb_lower', p_price)),
            float(latest.get('volume',  0)),
            0, 1.0, 1.0,
        ], dtype='float32')
        ppo_act, _ = ppo_model.predict(obs, deterministic=True)
        ppo_action = float(ppo_act[0])
        ppo_signal_label = 'BUY' if ppo_action > 0.3 else ('SELL' if ppo_action < -0.3 else 'HOLD')
        from ppo_backtest_cache import format_ppo_roi_line
        print(format_ppo_roi_line('AAOI', 'AAOI', 'ppo_aaoi_improved', df, ppo_action))
    except Exception as _ppo_err:
        print(f"\n⚠️  PPO model not available: {_ppo_err}")
    # ── End PPO ──────────────────────────────────────────────────────────────

    rsi = float(latest['rsi']); macd = float(latest['macd']); ms = float(latest['macd_signal'])
    s10 = float(latest['sma_10']); s30 = float(latest['sma_30'])
    avg_vol   = float(df['volume'].tail(20).mean())
    vol_ratio = float(latest['volume']) / avg_vol if avg_vol > 0 else 1.0
    candle_dir = 'up' if current_price > prev_close else 'down' if current_price < prev_close else 'flat'

    print(f"\n預測: {'買入機率' if pred == 1 else '不買入'} — 今日買入機率 P(buy): {buy_prob:.1f}%（P(not buy): {proba[0]*100:.1f}%）")
    print("\n" + "=" * 80 + "\n📊 技術指標\n" + "=" * 80)
    print(f"當前價格:        ${current_price:.2f}")
    print(f"今日漲跌幅:      {price_change_pct:+.2f}%")
    print(f"RSI: {rsi:.2f}  {'[超買]' if rsi > 70 else '[超賣]' if rsi < 30 else '[中性]'}")
    print(f"MACD: {macd:.4f}  {'[金叉]' if macd > ms else '[死叉]'}")
    print(f"均線: SMA10={s10:.2f}, SMA30={s30:.2f}  {'[多頭]' if s10 > s30 else '[空頭]'}")
    print(f"布林帶位置:      {float(latest['bb_position']):.1f}%")
    print(f"KD: K={float(latest['K']):.1f}, D={float(latest['D']):.1f}")
    print(f"量比:            {vol_ratio:.2f}x  {'[放量]' if vol_ratio > 1.5 else '[縮量]' if vol_ratio < 0.7 else '[正常]'}")
    print(f"量價方向:        {'價漲量增' if candle_dir == 'up' and vol_ratio >= 1.2 else '價跌量增' if candle_dir == 'down' and vol_ratio >= 1.2 else '中性'}")

    try:
        patterns = analyze_candlestick_patterns(df, days=5)
        print(format_pattern_output(patterns))
        print(f"\n型態評分調整: {get_pattern_score_adjustment(patterns):+.1f} 分")
    except Exception: pass

    print("\n" + "=" * 80 + "\n🎯 交易信號\n" + "=" * 80)
    if pred == 1 and buy_prob >= 60:
        print("🟢 買入信號 (BUY)")
        print(f"   今日買入機率 P(buy): {buy_prob:.1f}%（P(not buy): {100-buy_prob:.1f}%）")
        print(f"   建議買入價格: ${current_price*0.995:.2f} - ${current_price*1.005:.2f}")
    elif pred == 1 and buy_prob >= 52:
        print("🟡 弱買入信號 (WEAK BUY)")
        print(f"   今日買入機率 P(buy): {buy_prob:.1f}%（P(not buy): {100-buy_prob:.1f}%）")
    else:
        print("🔴 不建議買入 (HOLD/WAIT)")
        print(f"   今日買入機率 P(buy): {buy_prob:.1f}%（P(not buy): {100-buy_prob:.1f}%）（信心不足）")
    print("=" * 80)

    return {'ticker': TICKER, 'price': current_price, 'prediction': pred,
            'buy_probability': buy_prob, 'rsi': rsi, 'macd': macd}

if __name__ == "__main__":
    get_trading_signal()
