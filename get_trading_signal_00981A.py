"""
台股 00981A (主動統一台股增長) AI 交易信号生成器
使用 XGBoost + PPO 雙模型生成今日交易策略
"""
import os, sys, io, warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from datetime import datetime, timedelta

from model_accuracy_tracker import get_model_accuracy_display, should_mute_ai_signal
from tw_news_tracker import print_tavily_news_tw

CODE   = '00981A'
TICKER = '00981A.TW'
NAME   = '主動統一台股增長'

XGB_MODEL = 'xgb_00981A_tw_model.pkl'
PPO_MODEL = 'ppo_00981A_tw_improved'

FEAT = ['rsi','macd','macd_signal','macd_hist','bb_position','K','D','obv','obv_ma20',
        'sma_10','sma_30','sma_50','volatility','atr',
        'price_change_5d','price_change_10d','price_change_20d','ma50_slope']


# ── Trading environment (must match training) ────────────────────────────────
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
                         float(r.get('bb_u',0)),   float(r.get('bb_l',0)),   float(r.get('volume',0)),
                         float(self.profit),
                         (self.sh * p) / tv if tv > 0 else 0,
                         self.bal / tv if tv > 0 else 1], dtype=np.float32)
    def step(self, action):
        a = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        a = np.clip(a, -1, 1); p = float(self.df.iloc[self.i]['close'])
        if a < -0.1:
            s = int(self.sh * abs(a))
            if s > 0: self.bal += s * p; self.sh -= s
        elif a > 0.1:
            s = int((self.bal // p) * a)
            if s > 0: self.bal -= s * p; self.sh += s
        self.profit = (self.bal + self.sh * p) - 10_000.0
        self.i += 1; done = self.i >= len(self.df) - 1
        return self._obs(), self.profit / 10_000.0 + (0.01 if abs(a) > 0.1 else 0), done, False, {}


# ── Feature engineering ───────────────────────────────────────────────────────
def build_features(df):
    df = df.copy()
    df['sma_10']  = df['close'].rolling(10).mean()
    df['sma_30']  = df['close'].rolling(30).mean()
    df['sma_50']  = df['close'].rolling(50).mean()
    df['ema_12']  = df['close'].ewm(span=12).mean()
    df['ema_26']  = df['close'].ewm(span=26).mean()
    d = df['close'].diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    l = (-d.where(d < 0, 0)).rolling(14).mean()
    df['rsi']         = 100 - (100 / (1 + g / (l + 1e-10)))
    df['macd']        = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']
    df['bb_m']        = df['close'].rolling(20).mean()
    df['bb_s']        = df['close'].rolling(20).std()
    df['bb_u']        = df['bb_m'] + 2 * df['bb_s']
    df['bb_l']        = df['bb_m'] - 2 * df['bb_s']
    df['bb_position'] = ((df['close'] - df['bb_l']) / (df['bb_u'] - df['bb_l']) * 100).fillna(50)
    lo14 = df['low'].rolling(14).min()
    hi14 = df['high'].rolling(14).max()
    df['K']           = ((df['close'] - lo14) / (hi14 - lo14) * 100).fillna(50)
    df['D']           = df['K'].rolling(3).mean()
    df['obv']         = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ma20']    = df['obv'].rolling(20).mean()
    df['volatility']  = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
    hl = df['high'] - df['low']
    hc = np.abs(df['high'] - df['close'].shift())
    lc = np.abs(df['low']  - df['close'].shift())
    df['atr']              = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['price_change_5d']  = df['close'].pct_change(5)  * 100
    df['price_change_10d'] = df['close'].pct_change(10) * 100
    df['price_change_20d'] = df['close'].pct_change(20) * 100
    df['ma50_slope']       = df['sma_50'].diff(5) / df['sma_50'].shift(5) * 100
    return df.bfill().ffill()


# ── Warrant recommendation ────────────────────────────────────────────────────
def print_warrant_section(current_price, price_change_pct, signal, latest_date):
    """認購/認售權證建議 (法人觀點: 逢急跌可考慮台積電/00981A認購權證)"""
    try:
        today = datetime.strptime(latest_date, '%Y-%m-%d')
    except Exception:
        today = datetime.now()
    min_expiry = (today + timedelta(days=120)).strftime('%Y-%m-%d')

    call_low  = current_price * 0.85
    call_high = current_price * 1.00
    put_low   = current_price * 1.00
    put_high  = current_price * 1.15

    is_sharp_drop = price_change_pct <= -3.0
    is_buy  = 'BUY' in signal or '买入' in signal
    is_sell = 'SELL' in signal or '卖出' in signal

    print("\n" + "=" * 80)
    print("📋 認購/認售權證建議 (法人觀點)")
    print("=" * 80)
    print(f"00981A 現價:     NT${current_price:.2f}  (今日漲跌幅 {price_change_pct:+.2f}%)")
    print(f"建議到期日下限:  {min_expiry} 以後 (可操作天期 ≥ 120 天)")
    print()
    print("🟢 認購權證 (Call Warrant) — 看多 / 逢急跌佈局:")
    print(f"   建議履約價區間: NT${call_low:.2f} ~ NT${call_high:.2f}")
    print(f"   條件: 價內15%內 (履約價 ≥ 現價×85%，≤ 現價)  到期日 > {min_expiry}")
    if is_sharp_drop:
        print(f"   ⚡ 今日急跌 {price_change_pct:+.2f}%，符合法人建議的認購佈局時機!")
        print(f"   📌 標的: 台積電 (2330) 或 主動統一台股增長 (00981A) 相關認購權證")
    elif is_buy:
        print(f"   ✅ 買入信號確認，可考慮認購權證代替直接買入")
    else:
        print(f"   💡 目前非急跌時機，可觀察等待更佳進場點位")
    print()
    print("🔴 認售權證 (Put Warrant) — 看空 / 高點避險:")
    print(f"   建議履約價區間: NT${put_low:.2f} ~ NT${put_high:.2f}")
    print(f"   條件: 價內15%內 (履約價 ≥ 現價，≤ 現價×115%)  到期日 > {min_expiry}")
    if is_sell:
        print(f"   ⚡ 賣出信號確認，可考慮認售權證進行避險")
    else:
        print(f"   💡 認售可作為下跌保護工具備用")
    print()
    print("⚠️  注意: 選擇天期>4個月 | 價內Delta較高 | 建議流動性好的券商發行")


# ── Main signal function ──────────────────────────────────────────────────────
def get_trading_signal():
    accuracy_display = get_model_accuracy_display(CODE)
    print(f"🤖 {CODE} ({NAME}) | 準確度: {accuracy_display} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # 1. Download data
    print(f"\n📊 下載 {TICKER} 最新數據...")
    df_raw = yf.download(TICKER, period='300d', progress=False, auto_adjust=True)
    if df_raw.empty:
        print(f"❌ 無法取得 {TICKER} 數據"); return None
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.droplevel(1)
    df_raw = df_raw.rename(columns={'Close':'close','Volume':'volume',
                                    'Open':'open','High':'high','Low':'low'}).reset_index()
    print(f"✅ 取得 {len(df_raw)} 天數據")

    # 2. Build features
    df = build_features(df_raw)
    row = df.iloc[-1]
    try:
        latest_date = str(pd.to_datetime(df.iloc[-1]['Date']).date())
    except Exception:
        latest_date = datetime.now().strftime('%Y-%m-%d')

    current_price   = float(row['close'])
    prev_close      = float(df['close'].iloc[-2]) if len(df) > 1 else current_price
    price_change_pct = (current_price - prev_close) / prev_close * 100 if prev_close > 0 else 0.0
    rsi             = float(row['rsi'])
    rsi_prev        = float(df['rsi'].iloc[-2]) if len(df) > 1 else rsi
    macd            = float(row['macd'])
    macd_signal_val = float(row['macd_signal'])
    sma_10          = float(row['sma_10'])
    sma_30          = float(row['sma_30'])
    sma_50          = float(row['sma_50'])
    bb_upper        = float(row['bb_u'])
    bb_lower        = float(row['bb_l'])
    bb_position     = float(row['bb_position'])
    current_volume  = float(row['volume'])
    avg_vol_20      = float(df['volume'].tail(20).mean())
    volume_ratio    = current_volume / avg_vol_20 if avg_vol_20 > 0 else 1.0
    obv_now         = float(row['obv'])
    obv_ma          = float(row['obv_ma20'])

    print(f"\n最新數據日期: {latest_date}")
    print(f"當前價格:     NT${current_price:.2f}  ({price_change_pct:+.2f}%)")
    print(f"成交量:       {int(current_volume):,}  (量比 {volume_ratio:.2f}x)")

    # 3. Technical indicators
    print("\n" + "=" * 80)
    print("📊 技術指標分析")
    print("=" * 80)
    print(f"RSI (14):        {rsi:.2f}  {'[超買]' if rsi > 75 else '[超賣]' if rsi < 30 else '[中性]'}")
    print(f"RSI 變化:        {rsi_prev:.2f} → {rsi:.2f} ({rsi - rsi_prev:+.2f})")
    print(f"MACD:            {macd:.4f}  {'[金叉]' if macd > macd_signal_val else '[死叉]'}")
    print(f"SMA 10/30/50:    NT${sma_10:.2f} / NT${sma_30:.2f} / NT${sma_50:.2f}")
    trend_txt = '[多頭排列]' if sma_10 > sma_30 > sma_50 else '[空頭排列]' if sma_10 < sma_30 < sma_50 else '[混合]'
    print(f"均線排列:        {trend_txt}")
    print(f"布林帶位置:      {bb_position:.1f}%  (上軌 NT${bb_upper:.2f} / 下軌 NT${bb_lower:.2f})")
    print(f"OBV vs MA20:     {'[多頭]' if obv_now > obv_ma else '[空頭]'}  ({obv_now:,.0f} vs {obv_ma:,.0f})")
    print(f"量比:            {volume_ratio:.2f}x  {'[放量]' if volume_ratio > 1.5 else '[縮量]' if volume_ratio < 0.7 else '[正常]'}")

    # 4. XGBoost prediction
    print("\n" + "=" * 80)
    print("🧠 XGBoost 模型預測")
    print("=" * 80)
    xgb_prob = None
    xgb_pred = None
    try:
        xm = joblib.load(XGB_MODEL)
        feat_row = row[FEAT].values.reshape(1, -1)
        xgb_prob = float(xm.predict_proba(feat_row)[0][1])
        xgb_pred = int(xm.predict(feat_row)[0])
        print(f"今日買入機率 P(buy): {xgb_prob*100:.1f}%（P(not buy): {(1-xgb_prob)*100:.1f}%）  {'看多' if xgb_pred == 1 else '看空'}")
    except Exception as e:
        print(f"⚠️  XGBoost 載入失敗: {e}")

    # 5. PPO prediction
    print("\n" + "=" * 80)
    print("🤖 PPO 模型預測")
    print("=" * 80)
    ppo_action = 0.0
    try:
        ppo_model = PPO.load(PPO_MODEL)
        env = TradingEnv(df)
        env.i = len(df) - 1
        obs = env._obs()
        act, _ = ppo_model.predict(obs, deterministic=True)
        ppo_action = float(act[0]) if isinstance(act, np.ndarray) else float(act)
        ai_muted = should_mute_ai_signal(TICKER, threshold=52)
        if ai_muted:
            print("⚠️  AI模型準確度低於52%，已靜音AI動作 (action=0)")
            ppo_action = 0.0
        else:
            from ppo_backtest_cache import format_ppo_roi_line
            print(format_ppo_roi_line(CODE, TICKER, PPO_MODEL, df, ppo_action))
    except Exception as e:
        print(f"⚠️  PPO 載入失敗: {e}")

    # 6. Combined signal
    print("\n" + "=" * 80)
    print("🎯 綜合交易信號")
    print("=" * 80)

    # Score: XGBoost probability + PPO direction + technicals
    score = 50.0
    reasons = []
    warnings_list = []

    if xgb_prob is not None:
        score += (xgb_prob - 0.5) * 60   # ±30 pts
        if xgb_prob > 0.6:
            reasons.append(f"XGBoost 買入概率高 ({xgb_prob*100:.1f}%)")
        elif xgb_prob < 0.4:
            warnings_list.append(f"XGBoost 看空 ({xgb_prob*100:.1f}%)")

    if ppo_action > 0.1:
        score += ppo_action * 15
        reasons.append(f"PPO 建議買入 (action={ppo_action:+.2f})")
    elif ppo_action < -0.1:
        score += ppo_action * 15
        warnings_list.append(f"PPO 建議賣出 (action={ppo_action:+.2f})")

    if macd > macd_signal_val:
        score += 5; reasons.append("MACD 金叉")
    else:
        score -= 5; warnings_list.append("MACD 死叉")

    if sma_10 > sma_30:
        score += 5; reasons.append("均線多頭")
    else:
        score -= 5; warnings_list.append("均線空頭")

    if rsi < 35:
        score += 8; reasons.append(f"RSI 超賣 ({rsi:.1f})")
    elif rsi > 75:
        score -= 8; warnings_list.append(f"RSI 超買 ({rsi:.1f})")

    if obv_now > obv_ma:
        score += 4; reasons.append("OBV 多頭")
    else:
        score -= 4; warnings_list.append("OBV 空頭")

    score = max(0, min(100, score))

    if score >= 65:
        signal = "買入 (BUY)"; emoji = "🟢"
    elif score <= 35:
        signal = "賣出 (SELL)"; emoji = "🔴"
    else:
        signal = "持有 (HOLD)"; emoji = "🟡"

    print(f"\n{emoji} 信號: {signal}")
    print(f"   綜合評分: {score:.0f} / 100")
    if xgb_prob is not None:
        print(f"   今日買入機率 P(buy): {xgb_prob*100:.1f}%（P(not buy): {(1-xgb_prob)*100:.1f}%）")
    print(f"   PPO 動作值: {ppo_action:+.4f}")

    if warnings_list:
        print(f"\n   ⚠️  警告:")
        for w in warnings_list:
            print(f"      • {w}")

    if reasons:
        print(f"\n   📌 理由:")
        for i, r in enumerate(reasons, 1):
            print(f"      {i}. {r}")

    print(f"\n   💡 操作建議:")
    if score >= 65:
        print(f"      • 可考慮買入，建議分批進場")
        print(f"      • 參考買入區間: NT${current_price * 0.995:.2f} ~ NT${current_price:.2f}")
        print(f"      • 止損設置: NT${current_price * 0.95:.2f} (-5%)")
    elif score <= 35:
        print(f"      • 技術面偏弱，建議觀望或減倉")
        print(f"      • 關注支撐位: NT${bb_lower:.2f}")
    else:
        print(f"      • 觀望為主，等待更明確信號")
        print(f"      • 支撐位: NT${bb_lower:.2f}  壓力位: NT${bb_upper:.2f}")

    # 7. Warrant recommendation
    print_warrant_section(current_price, price_change_pct, signal, latest_date)

    # 8. Summary
    print("\n" + "=" * 80)
    print(f"\n📱 快速摘要:")
    print(f"   ETF: {TICKER} ({NAME})")
    print(f"   日期: {latest_date}")
    print(f"   價格: NT${current_price:.2f}  ({price_change_pct:+.2f}%)")
    print(f"   信號: {signal}")
    print(f"   評分: {score:.0f}/100")
    print(f"   {get_model_accuracy_display(CODE)}")

    return {
        'date': latest_date, 'symbol': TICKER, 'name': NAME,
        'current_price': current_price, 'price_change_pct': price_change_pct,
        'signal': signal, 'score': score,
        'xgb_prob': xgb_prob, 'ppo_action': ppo_action,
        'rsi': rsi, 'macd': macd,
    }


# ── News ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    result = get_trading_signal()
    if not result:
        print("\n❌ 信號生成失敗")
    print('\n' + '=' * 80)
    print(f'🌐 {CODE} {NAME} 即時新聞  (Tavily REST API)')
    print('=' * 80)
    print_tavily_news_tw(CODE, NAME, max_results=5)
