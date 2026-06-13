"""
台股 2882 (國泰金) AI 交易信号生成器
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
from datetime import datetime

from model_accuracy_tracker import get_model_accuracy_display, should_mute_ai_signal
from tw_news_tracker import print_tavily_news_tw

CODE   = '2882'
TICKER = '2882.TW'
NAME   = '國泰金'

XGB_MODEL = 'xgb_2882_tw_model.pkl'
PPO_MODEL = 'ppo_2882_tw_improved'

FEAT = ['rsi','macd','macd_signal','macd_hist','bb_position','K','D','obv','obv_ma20',
        'sma_10','sma_30','sma_50','volatility','atr',
        'price_change_5d','price_change_10d','price_change_20d','ma50_slope']


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


def get_trading_signal():
    accuracy_display = get_model_accuracy_display(CODE)
    print(f"🤖 {CODE} ({NAME}) | 準確度: {accuracy_display} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # 1. Download data
    print(f"\n📊 下載 {TICKER} 最新數據...")
    df_raw = yf.download(TICKER, period='300d', progress=False, auto_adjust=True)
    if df_raw.empty:
        print(f"❌ 無法取得數據"); return None
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.droplevel(1)
    df_raw = df_raw.rename(columns={'Close':'close','Volume':'volume',
                                    'Open':'open','High':'high','Low':'low'}).reset_index()
    print(f"✅ 取得 {len(df_raw)} 天數據")

    # Analyst info
    try:
        info = yf.Ticker(TICKER).info
        target_price  = info.get('targetMeanPrice')
        target_high   = info.get('targetHighPrice')
        rec_mean      = info.get('recommendationMean')
        rec_key       = info.get('recommendationKey', '')
        num_analysts  = info.get('numberOfAnalystOpinions', 0)
        if target_price and num_analysts > 0:
            print(f"   📊 分析師目標價: NT${target_price:.2f} (平均) / NT${target_high:.2f} (最高)")
            print(f"   📊 分析師評級:   {rec_key} ({rec_mean:.1f}/5, {num_analysts}位)")
    except Exception:
        target_price = target_high = rec_mean = None

    # 2. Build features
    df = build_features(df_raw)
    row = df.iloc[-1]
    try:
        latest_date = str(pd.to_datetime(df.iloc[-1]['Date']).date())
    except Exception:
        latest_date = datetime.now().strftime('%Y-%m-%d')

    current_price    = float(row['close'])
    prev_close       = float(df['close'].iloc[-2]) if len(df) > 1 else current_price
    price_change_pct = (current_price - prev_close) / prev_close * 100 if prev_close > 0 else 0.0
    rsi              = float(row['rsi'])
    rsi_prev         = float(df['rsi'].iloc[-2]) if len(df) > 1 else rsi
    macd             = float(row['macd'])
    macd_sig         = float(row['macd_signal'])
    sma_10           = float(row['sma_10'])
    sma_30           = float(row['sma_30'])
    sma_50           = float(row['sma_50'])
    bb_upper         = float(row['bb_u'])
    bb_lower         = float(row['bb_l'])
    bb_position      = float(row['bb_position'])
    current_volume   = float(row['volume'])
    avg_vol_20       = float(df['volume'].tail(20).mean())
    volume_ratio     = current_volume / avg_vol_20 if avg_vol_20 > 0 else 1.0
    obv_now          = float(row['obv'])
    obv_ma           = float(row['obv_ma20'])
    ma50_slope       = float(row['ma50_slope'])

    print(f"\n最新數據日期: {latest_date}")
    print(f"當前價格:     NT${current_price:.2f}  ({price_change_pct:+.2f}%)")
    print(f"成交量:       {int(current_volume):,}  (量比 {volume_ratio:.2f}x)")

    # 3. Technical analysis
    print("\n" + "=" * 80)
    print("📊 技術指標分析")
    print("=" * 80)
    print(f"RSI (14):        {rsi:.2f}  {'[超買]' if rsi > 75 else '[超賣]' if rsi < 30 else '[中性]'}")
    print(f"RSI 變化:        {rsi_prev:.2f} → {rsi:.2f}  ({rsi - rsi_prev:+.2f})")
    print(f"MACD:            {macd:.4f}  {'[金叉]' if macd > macd_sig else '[死叉]'}")
    print(f"MACD Signal:     {macd_sig:.4f}")
    print(f"SMA 10/30/50:    NT${sma_10:.2f} / NT${sma_30:.2f} / NT${sma_50:.2f}")
    if sma_10 > sma_30 > sma_50:
        ma_trend = '🟢 多頭排列'
    elif sma_10 < sma_30 < sma_50:
        ma_trend = '🔴 空頭排列'
    else:
        ma_trend = '🟡 混合排列'
    print(f"均線排列:        {ma_trend}")
    print(f"MA50 斜率:       {ma50_slope:+.4f}%  {'[上升]' if ma50_slope > 0 else '[下降]'}")
    print(f"布林帶位置:      {bb_position:.1f}%  (上軌 NT${bb_upper:.2f} / 下軌 NT${bb_lower:.2f})")
    print(f"OBV vs MA20:     {'[多頭]' if obv_now > obv_ma else '[空頭]'}")
    print(f"量比:            {volume_ratio:.2f}x  {'[放量]' if volume_ratio > 1.5 else '[縮量]' if volume_ratio < 0.7 else '[正常]'}")
    print(f"今日漲跌幅:      {price_change_pct:+.2f}%")

    # Analyst target vs price
    if target_price:
        upside = (target_price - current_price) / current_price * 100
        print(f"\n分析師目標價:    NT${target_price:.2f}  (距現價 {upside:+.1f}%)")

    # 4. XGBoost
    print("\n" + "=" * 80)
    print("🧠 XGBoost 模型  (準確度 70.09%)")
    print("=" * 80)
    xgb_prob = xgb_pred = None
    try:
        xm = joblib.load(XGB_MODEL)
        feat_row = row[FEAT].values.reshape(1, -1)
        xgb_prob = float(xm.predict_proba(feat_row)[0][1])
        xgb_pred = int(xm.predict(feat_row)[0])
        bar = '█' * int(xgb_prob * 20) + '░' * (20 - int(xgb_prob * 20))
        print(f"今日買入機率 P(buy): {xgb_prob*100:.1f}%（P(not buy): {(1-xgb_prob)*100:.1f}%）  [{bar}]  {'看多 📈' if xgb_pred == 1 else '看空 📉'}")
    except Exception as e:
        print(f"⚠️  XGBoost 失敗: {e}")

    # 5. PPO
    print("\n" + "=" * 80)
    print("🤖 PPO 模型  (準確度 69.95%)")
    print("=" * 80)
    ppo_action = 0.0
    try:
        ppo_model = PPO.load(PPO_MODEL)
        env = TradingEnv(df)
        env.i = len(df) - 1
        obs = env._obs()
        act, _ = ppo_model.predict(obs, deterministic=True)
        ppo_action = float(act[0]) if isinstance(act, np.ndarray) else float(act)
        if should_mute_ai_signal(TICKER, threshold=52):
            print("⚠️  準確度低於52%門檻，已靜音 (action=0)")
            ppo_action = 0.0
        else:
            from ppo_backtest_cache import format_ppo_roi_line
            print(format_ppo_roi_line(CODE, TICKER, PPO_MODEL, df, ppo_action))
    except Exception as e:
        print(f"⚠️  PPO 失敗: {e}")

    # 6. Combined signal
    print("\n" + "=" * 80)
    print("🎯 綜合交易信號")
    print("=" * 80)

    score = 50.0
    reasons, warnings_list = [], []

    if xgb_prob is not None:
        score += (xgb_prob - 0.5) * 60
        if xgb_prob > 0.6:
            reasons.append(f"XGBoost 買入概率 {xgb_prob*100:.1f}% (看多)")
        elif xgb_prob < 0.4:
            warnings_list.append(f"XGBoost 看空 ({xgb_prob*100:.1f}%)")

    if ppo_action > 0.1:
        score += ppo_action * 15
        reasons.append(f"PPO 建議買入 ({ppo_action:+.2f})")
    elif ppo_action < -0.1:
        score += ppo_action * 15
        warnings_list.append(f"PPO 建議賣出 ({ppo_action:+.2f})")

    if macd > macd_sig:
        score += 5; reasons.append("MACD 金叉")
    else:
        score -= 5; warnings_list.append("MACD 死叉")

    if sma_10 > sma_30:
        score += 5; reasons.append("均線多頭 (SMA10 > SMA30)")
    else:
        score -= 5; warnings_list.append("均線空頭")

    if rsi < 35:
        score += 8; reasons.append(f"RSI 超賣 ({rsi:.1f}) — 反彈機會")
    elif rsi > 75:
        score -= 8; warnings_list.append(f"RSI 超買 ({rsi:.1f})")

    if ma50_slope > 0.05:
        score += 4; reasons.append(f"MA50 上升趨勢 (+{ma50_slope:.2f}%)")
    elif ma50_slope < -0.05:
        score -= 4; warnings_list.append(f"MA50 下降趨勢 ({ma50_slope:.2f}%)")

    if obv_now > obv_ma:
        score += 3; reasons.append("OBV 多頭 (資金流入)")
    else:
        score -= 3

    if target_price and current_price > 0:
        upside = (target_price - current_price) / current_price * 100
        if upside > 15:
            score += 5; reasons.append(f"目標價上漲空間 {upside:.1f}%")
        elif upside < -5:
            score -= 5; warnings_list.append(f"股價超越目標價 ({upside:.1f}%)")

    if bb_position < 20:
        score += 5; reasons.append(f"接近布林帶下軌 ({bb_position:.0f}%) — 超跌支撐")
    elif bb_position > 85:
        score -= 5; warnings_list.append(f"接近布林帶上軌 ({bb_position:.0f}%)")

    score = max(0, min(100, score))

    if score >= 65:
        signal = "買入 (BUY)"; emoji = "🟢"
    elif score <= 35:
        signal = "賣出 (SELL)"; emoji = "🔴"
    else:
        signal = "持有 (HOLD)"; emoji = "🟡"

    print(f"\n{emoji} 信號: {signal}")
    print(f"   綜合評分: {score:.0f} / 100")
    print(f"   今日買入機率 P(buy): {xgb_prob*100:.1f}%（P(not buy): {(1-xgb_prob)*100:.1f}%）" if xgb_prob is not None else "")
    print(f"   PPO 動作值:       {ppo_action:+.4f}")

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
        if target_price:
            print(f"      • 目標價: NT${target_price:.2f}  (分析師平均)")
    elif score <= 35:
        print(f"      • 技術面偏弱，建議觀望或減倉")
        print(f"      • 關注支撐位: NT${bb_lower:.2f}")
    else:
        print(f"      • 觀望為主，等待更明確信號")
        print(f"      • 支撐位: NT${bb_lower:.2f}  壓力位: NT${bb_upper:.2f}")

    # 7. Summary
    print("\n" + "=" * 80)
    print(f"\n📱 快速摘要:")
    print(f"   股票: {TICKER} ({NAME})")
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


if __name__ == '__main__':
    result = get_trading_signal()
    if not result:
        print("\n❌ 信號生成失敗")
    print('\n' + '=' * 80)
    print(f'🌐 {CODE} {NAME} 即時新聞  (Tavily REST API)')
    print('=' * 80)
    print_tavily_news_tw(CODE, NAME, max_results=5)
