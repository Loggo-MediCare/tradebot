"""


3028.TW (增你強) PPO 交易信號生成器


"""


TICKER = '3028.TW'


MODEL_FILE = 'ppo_3028_tw_improved'


NAME = '增你強'


import os


os.chdir(os.path.dirname(os.path.abspath(__file__)))


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import sys, io


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


import numpy as np, pandas as pd, yfinance as yf


from datetime import datetime


from stable_baselines3 import PPO


from model_accuracy_tracker import get_model_accuracy_display


from technical_indicator_score import calculate_indicator_score
from tw_news_tracker import print_tavily_news_tw


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


    lo14 = df['low'].rolling(14).min()


    hi14 = df['high'].rolling(14).max()


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


    print("🤖 " + TICKER + " (" + NAME + ") | 準確度: " + accuracy_display +


          " | " + datetime.now().strftime('%Y-%m-%d %H:%M'))


    print("=" * 80)


    try:


        model = PPO.load(MODEL_FILE)


        print("✅ PPO模型加載成功: " + MODEL_FILE + ".zip")


    except Exception as e:


        print("❌ 模型加載失敗: " + str(e)); return None


    print("\n📊 下載 " + TICKER + " 最新數據...")


    try:


        df = yf.download(TICKER, period='1y', progress=False)


        if df.empty: print("❌ 無法獲取數據"); return None


        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)


        df = df.rename(columns={'Close':'close','Volume':'volume','Open':'open','High':'high','Low':'low'}).reset_index()


        print("✅ 成功下載 " + str(len(df)) + " 天數據")


    except Exception as e:


        print("❌ 數據下載失敗: " + str(e)); return None


    df = add_technical_indicators(df)


    latest = df.iloc[-1]


    current_price = float(latest['close'])


    balance = 10000.0


    obs = np.array([


        0.0, balance, current_price,


        float(latest['sma_10']), float(latest['sma_30']), float(latest['sma_50']),


        float(latest['rsi']),    float(latest['macd']),   float(latest['macd_signal']),


        float(latest['bb_upper']), float(latest['bb_lower']),


        float(latest['volume']),


        0.0, 0.0, 1.0,


    ], dtype=np.float32)


    print("\n🧠 PPO AI 模型分析中...")


    action, _ = model.predict(obs, deterministic=True)


    action_val = float(action[0])


    buy_prob = action_val * 50 + 50


    pred = 1 if action_val > 0.1 else 0


    print(f"PPO動作值: {action_val:+.4f}  (買入信心: {buy_prob:.1f}%)")


    print("\n" + "=" * 80 + "\n📊 技術指標\n" + "=" * 80)


    rsi = float(latest['rsi']); macd = float(latest['macd']); ms = float(latest['macd_signal'])


    s10 = float(latest['sma_10']); s30 = float(latest['sma_30'])


    print(f"當前價格: NT${current_price:.2f}")


    print(f"RSI: {rsi:.2f}  " + ("[超買]" if rsi > 70 else "[超賣]" if rsi < 30 else "[中性]"))


    print(f"MACD: {macd:.4f}  " + ("[金叉]" if macd > ms else "[死叉]"))


    print(f"均線: SMA10={s10:.2f}, SMA30={s30:.2f}  " + ("[多頭]" if s10 > s30 else "[空頭]"))


    print(f"布林帶位置: {float(latest['bb_position']):.1f}%")


    print(f"KD: K={float(latest['K']):.1f}, D={float(latest['D']):.1f}")


    # ── Technical indicator scoring ──────────────────────────────────────


    indicator_score, ind_reasons, ind_warnings = calculate_indicator_score(latest, df)


    print(f"\n📐 技術指標評分: {indicator_score:+.0f} 分")


    for r in ind_reasons:  print(f"   ✅ {r}")


    for w in ind_warnings: print(f"   ⚠️  {w}")


    pattern_score = 0.0; gap_up_count = 0; bearish_count = 0


    try:


        patterns = analyze_candlestick_patterns(df, days=5)


        print(format_pattern_output(patterns))


        pattern_score = get_pattern_score_adjustment(patterns)


        print("\n型態評分調整: " + f"{pattern_score:+.1f}" + " 分")


        gap_up_count  = patterns.get('bullish_signals', []).count('上升缺口')


        bearish_count = len(patterns.get('bearish_signals', []))


    except Exception:


        pass


    eff_prob = min(100.0, max(0.0, buy_prob + pattern_score * 2 + indicator_score * 2))


    strong_pattern = (gap_up_count >= 3) or (pattern_score >= 8)


    mod_pattern    = (gap_up_count >= 2) or (pattern_score >= 6)


    print("\n" + "=" * 80 + "\n🎯 交易信號\n" + "=" * 80)


    if (strong_pattern or (mod_pattern and eff_prob >= 60)) and bearish_count == 0:


        label = "← 強力型態主導" if strong_pattern else "← 型態+AI雙重確認"


        print(f"🟢 買入信號 (BUY) {label}")


        print(f"   ⚡ {gap_up_count} 個上升缺口 / 型態分 {pattern_score:+.1f} → 覆蓋 AI 低信心")


        print(f"   PPO動作: {action_val:+.4f}  |  型態調整後有效信心度: {eff_prob:.1f}%")


        print(f"   建議買入價格: NT${current_price*0.995:.2f} - NT${current_price*1.005:.2f}")


    elif (strong_pattern and bearish_count > 0) or (mod_pattern and bearish_count == 0):


        print("🟡 弱買入信號 (WEAK BUY) ← 型態看多")


        if gap_up_count > 0:


            print(f"   📈 {gap_up_count} 個上升缺口 / 型態分 {pattern_score:+.1f}")


        if bearish_count > 0:


            print(f"   ⚠  存在 {bearish_count} 個空頭型態，謹慎操作")


        print(f"   PPO動作: {action_val:+.4f}  |  型態調整後有效信心度: {eff_prob:.1f}%")


    elif pred == 1 and eff_prob >= 60:


        print("🟢 買入信號 (BUY)")


        print(f"   信心度: {eff_prob:.1f}%" +


              (f"  (PPO {buy_prob:.1f}% + 型態 {pattern_score:+.1f}分)" if pattern_score != 0 else ""))


        print(f"   建議買入價格: NT${current_price*0.995:.2f} - NT${current_price*1.005:.2f}")


    elif (pred == 1 and eff_prob >= 52) or eff_prob >= 55:


        print("🟡 弱買入信號 (WEAK BUY)")


        print(f"   信心度: {eff_prob:.1f}%" +


              (f"  (PPO {buy_prob:.1f}% + 型態 {pattern_score:+.1f}分)" if pattern_score != 0 else ""))


    else:


        print("🔴 不建議買入 (HOLD/WAIT)")


        print(f"   買入信心度不足: {buy_prob:.1f}%" +


              (f"  |  型態調整後: {eff_prob:.1f}%" if pattern_score != 0 else ""))


    print("=" * 80)


    return {'ticker': TICKER, 'price': current_price, 'prediction': pred,


            'buy_probability': eff_prob, 'rsi': rsi, 'macd': macd}


if __name__ == "__main__":


    get_trading_signal()



# ── Tavily 即時新聞 ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\n' + '=' * 80)
    print('🌐 3028 增你強 即時新聞  (Tavily REST API)')
    print('=' * 80)
    print_tavily_news_tw('3028', '增你強', max_results=5)