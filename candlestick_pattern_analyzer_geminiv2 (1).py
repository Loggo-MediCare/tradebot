# ================================================
# 蠟燭圖型態嚴格審查系統 (CIO 重構修正版)
# ================================================
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import warnings

# 環境設定
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def run_strict_analysis(symbol="2313.TW"):
    print(f"【執行】正在以 CIO 嚴格標準審查 {symbol} 的 K 線體質...\n")

    # 1. 下載資料
    try:
        df = yf.download(symbol, period="6mo", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        for col in df.columns:
            df[col] = df[col].astype(float)
    except Exception as e:
        print(f"❌ 下載錯誤: {e}")
        return

    # 2. 計算技術指標 (趨勢與動能過濾)
    # 計算 MA
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # 計算 乖離率 (Bias)
    df['Bias10'] = (df['Close'] - df['MA10']) / df['MA10'] * 100
    
    # 計算 RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 計算成交量均線 (20日)
    df['Vol_Avg20'] = df['Volume'].rolling(window=20).mean()

    # 3. 定義 K 線細節
    df['body_size'] = abs(df['Close'] - df['Open'])
    df['total_range'] = df['High'] - df['Low']
    df['upper_shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    # 比例計算
    df['ratio_lower'] = df['lower_shadow'] / (df['body_size'] + 0.001)
    df['ratio_upper'] = df['upper_shadow'] / (df['body_size'] + 0.001)
    df['body_percent'] = df['body_size'] / (df['total_range'] + 0.001)

    # 4. 嚴格型態判定
    bullish_signals = []
    bearish_signals = []
    
    check_days = 10
    recent_df = df.tail(check_days)
    
    print("-" * 80)
    print(f"{'日期':<12} | {'收盤':<8} | {'型態判定':<20}")
    print("-" * 80)

    for i in range(len(recent_df)):
        current_idx = -check_days + i
        row = recent_df.iloc[i]
        prev_row = df.iloc[df.index.get_loc(recent_df.index[i]) - 1]
        date_str = recent_df.index[i].strftime('%Y-%m-%d')
        
        findings = []
        
        # --- A. 槌子線 (底部訊號) ---
        # 定義：低檔(Bias < 0)、下影線 > 2倍實體、上影線極小、實體小
        if row['Bias10'] < 0 and row['ratio_lower'] >= 2.0 and row['ratio_upper'] < 0.6 and row['body_percent'] < 0.4:
            findings.append("🔨 槌子線")
            bullish_signals.append((date_str, "槌子線", row['Close']))

        # --- B. 吊人線 (高檔警訊 - 嚴格版) ---
        # 修正：下影線 > 2.5倍、上影線必須小於實體一半、處於高位(RSI > 60 或 Bias > 2)
        is_uptrend = (row['RSI'] > 60) or (row['Bias10'] > 2)
        if is_uptrend and row['ratio_lower'] >= 2.5 and row['ratio_upper'] < 0.5 and row['body_percent'] < 0.3:
            findings.append("😵 吊人線")
            bearish_signals.append((date_str, "吊人線", row['Close']))

        # --- C. 缺口判定 (修復 Bug) ---
        # 上升缺口：今日最低 > 昨日最高
        if row['Low'] > prev_row['High'] * 1.002:
            findings.append("🚀 上升缺口")
            bullish_signals.append((date_str, "上升缺口", row['Close']))
        # 下降缺口：今日最高 < 昨日最低
        elif row['High'] < prev_row['Low'] * 0.998:
            findings.append("📉 下降缺口")
            bearish_signals.append((date_str, "下降缺口", row['Close']))

        # --- D. 十字線 ---
        if row['body_percent'] < 0.1:
            findings.append("✨ 十字線")

        # 顯示每日結果
        pattern_str = ", ".join(findings) if findings else "😴 盤整/無特定型態"
        print(f"{date_str:<12} | {row['Close']:<10.2f} | {pattern_str}")

    # 5. 總結摘要
    print("\n" + "=" * 80)
    print("【總結摘要】型態統計與建議")
    print("=" * 80)
    
    print(f"📈 看漲訊號 ({len(bullish_signals)} 個)：")
    for date, pat, price in bullish_signals:
        print(f"  • {date}: {pat} @ {price:.2f}")

    print(f"\n📉 看跌訊號 ({len(bearish_signals)} 個)：")
    for date, pat, price in bearish_signals:
        print(f"  • {date}: {pat} @ {price:.2f}")

    # 投資建議邏輯
    net_score = len(bullish_signals) - len(bearish_signals)
    print("\n🎯 最終建議：")
    if net_score > 0:
        print("🟢 多頭動能佔優，建議回測支撐不破時佈局。")
    elif net_score < 0:
        print("🔴 高檔壓力浮現，建議分批獲利了結或設置嚴格止損。")
    else:
        print("⚖️ 多空力道均勻，建議觀察後續放量方向。")
    print("=" * 80)

# 執行分析
if __name__ == "__main__":
    run_strict_analysis("SNDK")