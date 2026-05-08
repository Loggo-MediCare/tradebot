# ================================================
# 蠟燭圖型態嚴格審查系統 (Enhanced with Pattern Observations)
# ================================================
import subprocess
import sys
import os

# 1. 安裝環境
print("【步驟1】正在啟動「嚴格審查模式」... (參照 HAMMER_2.pdf)")
try:
    subprocess.run(['pip', 'install', 'yfinance', 'pandas', 'numpy', 'matplotlib'], check=True, capture_output=True)
except:
    pass

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
print("✓ 審查員就位！\n")

# ================================================
# 2. 設定與下載
# ================================================
symbol = "SNDK"  # <--- 阿姨請在這裡改成妳要檢查的股票
print(f"【步驟2】正在嚴格檢查 {symbol} 的 K 線體質...")

try:
    df = yf.download(symbol, period="6mo", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    for col in df.columns:
        df[col] = df[col].astype(float)

    current_price = df['Close'].iloc[-1]
    print(f"   目前股價：{current_price:.2f}\n")

    # ================================================
    # 3. 書本嚴格定義邏輯 (Strict Logic from PDF)
    # ================================================

    # A. 計算實體與影線 (Body & Shadows)
    # [cite: 10010] Real body is range between open and close
    df['Body_Size'] = abs(df['Close'] - df['Open'])
    df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['Total_Range'] = df['High'] - df['Low']

    # B. 計算比例 (The Ratio)
    # Shadow should be about 3 times as long as the body
    # 我們設定嚴格門檻為 2.5倍 (接近3倍)，寬鬆門檻為 2倍
    df['Ratio_Lower'] = df['Lower_Shadow'] / (df['Body_Size'] + 0.001)
    df['Ratio_Upper'] = df['Upper_Shadow'] / (df['Body_Size'] + 0.001)

    # C. 定義趨勢 (Trend Context)
    # [cite: 10041] Context is crucial. Need to know previous bars.
    # 使用 10日均線斜率來判斷短期趨勢
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Trend'] = np.where(df['Close'] > df['MA10'], '上升', '下跌')

    # D. 計算漲跌幅
    df['Change_Pct'] = df['Close'].pct_change() * 100

    # E. 判斷K線顏色
    df['Color'] = np.where(df['Close'] >= df['Open'], '紅', '黑')

    # ================================================
    # 4. 綜合診斷 (最近 10 天) + 型態觀察統計
    # ================================================
    print("【步驟3】書本「嚴格標準」診斷報告 (最近 10 天)")
    print("=" * 80)

    recent_days = df.iloc[-10:]

    # 統計變數
    hammer_count = 0
    hanging_man_count = 0
    doji_count = 0
    bullish_signals = []
    bearish_signals = []

    for date, row in recent_days.iterrows():
        date_str = date.strftime('%Y-%m-%d')
        close_price = row['Close']
        open_price = row['Open']
        trend = row['Trend']
        color = row['Color']
        change = row['Change_Pct']

        print(f"\n📅 日期：{date_str} | 收盤：{close_price:.2f} | {color}K | 漲跌：{change:+.2f}% | 趨勢：{trend}")

        signals = []
        pattern_type = None

        # --- 1. 嚴格槌子與吊人 (Hammer & Hanging Man) ---
        # 條件：下影線長，上影線極短，小實體
        is_long_lower = row['Ratio_Lower'] >= 2.0  # 至少2倍
        is_strict_lower = row['Ratio_Lower'] >= 3.0  # 書本要求的3倍
        is_small_upper = row['Upper_Shadow'] < row['Body_Size']  # 上影線要很短
        is_small_body = row['Body_Size'] < (row['Total_Range'] * 0.3)  # 實體小於全日範圍30%

        if is_long_lower and is_small_body:
            lower_ratio = row['Ratio_Lower']

            if trend == '下跌':
                # [cite: 9686] Hammer at end of downtrend
                quality = "🔥書本級(3倍長腳)" if is_strict_lower else "⚠️普通級(2倍長腳)"
                signal_text = f"🔨 [槌子線] 底部止跌訊號 ({quality})"
                signal_text += f"\n      📊 觀察重點：下影線是實體的 {lower_ratio:.1f} 倍"
                signal_text += f"\n      📊 位置：出現在{trend}趨勢末端"
                signal_text += f"\n      📊 確認：隔日若收紅上漲，可確認反轉"
                signals.append(signal_text)
                hammer_count += 1
                bullish_signals.append((date_str, '槌子線', close_price))
                pattern_type = 'hammer'

            elif trend == '上升':
                # [cite: 9698] Hanging man at end of uptrend
                signal_text = f"😵 [吊人線] 高檔懸空警訊"
                signal_text += f"\n      📊 觀察重點：下影線是實體的 {lower_ratio:.1f} 倍"
                signal_text += f"\n      📊 位置：出現在{trend}趨勢高點"
                signal_text += f"\n      📊 確認：隔日若收黑下跌，賣壓確認"
                signal_text += f"\n      ⚠️  警告：這是頭部訊號，建議減碼"
                signals.append(signal_text)
                hanging_man_count += 1
                bearish_signals.append((date_str, '吊人線', close_price))
                pattern_type = 'hanging_man'

        # --- 2. 十字線 (Doji) - 增強觀察 ---
        # [cite: 10045] Open and close are same or nearly same
        doji_threshold = close_price * 0.002  # 0.2% 容差
        if row['Body_Size'] <= doji_threshold:
            doji_type = "標準十字"

            # 細分十字線類型
            if row['Upper_Shadow'] > row['Lower_Shadow'] * 2:
                doji_type = "墓碑十字 (Gravestone Doji)"
                implication = "看跌訊號，買方失守"
                bearish_signals.append((date_str, doji_type, close_price))
            elif row['Lower_Shadow'] > row['Upper_Shadow'] * 2:
                doji_type = "蜻蜓十字 (Dragonfly Doji)"
                implication = "看漲訊號，賣方失守"
                bullish_signals.append((date_str, doji_type, close_price))
            else:
                doji_type = "標準十字 (Classic Doji)"
                implication = "多空平衡，準備變盤"

            signal_text = f"✨ [{doji_type}] {implication}"
            signal_text += f"\n      📊 觀察重點：開盤價 {open_price:.2f} ≈ 收盤價 {close_price:.2f}"
            signal_text += f"\n      📊 上影線：{row['Upper_Shadow']:.2f} | 下影線：{row['Lower_Shadow']:.2f}"
            signal_text += f"\n      📊 意義：多空力量平衡，市場猶豫不決"
            signal_text += f"\n      📊 後市：需觀察隔日方向確認"
            signals.append(signal_text)
            doji_count += 1
            pattern_type = 'doji'

        # --- 3. 吞噬型態 (Engulfing) - 手動判斷 ---
        day_idx = df.index.get_loc(date)
        if day_idx > 0:
            prev_row = df.iloc[day_idx - 1]
            prev_body_top = max(prev_row['Open'], prev_row['Close'])
            prev_body_bottom = min(prev_row['Open'], prev_row['Close'])
            curr_body_top = max(row['Open'], row['Close'])
            curr_body_bottom = min(row['Open'], row['Close'])

            # 多頭吞噬：今天紅K完全包住昨天黑K
            if (color == '紅' and prev_row['Color'] == '黑' and
                curr_body_bottom < prev_body_bottom and curr_body_top > prev_body_top):
                signal_text = "🐉 [多頭吞噬] 大紅棒吃掉昨天的黑棒"
                signal_text += f"\n      📊 觀察重點：今日實體完全包覆昨日"
                signal_text += f"\n      📊 昨日：{prev_row['Close']:.2f} | 今日：{close_price:.2f}"
                signal_text += f"\n      📊 意義：買盤強勢反攻，趨勢可能反轉"
                if trend == '下跌':
                    signal_text += "\n      ✓ 書本認證投資等級！底部反轉"
                signals.append(signal_text)
                bullish_signals.append((date_str, '多頭吞噬', close_price))

            # 空頭吞噬：今天黑K完全包住昨天紅K
            elif (color == '黑' and prev_row['Color'] == '紅' and
                  curr_body_bottom < prev_body_bottom and curr_body_top > prev_body_top):
                signal_text = "🐻 [空頭吞噬] 大黑棒吃掉昨天的紅棒"
                signal_text += f"\n      📊 觀察重點：今日實體完全包覆昨日"
                signal_text += f"\n      📊 昨日：{prev_row['Close']:.2f} | 今日：{close_price:.2f}"
                signal_text += f"\n      📊 意義：賣壓強勢反撲，趨勢可能反轉"
                if trend == '上升':
                    signal_text += "\n      ✓ 書本認證投資等級！頂部反轉"
                signals.append(signal_text)
                bearish_signals.append((date_str, '空頭吞噬', close_price))

        # --- 4. 缺口 (Windows) ---
        if day_idx > 0:
            prev_row = df.iloc[day_idx - 1]
            prev_high = prev_row['High']
            prev_low = prev_row['Low']

            if row['Low'] > prev_high:
                gap_size = row['Low'] - prev_high
                signal_text = f"🚀 [上升缺口 (Window)] 強力看漲"
                signal_text += f"\n      📊 觀察重點：缺口大小 {gap_size:.2f}"
                signal_text += f"\n      📊 意義：買盤強勁，不應回補缺口"
                signal_text += f"\n      📊 確認：若回補缺口則趨勢轉弱"
                signals.append(signal_text)
                bullish_signals.append((date_str, '上升缺口', close_price))

            elif row['High'] < prev_low:
                gap_size = prev_low - row['High']
                signal_text = f"📉 [下降缺口 (Window)] 強力看跌"
                signal_text += f"\n      📊 觀察重點：缺口大小 {gap_size:.2f}"
                signal_text += f"\n      📊 意義：賣壓沈重，持續下跌"
                signal_text += f"\n      📊 確認：若回補缺口則賣壓減輕"
                signals.append(signal_text)
                bearish_signals.append((date_str, '下降缺口', close_price))

        # 顯示結果
        if signals:
            for s in signals:
                print(f"   👉 {s}")
        else:
            print("   😴 (未符合書本嚴格定義)")

        print("-" * 80)

    # ================================================
    # 5. 統計摘要與投資建議
    # ================================================
    print("\n" + "=" * 80)
    print("【步驟4】型態統計摘要 (最近 10 天)")
    print("=" * 80)
    print(f"\n📊 型態出現次數：")
    print(f"   🔨 槌子線：{hammer_count} 次 (看漲訊號)")
    print(f"   😵 吊人線：{hanging_man_count} 次 (看跌訊號)")
    print(f"   ✨ 十字線：{doji_count} 次 (變盤訊號)")

    print(f"\n💡 訊號統計：")
    print(f"   📈 看漲訊號：{len(bullish_signals)} 個")
    if bullish_signals:
        for date, pattern, price in bullish_signals[-3:]:  # 最近3個
            print(f"      • {date}: {pattern} @ {price:.2f}")

    print(f"   📉 看跌訊號：{len(bearish_signals)} 個")
    if bearish_signals:
        for date, pattern, price in bearish_signals[-3:]:  # 最近3個
            print(f"      • {date}: {pattern} @ {price:.2f}")

    # 綜合判斷
    print(f"\n🎯 投資建議：")
    net_signal = len(bullish_signals) - len(bearish_signals)
    if net_signal > 2:
        print("   ✅ 多頭訊號較強，可考慮逢低布局")
    elif net_signal < -2:
        print("   ❌ 空頭訊號較強，建議減碼或觀望")
    else:
        print("   ⚖️  多空交戰，建議觀察後續確認訊號")

    print("\n" + "=" * 80)
    print("✅ 審查完畢！請依據型態觀察做好風險管理！")
    print("=" * 80)

except Exception as e:
    print(f"❌ 發生錯誤: {e}")
    import traceback
    traceback.print_exc()
