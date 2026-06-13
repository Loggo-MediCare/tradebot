"""
MA50_slope 斜率分析模組
用於計算和解讀MA50的趨勢方向
"""

import pandas as pd
import numpy as np


def calculate_ma50_slope(prices_or_df, window=50, slope_period=5, current_rsi=None, profit_positive=None):
    """
    計算MA50斜率

    Parameters:
    -----------
    prices_or_df : pd.Series or pd.DataFrame
        價格序列或包含'close'欄位的DataFrame
    window : int
        移動平均窗口（預設50）
    slope_period : int
        計算斜率的天數（預設5）
    current_rsi : float, optional
        當前 RSI 值。若 RSI >= 92，將抑制「強烈買入信號」。
    profit_positive : bool, optional
        公司是否獲利。用於 RSI 極端超賣時的二次判斷。

    Returns:
    --------
    dict: 包含以下鍵值
        - slope: float, 斜率絕對值
        - slope_pct: float, 斜率百分比
        - ma50_current: float, 當前MA50值
        - trend: str, 趨勢判斷
        - signal: str, 交易信號
        - color: str, 顏色標記
    """
    # 處理輸入
    if isinstance(prices_or_df, pd.DataFrame):
        prices = prices_or_df['close']
    else:
        prices = prices_or_df

    # 1. 計算MA50
    ma50 = prices.rolling(window=window).mean()

    # 2. 檢查是否有足夠的數據
    if len(ma50.dropna()) < slope_period:
        return {
            'slope': 0,
            'slope_pct': 0,
            'ma50_current': ma50.iloc[-1] if not ma50.empty else 0,
            'trend': '數據不足',
            'signal': '觀望',
            'color': '🟡',
            'description': '數據不足以計算斜率'
        }

    # 3. 取最近N天的MA50
    recent_ma50 = ma50.dropna().tail(slope_period)

    if len(recent_ma50) < 2:
        return {
            'slope': 0,
            'slope_pct': 0,
            'ma50_current': ma50.iloc[-1],
            'trend': '數據不足',
            'signal': '觀望',
            'color': '🟡',
            'description': '數據不足以計算斜率'
        }

    # 4. 線性回歸計算斜率
    x = np.arange(len(recent_ma50))
    y = recent_ma50.values
    slope = np.polyfit(x, y, 1)[0]

    # 5. 計算斜率百分比
    current_ma50 = ma50.iloc[-1]
    slope_pct = (slope / current_ma50) * 100 if current_ma50 != 0 else 0

    # 6. 判斷趨勢
    # 使用更嚴格門檻: 只有斜率百分比 > 1.0% 才視為「強勢上升」
    if slope_pct > 1.0 and (current_rsi is None or current_rsi < 92):
        trend = "強勢上升"
        signal = "強烈買入信號"
        color = "🟢"
        description = "MA50強勢上揚，趨勢明確向上"
    elif slope_pct > 1.0 and current_rsi is not None and current_rsi >= 92:
        trend = "強勢上升"
        signal = "MA50強勢上升，但RSI>=92已抑制"
        color = "🟡"
        description = "MA50雖強勢上升，但RSI>=92，強烈買入信號已抑制"
    elif slope_pct > 0:
        trend = "溫和上升"
        signal = "MA50微幅上揚，但未達1.0%"
        color = "🟢"
        description = "MA50微幅上揚，但未達1.0%，交易信號靜音"
    elif slope_pct > -0.1:
        trend = "盤整/橫盤"
        signal = "觀望"
        color = "🟡"
        description = "MA50走平，趨勢不明"
    elif slope_pct > -1.0:
        trend = "溫和下降"
        signal = "謹慎看空"
        color = "🔴"
        description = "MA50微幅下滑，趨勢偏空"
    elif current_rsi is None or current_rsi > 10:
        trend = "明顯下降"
        signal = "避免買入"
        color = "🔴"
        description = "MA50明顯下滑，趨勢向下"
    else:
        trend = "明顯下降"
        if profit_positive is True:
            signal = "可考慮分批買入"
            color = "🟡"
            description = "MA50明顯下滑，但RSI<=10且公司獲利為正，可考慮分批買入"
        elif profit_positive is False:
            signal = "觀望"
            color = "🟡"
            description = "MA50明顯下滑，RSI<=10但公司獲利不佳，先觀望"
        else:
            signal = "觀望"
            color = "🟡"
            description = "MA50明顯下滑，但RSI<=10屬極端超賣，請先確認公司獲利再決定"

    return {
        'slope': slope,
        'slope_pct': slope_pct,
        'ma50_current': current_ma50,
        'current_rsi': current_rsi,
        'profit_positive': profit_positive,
        'trend': trend,
        'signal': signal,
        'color': color,
        'description': description
    }


def format_ma50_slope_output(slope_info):
    """
    格式化MA50斜率輸出

    Parameters:
    -----------
    slope_info : dict
        calculate_ma50_slope()的返回結果

    Returns:
    --------
    str: 格式化的輸出文本
    """
    output = []
    output.append("=" * 80)
    output.append(f"{slope_info['color']} MA50趨勢分析")
    output.append("=" * 80)
    output.append(f"當前MA50:      NT${slope_info['ma50_current']:.2f}")
    output.append(f"MA50斜率:      {slope_info['slope']:+.6f}")
    output.append(f"斜率百分比:    {slope_info['slope_pct']:+.4f}%")
    output.append(f"趨勢判斷:      {slope_info['trend']}")
    output.append(f"交易信號:      {slope_info['signal']}")
    output.append(f"\n💡 說明: {slope_info['description']}")

    return "\n".join(output)


def get_ma50_slope_score_adjustment(slope_info):
    """
    根據MA50斜率調整交易評分

    Parameters:
    -----------
    slope_info : dict
        calculate_ma50_slope()的返回結果

    Returns:
    --------
    float: 評分調整值 (-15 到 +15)
    """
    slope_pct = slope_info['slope_pct']
    current_rsi = slope_info.get('current_rsi')

    # RSI過熱時抑制MA50的正向加分
    if current_rsi is not None and current_rsi >= 92 and slope_pct > 0:
        return 0

    # 買入信號的加分 (更嚴格: 只有 >1.0% 才給正分)
    if slope_pct > 1.0:
        return 15  # 超強上升趨勢
    elif slope_pct > 0:
        return 0   # 溫和上升但靜音，不加分
    elif slope_pct > -0.05:
        return 0   # 盤整
    elif slope_pct > -0.15:
        return -5  # 溫和下降
    elif slope_pct > -0.3:
        return -10 # 明顯下降
    else:
        return -15 # 急速下降
