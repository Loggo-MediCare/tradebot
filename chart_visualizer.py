"""
K線圖表視覺化模組
==================
繪製 K 棒 + 均線 + 爆量標記 + 型態區域

功能:
1. 標準 K 線圖 + 均線
2. 爆量/真假突破標記
3. 型態區域高亮
4. 籌碼選股掃描器

用法:
    from chart_visualizer import plot_candlestick, SmartMoneyBot
    plot_candlestick(df, "2330.TW")
"""

import os
import numpy as np
import pandas as pd

# Avoid Tk/Tcl dependency when a GUI backend is unavailable.
if not os.environ.get("MPLBACKEND"):
    os.environ["MPLBACKEND"] = "Agg"

try:
    import mplfinance as mpf
    import matplotlib.pyplot as plt
    MPF_AVAILABLE = True
except ImportError:
    MPF_AVAILABLE = False
    print("Warning: mplfinance not installed. Chart functions will be limited.")


# ======================================================
# K 線圖繪製
# ======================================================

def prepare_chart_data(df):
    """
    準備圖表數據，計算均線和爆量指標
    """
    chart_df = df.copy()

    # 確保列名正確
    column_map = {
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume'
    }
    for old, new in column_map.items():
        if old in chart_df.columns:
            chart_df.rename(columns={old: new}, inplace=True)

    # 計算均線
    chart_df['MA5'] = chart_df['close'].rolling(5).mean()
    chart_df['MA10'] = chart_df['close'].rolling(10).mean()
    chart_df['MA20'] = chart_df['close'].rolling(20).mean()
    chart_df['MA50'] = chart_df['close'].rolling(50).mean()

    # 計算成交量均線
    chart_df['Vol_MA5'] = chart_df['volume'].rolling(5).mean()
    chart_df['Vol_MA20'] = chart_df['volume'].rolling(20).mean()

    # 爆量判斷 (量比 > 1.5)
    chart_df['爆量'] = chart_df['volume'] > chart_df['Vol_MA20'] * 1.5

    # 突破判斷
    chart_df['20日高'] = chart_df['high'].rolling(20).max()
    chart_df['真突破'] = (chart_df['close'] > chart_df['20日高'].shift(1)) & chart_df['爆量']
    chart_df['假突破'] = (chart_df['close'] > chart_df['20日高'].shift(1)) & (~chart_df['爆量'])

    return chart_df


def plot_candlestick(df, stock_name="Stock", save_path=None):
    """
    繪製 K 線圖 + 均線 + 爆量標記

    Args:
        df: DataFrame with OHLCV data
        stock_name: 股票名稱
        save_path: 儲存路徑 (None 則顯示)

    Returns:
        DataFrame with calculated indicators
    """
    if not MPF_AVAILABLE:
        print("mplfinance not available. Cannot plot chart.")
        return None

    # 準備數據
    chart_df = prepare_chart_data(df)

    # 設定 index 為 datetime
    if 'Date' in chart_df.columns:
        chart_df['Date'] = pd.to_datetime(chart_df['Date'])
        chart_df.set_index('Date', inplace=True)
    elif not isinstance(chart_df.index, pd.DatetimeIndex):
        chart_df.index = pd.to_datetime(chart_df.index)

    # 重命名列為 mplfinance 格式
    plot_df = chart_df[['open', 'high', 'low', 'close', 'volume']].copy()
    plot_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # 台灣股市風格: 紅漲綠跌
    mc = mpf.make_marketcolors(up='r', down='g', inherit=True)
    style = mpf.make_mpf_style(marketcolors=mc)

    # 設定附加圖層
    apds = []

    # 均線
    if 'MA5' in chart_df.columns:
        apds.append(mpf.make_addplot(chart_df['MA5'], color='blue', width=0.8))
    if 'MA10' in chart_df.columns:
        apds.append(mpf.make_addplot(chart_df['MA10'], color='orange', width=0.8))
    if 'MA20' in chart_df.columns:
        apds.append(mpf.make_addplot(chart_df['MA20'], color='purple', width=1.0))

    # 成交量均線
    if 'Vol_MA20' in chart_df.columns:
        apds.append(mpf.make_addplot(chart_df['Vol_MA20'], panel=1, color='blue', width=0.8))

    # 爆量標記 (紅色 bar)
    if '爆量' in chart_df.columns:
        surge_vol = chart_df['volume'].where(chart_df['爆量'], np.nan)
        apds.append(mpf.make_addplot(surge_vol, panel=1, type='bar', color='red', alpha=0.5))

    # 繪圖
    fig_params = {
        'type': 'candle',
        'style': style,
        'volume': True,
        'title': f'\n{stock_name} K線圖 + 爆量分析',
        'ylabel': 'Price',
        'ylabel_lower': 'Volume',
        'figsize': (14, 8)
    }

    if apds:
        fig_params['addplot'] = apds

    if save_path:
        if os.path.exists(save_path):
            os.remove(save_path)
        fig_params['savefig'] = save_path
        mpf.plot(plot_df, **fig_params)
        plt.close('all')
        print(f"圖表已儲存: {save_path}")
    else:
        mpf.plot(plot_df, **fig_params)

    return chart_df[['close', 'volume', 'Vol_MA20', '爆量', '真突破', '假突破']].tail(10)


def plot_smart_k_bars(ticker, start_date='2023-01-01', save=True):
    """
    繪製《小散戶這樣追籌碼賺1億》風格的K線圖
    包含: K棒、均線(5, 10, 20日)、成交量

    Args:
        ticker: 股票代碼
        start_date: 開始日期
        save: 是否儲存圖片
    """
    if not MPF_AVAILABLE:
        print("mplfinance not available")
        return None

    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(start=start_date)

        if df.empty:
            print(f"找不到 {ticker} 的資料")
            return None

        # 計算關鍵均線 (5日, 10日, 20日)
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()

        # 爆量檢測
        df['Vol_MA5'] = df['Volume'].rolling(window=5).mean()

        # 台灣股市風格: 紅漲綠跌
        mc = mpf.make_marketcolors(up='r', down='g', inherit=True)
        style = mpf.make_mpf_style(marketcolors=mc)

        # 均線 plot
        apds = [
            mpf.make_addplot(df['MA5'], color='blue', width=1.0),
            mpf.make_addplot(df['MA10'], color='orange', width=1.0),
            mpf.make_addplot(df['MA20'], color='purple', width=1.2),
        ]

        # 繪製
        save_path = f"{ticker.replace('.', '_')}_chart.png" if save else None

        mpf.plot(df, type='candle', style=style,
                 addplot=apds,
                 volume=True,
                 title=f'\n{ticker} - Volume Breakout & MA Strategy',
                 ylabel='Price',
                 ylabel_lower='Volume',
                 savefig=save_path if save else None,
                 figsize=(14, 8))

        if save:
            print(f"圖表已儲存: {save_path}")

        return df

    except Exception as e:
        print(f"Error: {e}")
        return None


def plot_with_pattern_zones(df, patterns, stock_name="Stock", save_path=None):
    """
    繪製 K 線圖並高亮型態區域

    Args:
        df: DataFrame with OHLCV data
        patterns: list of detected patterns
        stock_name: 股票名稱
        save_path: 儲存路徑
    """
    if not MPF_AVAILABLE:
        print("mplfinance not available.")
        return

    chart_df = prepare_chart_data(df)

    if 'Date' in chart_df.columns:
        chart_df['Date'] = pd.to_datetime(chart_df['Date'])
        chart_df.set_index('Date', inplace=True)

    plot_df = chart_df[['open', 'high', 'low', 'close', 'volume']].copy()
    plot_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    mc = mpf.make_marketcolors(up='r', down='g', inherit=True)
    style = mpf.make_mpf_style(marketcolors=mc)

    # 高亮區域
    fig, axes = mpf.plot(plot_df, type='candle', style=style, volume=True,
                         title=f'\n{stock_name} - Pattern Analysis',
                         returnfig=True, figsize=(14, 8))

    ax = axes[0]

    # 根據型態高亮不同區域
    if "TRIANGLE" in patterns:
        start_idx = max(0, len(plot_df) - 60)
        end_idx = len(plot_df) - 5
        ax.axvspan(start_idx, end_idx, alpha=0.2, color='orange', label='Triangle')

    if "FLAG" in patterns:
        start_idx = max(0, len(plot_df) - 40)
        end_idx = len(plot_df) - 10
        ax.axvspan(start_idx, end_idx, alpha=0.2, color='blue', label='Flag')

    if "W_BOTTOM" in patterns:
        start_idx = max(0, len(plot_df) - 80)
        end_idx = len(plot_df) - 5
        ax.axvspan(start_idx, end_idx, alpha=0.2, color='green', label='W Bottom')

    ax.legend()

    if save_path:
        fig.savefig(save_path)
        print(f"圖表已儲存: {save_path}")
    else:
        plt.show()


# ======================================================
# 籌碼選股掃描器 (SmartMoneyBot)
# ======================================================

class SmartMoneyBot:
    """
    籌碼選股機器人 - 實作《小散戶這樣追籌碼賺1億》策略

    策略條件:
    1. 爆量: 當日成交量 > 5日均量 * 2
    2. 突破: 收盤價 > 20日線 且 漲幅 > 3%
    3. 均線多頭: 價格 > MA5 > MA10
    """

    def __init__(self, ticker_list):
        self.tickers = ticker_list
        self.selections = []

    def fetch_data(self, ticker):
        """下載股票數據"""
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            df = stock.history(period="6mo")

            if df.empty:
                return None

            # 標準化列名
            df.columns = [c.lower() for c in df.columns]
            return df

        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return None

    def check_strategy(self, df):
        """
        實作《小散戶這樣追籌碼賺1億》策略:
        1. 均線糾結: 5, 10, 20日均線差距在 5% 以內 (整理區間)
        2. 帶量突破: 收盤價 > 20日均線 AND 成交量 > 2倍的5日均量
        3. 多頭排列: 價格 > MA5 > MA10 > MA20

        Returns:
            (bool, str): (是否符合, 原因說明)
        """
        if len(df) < 25:
            return False, "資料不足"

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        # 計算均線
        ma5 = df['close'].rolling(5).mean().iloc[-1]
        ma10 = df['close'].rolling(10).mean().iloc[-1]
        ma20 = df['close'].rolling(20).mean().iloc[-1]
        vol_ma5 = df['volume'].rolling(5).mean().iloc[-1]

        # 條件 1: 爆量 (當日量 > 2 * 5日均量)
        volume_ratio = curr['volume'] / vol_ma5 if vol_ma5 > 0 else 0
        is_explosive = volume_ratio > 2.0

        # 條件 2: 突破 (收盤價突破20日線 且 漲幅 > 3%)
        price_change = (curr['close'] - prev['close']) / prev['close'] if prev['close'] > 0 else 0
        is_breakout = (curr['close'] > ma20) and (price_change > 0.03)

        # 條件 3: 均線多頭排列 (價格 > MA5 > MA10 > MA20)
        is_bullish = curr['close'] > ma5 and ma5 > ma10 and ma10 > ma20

        # 條件 4: 均線糾結後發散 (均線差距 < 5% 代表整理)
        ma_max = max(ma5, ma10, ma20)
        ma_min = min(ma5, ma10, ma20)
        ma_convergence = (ma_max - ma_min) / ma_min < 0.05 if ma_min > 0 else False

        # 判斷信號
        if is_explosive and is_breakout:
            msg = f"爆量突破! 收盤: {curr['close']:.2f}, 量能: {volume_ratio:.1f}x, 漲幅: {price_change:.1%}"
            if ma_convergence:
                msg += " (均線糾結後發散)"
            return True, msg

        if is_bullish and is_explosive:
            return True, f"多頭爆量! 收盤: {curr['close']:.2f}, 量能: {volume_ratio:.1f}x, 多頭排列"

        if ma_convergence and is_breakout:
            return True, f"糾結突破! 收盤: {curr['close']:.2f}, 均線收斂後向上突破"

        return False, ""

    def run_scan(self):
        """執行選股掃描"""
        print("=" * 60)
        print("籌碼選股掃描器 (Smart Money Bot)")
        print("=" * 60)
        print(f"掃描 {len(self.tickers)} 檔股票...\n")

        self.selections = []

        for ticker in self.tickers:
            df = self.fetch_data(ticker)

            if df is not None:
                match, reason = self.check_strategy(df)
                if match:
                    print(f"  [BUY] {ticker}: {reason}")
                    self.selections.append({'ticker': ticker, 'reason': reason})

        print(f"\n選出 {len(self.selections)} 檔股票")
        print("=" * 60)

        return self.selections

    def generate_report(self):
        """生成選股報告"""
        if not self.selections:
            return "未選出任何股票"

        report = []
        report.append("=" * 60)
        report.append("籌碼選股報告")
        report.append("=" * 60)

        for item in self.selections:
            report.append(f"\n{item['ticker']}")
            report.append(f"  原因: {item['reason']}")

        return "\n".join(report)


# ======================================================
# 快速分析函數
# ======================================================

def analyze_and_plot(ticker, save=False):
    """
    快速分析並繪圖

    Args:
        ticker: 股票代碼
        save: 是否儲存圖片
    """
    try:
        import yfinance as yf

        print(f"下載 {ticker} 數據...")
        stock = yf.Ticker(ticker)
        df = stock.history(period="6mo")

        if df.empty:
            print(f"無法取得 {ticker} 數據")
            return

        df.columns = [c.lower() for c in df.columns]

        # 繪圖
        save_path = f"{ticker.replace('.', '_')}_chart.png" if save else None
        result = plot_candlestick(df, ticker, save_path)

        if result is not None:
            print(f"\n{ticker} 最近10日數據:")
            print(result)

        # 型態分析
        from pattern_engine import pattern_engine
        patterns = pattern_engine(df)

        if patterns:
            print(f"\n檢測到型態: {', '.join(patterns)}")

    except Exception as e:
        print(f"Error: {e}")


# ======================================================
# 主程序測試
# ======================================================

if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 60)
    print("Chart Visualizer Test")
    print("=" * 60)

    # 模擬測試數據
    np.random.seed(42)
    n = 100

    prices = 100 + np.cumsum(np.random.randn(n) * 2)
    volumes = np.random.randint(100000, 500000, n)

    # 模擬爆量
    volumes[-3] = volumes[-10:-3].mean() * 2.5
    volumes[-1] = volumes[-10:-1].mean() * 2.0

    df = pd.DataFrame({
        'Date': pd.date_range(start='2025-10-01', periods=n),
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': volumes
    })

    # 準備圖表數據
    chart_df = prepare_chart_data(df)

    print("\nChart data summary:")
    print(chart_df[['close', 'volume', 'Vol_MA20', '爆量', '真突破', '假突破']].tail(10))

    print("\n" + "=" * 60)
    print("SmartMoneyBot Test (with sample data)")
    print("=" * 60)

    # 測試 SmartMoneyBot
    print("\nTo run SmartMoneyBot with real data:")
    print("  bot = SmartMoneyBot(['2330.TW', '2317.TW', 'NVDA', 'TSLA'])")
    print("  results = bot.run_scan()")

    if MPF_AVAILABLE:
        print("\nmplfinance is available. You can plot charts.")
        print("  plot_candlestick(df, 'TestStock')")
    else:
        print("\nmplfinance not installed. Install with: pip install mplfinance")
