# -*- coding: utf-8 -*-
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
import sys

# ── Fix Windows console encoding ──────────────────────────────────────────────
# Must be done at module level, not just in __main__, because every print()
# with Chinese text (e.g. "圖表已儲存") will UnicodeEncodeError on a cp950
# console otherwise.
if sys.platform == 'win32':
    import io as _io
    try:
        if hasattr(sys.stdout, 'buffer') and \
                getattr(sys.stdout, 'encoding', '').lower().replace('-', '') != 'utf8':
            sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'buffer') and \
                getattr(sys.stderr, 'encoding', '').lower().replace('-', '') != 'utf8':
            sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, TypeError):
        pass

import numpy as np
import pandas as pd

# Avoid Tk/Tcl dependency when a GUI backend is unavailable.
if not os.environ.get("MPLBACKEND"):
    os.environ["MPLBACKEND"] = "Agg"


def _configure_cjk_fonts() -> None:
    """Configure matplotlib to use a CJK-capable font for Chinese characters.

    Separated from the mplfinance import so font setup always runs even if
    mplfinance is unavailable.  Traditional Chinese fonts (繁體, used in
    Taiwan) are listed before Simplified Chinese ones.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        # Ordered by preference: Traditional Chinese (繁體) first for Taiwan
        cjk_candidates = [
            'Microsoft JhengHei',   # Windows 繁體中文 (微軟正黑體)  ← was missing
            'Microsoft YaHei',      # Windows 簡體中文 (微軟雅黑)
            'PMingLiU',             # Windows 繁體中文 (新細明體)
            'MingLiU',              # Windows 繁體中文 (細明體)
            'SimHei',               # Windows 簡體中文
            'Heiti TC',             # macOS 繁體中文
            'STHeiti',              # macOS
            'WenQuanYi Micro Hei',  # Linux
            'Noto Sans CJK TC',     # Cross-platform 繁體
            'Noto Sans CJK SC',     # Cross-platform 簡體
            'Arial Unicode MS',     # macOS broad Unicode fallback
        ]

        # Filter to fonts actually installed; keep order
        available = {f.name for f in font_manager.fontManager.ttflist}
        ranked = [f for f in cjk_candidates if f in available]

        # Use found fonts first, then full candidate list as final fallback
        plt.rcParams['font.sans-serif'] = (ranked or cjk_candidates) + ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False   # prevent □ for minus sign
    except Exception:
        pass


_configure_cjk_fonts()

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

def prepare_chart_data(df, macd_foot_shrink_threshold_pct: float = 10.0):
    """準備圖表數據，計算均線、爆量指標、MACD 與「收腳/跳空」事件。

    macd_foot_shrink_threshold_pct:
      - 日線 MACD histogram 在負值區間「絕對值縮短」的比例門檻
        (例如 10 表示縮短 >= 10% 才算有效收腳)
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

    # =====================
    # MACD + Histogram
    # =====================
    ema12 = chart_df['close'].ewm(span=12, adjust=False).mean()
    ema26 = chart_df['close'].ewm(span=26, adjust=False).mean()
    chart_df['MACD'] = ema12 - ema26
    chart_df['MACD_Signal'] = chart_df['MACD'].ewm(span=9, adjust=False).mean()
    chart_df['MACD_Hist'] = chart_df['MACD'] - chart_df['MACD_Signal']

    # =====================
    # 「收腳」(Histogram 仍為負，但絕對值縮短)
    # =====================
    prev = chart_df['MACD_Hist'].shift(1)
    curr = chart_df['MACD_Hist']
    shrink_pct = (prev.abs() - curr.abs()) / (prev.abs() + 1e-12) * 100
    chart_df['MACD_FOOT_SHRINK_PCT'] = shrink_pct
    chart_df['MACD_FOOT'] = (
        (curr < 0) & (prev < 0) & (curr.abs() < prev.abs()) &
        (shrink_pct >= float(macd_foot_shrink_threshold_pct))
    )

    # 「跳空向上」：今天開盤 > 昨天最高
    if 'open' in chart_df.columns and 'high' in chart_df.columns:
        chart_df['GAP_UP'] = chart_df['open'] > chart_df['high'].shift(1)
    else:
        chart_df['GAP_UP'] = False

    # 「收腳後隔日跳空確認」：昨天收腳 + 今天跳空
    chart_df['FOOT_GAP_CONFIRM'] = (
        chart_df['MACD_FOOT'].shift(1).fillna(False).astype(bool)
        & chart_df['GAP_UP'].fillna(False).astype(bool)
    )

    # Volume surge (ratio > 1.5x 20-day avg)
    chart_df['VolSurge'] = chart_df['volume'] > chart_df['Vol_MA20'] * 1.5

    # Breakout detection
    chart_df['High20'] = chart_df['high'].rolling(20).max()
    chart_df['TrueBreakout'] = (chart_df['close'] > chart_df['High20'].shift(1)) & chart_df['VolSurge']
    chart_df['FakeBreakout'] = (chart_df['close'] > chart_df['High20'].shift(1)) & (~chart_df['VolSurge'])

    return chart_df



def plot_candlestick(
    df,
    stock_name="Stock",
    save_path=None,
    show_macd: bool = True,
    show_macd_foot: bool = True,
    macd_foot_shrink_threshold_pct: float = 10.0,
):
    """繪製 K 線圖 + 均線 + 爆量標記 + (可選) MACD + 「收腳/跳空」標記。"""

    if not MPF_AVAILABLE:
        print("mplfinance not available. Cannot plot chart.")
        return None

    # 準備數據
    chart_df = prepare_chart_data(df, macd_foot_shrink_threshold_pct=macd_foot_shrink_threshold_pct)

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

    apds = []

    # 均線
    apds.append(mpf.make_addplot(chart_df['MA5'], color='blue', width=0.8))
    apds.append(mpf.make_addplot(chart_df['MA10'], color='orange', width=0.8))
    apds.append(mpf.make_addplot(chart_df['MA20'], color='purple', width=1.0))

    # 成交量均線
    if 'Vol_MA20' in chart_df.columns:
        apds.append(mpf.make_addplot(chart_df['Vol_MA20'], panel=1, color='blue', width=0.8))

    # Volume surge bar overlay
    if 'VolSurge' in chart_df.columns:
        surge_vol = chart_df['volume'].where(chart_df['VolSurge'], np.nan)
        apds.append(mpf.make_addplot(surge_vol, panel=1, type='bar', color='red', alpha=0.5))

    # MACD histogram + MACD/Signal lines (panel=2)
    if show_macd and 'MACD_Hist' in chart_df.columns:
        hist = chart_df['MACD_Hist']
        hist_pos = hist.where(hist >= 0, np.nan)
        hist_neg = hist.where(hist < 0, np.nan)
        apds.append(mpf.make_addplot(hist_pos, panel=2, type='bar', color='#2ecc71', alpha=0.7, width=0.8))
        apds.append(mpf.make_addplot(hist_neg, panel=2, type='bar', color='#e74c3c', alpha=0.7, width=0.8))
        apds.append(mpf.make_addplot(pd.Series(0, index=hist.index), panel=2, color='gray', width=0.5))
        apds.append(mpf.make_addplot(chart_df['MACD'], panel=2, color='#3498db', width=1.2))
        apds.append(mpf.make_addplot(chart_df['MACD_Signal'], panel=2, color='#e67e22', width=1.2))

    # 收腳/跳空標記（主圖）
    if show_macd_foot and 'MACD_FOOT' in chart_df.columns:
        foot_price = chart_df['low'].where(chart_df['MACD_FOOT'], np.nan)
        if foot_price.notna().any():
            apds.append(mpf.make_addplot(foot_price * 0.995, type='scatter', marker='^', markersize=120, color='orange', alpha=0.9))

        confirm_price = chart_df['low'].where(chart_df.get('FOOT_GAP_CONFIRM', False), np.nan)
        if isinstance(confirm_price, pd.Series) and confirm_price.notna().any():
            apds.append(mpf.make_addplot(confirm_price * 0.99, type='scatter', marker='*', markersize=180, color='purple', alpha=0.9))

    fig_params = {
        'type': 'candle',
        'style': style,
        'volume': True,
        'title': f'\n{stock_name} Candlestick + Volume Surge (MACD / Hist-Foot)',
        'ylabel': 'Price',
        'ylabel_lower': 'Volume',
        'figsize': (14, 10),
        'addplot': apds,
    }

    if show_macd:
        fig_params['panel_ratios'] = (3, 1, 1)

    # Resolve save path before plotting
    if save_path:
        from datetime import datetime as _dt
        _date_str = _dt.now().strftime('%Y%m%d')
        _base, _ext = os.path.splitext(save_path)
        if not _base.endswith(f'_{_date_str}'):
            save_path = f"{_base}_{_date_str}{_ext}"
        if os.path.exists(save_path):
            os.remove(save_path)

    fig_params['returnfig'] = True
    fig, axes = mpf.plot(plot_df, **fig_params)

    # Add legend to MACD panel
    if show_macd:
        import matplotlib.patches as _mpatches
        from matplotlib.lines import Line2D as _Line2D
        legend_elements = [
            _Line2D([0], [0], color='#3498db', linewidth=1.2, label='MACD'),
            _Line2D([0], [0], color='#e67e22', linewidth=1.2, label='Signal'),
            _mpatches.Patch(facecolor='#2ecc71', alpha=0.7, label='Hist +'),
            _mpatches.Patch(facecolor='#e74c3c', alpha=0.7, label='Hist -'),
        ]
        for ax in reversed(axes):
            try:
                if ax.lines or ax.patches:
                    ax.legend(handles=legend_elements, loc='upper left', fontsize=7, framealpha=0.7)
                    break
            except Exception:
                pass

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close('all')
        print(f"圖表已儲存: {save_path}")
    else:
        plt.show()
        plt.close('all')

    cols = ['close', 'volume', 'Vol_MA20', 'VolSurge', 'TrueBreakout', 'FakeBreakout']
    for extra in ['MACD', 'MACD_Signal', 'MACD_Hist', 'MACD_FOOT', 'MACD_FOOT_SHRINK_PCT', 'GAP_UP', 'FOOT_GAP_CONFIRM']:
        if extra in chart_df.columns:
            cols.append(extra)
    cols = [c for c in cols if c in chart_df.columns]
    return chart_df[cols].tail(10)




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
    # sys.stdout UTF-8 encoding is already handled at module level above;
    # no need to re-wrap here (double-wrapping causes a ValueError).

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
    print(chart_df[['close', 'volume', 'Vol_MA20', 'VolSurge', 'TrueBreakout', 'FakeBreakout']].tail(10))

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
