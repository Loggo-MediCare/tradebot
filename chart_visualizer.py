"""
K線圖視覺化模組
Candlestick chart generator using matplotlib
"""
import os
os.environ['MPLBACKEND'] = 'Agg'


def plot_candlestick(df, symbol, save_path=None, title=None):
    """
    繪製K線圖並儲存為PNG

    Args:
        df: DataFrame with close/open/high/low/volume columns
        symbol: stock ticker symbol
        save_path: output file path (default: {symbol}_chart.png)
        title: chart title
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import pandas as pd
        import numpy as np

        if df is None or len(df) < 5:
            return

        # Normalise column names
        col_map = {}
        for c in df.columns:
            col_map[c.lower()] = c
        def gcol(name):
            return col_map.get(name, name)

        close  = df[gcol('close')].astype(float)
        open_  = df[gcol('open')].astype(float)
        high   = df[gcol('high')].astype(float)
        low    = df[gcol('low')].astype(float)

        fig, ax = plt.subplots(figsize=(14, 6))

        x = range(len(df))
        for i, (o, h, l, c) in enumerate(zip(open_, high, low, close)):
            color = '#26a69a' if c >= o else '#ef5350'
            ax.plot([i, i], [l, h], color=color, linewidth=0.8)
            body_h = abs(c - o)
            body_y = min(c, o)
            rect = mpatches.FancyBboxPatch(
                (i - 0.3, body_y), 0.6, max(body_h, 0.01),
                boxstyle="square,pad=0", facecolor=color, edgecolor=color
            )
            ax.add_patch(rect)

        # SMA lines
        for period, color, lw in [(10, '#ff9800', 1.0), (30, '#2196f3', 1.0), (50, '#9c27b0', 1.2)]:
            if len(close) >= period:
                sma = close.rolling(period).mean()
                ax.plot(x, sma, color=color, linewidth=lw, label=f'SMA{period}', alpha=0.8)

        ax.set_title(title or f"{symbol} K線圖", fontsize=14, pad=10)
        ax.set_xlabel("交易日")
        ax.set_ylabel("價格")
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlim(-1, len(df))
        plt.tight_layout()

        if save_path is None:
            save_path = f"{symbol}_chart.png"
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"   📈 K線圖已儲存: {save_path}")

    except Exception as e:
        print(f"   ⚠️  K線圖生成失敗: {e}")
