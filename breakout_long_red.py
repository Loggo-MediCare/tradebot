"""
突破長紅K線檢測模組
Detects strong bullish breakout candles (long red/green bars with volume)
"""
import numpy as np


def get_breakout_long_red_signal(df, min_body_pct=3.0, volume_ratio_threshold=1.5):
    """
    檢測突破長紅K線信號

    Args:
        df: DataFrame with close/open/high/low/volume columns
        min_body_pct: minimum candle body size as % of price
        volume_ratio_threshold: volume must be this multiple of 20-day average

    Returns:
        dict: {
            'detected': bool,
            'score_adjustment': int,
            'signal_text': str,
            'body_pct': float,
            'volume_ratio': float
        }
    """
    result = {
        'detected': False,
        'score_adjustment': 0,
        'signal_text': '',
        'body_pct': 0.0,
        'volume_ratio': 0.0
    }

    try:
        if df is None or len(df) < 21:
            return result

        close_col  = 'close'  if 'close'  in df.columns else 'Close'
        open_col   = 'open'   if 'open'   in df.columns else 'Open'
        volume_col = 'volume' if 'volume' in df.columns else 'Volume'

        latest      = df.iloc[-1]
        close_price = float(latest[close_col])
        open_price  = float(latest[open_col])
        volume      = float(latest[volume_col])

        # Body size as % of open price
        body_pct = (close_price - open_price) / (open_price + 1e-10) * 100

        # 20-day average volume
        avg_vol = float(df[volume_col].iloc[-21:-1].mean())
        vol_ratio = volume / (avg_vol + 1e-10)

        result['body_pct']     = round(body_pct, 2)
        result['volume_ratio'] = round(vol_ratio, 2)

        if body_pct >= min_body_pct and vol_ratio >= volume_ratio_threshold:
            result['detected']        = True
            result['score_adjustment'] = 15
            result['signal_text']      = (
                f"突破長紅K線 (漲幅{body_pct:.1f}%, 量比{vol_ratio:.1f}x)"
            )
        elif body_pct >= min_body_pct * 0.7 and vol_ratio >= volume_ratio_threshold * 0.8:
            # Weaker signal
            result['detected']        = True
            result['score_adjustment'] = 8
            result['signal_text']      = (
                f"長紅K線 (漲幅{body_pct:.1f}%, 量比{vol_ratio:.1f}x)"
            )

    except Exception:
        pass

    return result
