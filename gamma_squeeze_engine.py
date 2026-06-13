"""
Gamma Squeeze 機率引擎 (Gamma Squeeze Probability Engine)
======================================
綜合多項籌碼/動能訊號，計算 0-100 分的「Gamma Squeeze 形成機率」。

複用:
  - gamma_flip_gate.py    : GEX / Gamma Flip 計算 (_fetch_nearest_chain, get_gamma_flip)
  - finbert_enhanced_scoring.py : FinBERT 新聞情緒分析 (calculate_sentiment_score)

評分公式 (0-100，可被 FinBERT 情緒加減分推高/推低，最終夾在 0-100):
    GEX 翻正 (現價 >= Gamma Flip)      : +20
    Call/Put Volume Ratio > 2          : +20
    Chaikin Volatility > 30%           : +10
    RVOL (相對成交量) > 2               : +15
    現價突破 Call Wall                  : +35

機率區間:
    < 30   : 無 Gamma Squeeze 風險
    30-50  : 可能形成
    50-70  : 高機率形成
    > 70   : 已開始 Gamma Squeeze

用法:
    python gamma_squeeze_engine.py MU AVGO NVDA TSM
"""

import pandas as pd
import yfinance as yf

from gamma_flip_gate import get_gamma_flip, _fetch_nearest_chain

try:
    from finbert_enhanced_scoring import calculate_sentiment_score
except ImportError:
    calculate_sentiment_score = None


# ======================================================
# 個別指標計算
# ======================================================

def get_chaikin_volatility(df, ema_period=10, roc_period=10):
    """Chaikin Volatility (%)

    CV = (EMA(High-Low) - EMA(High-Low) N期前) / EMA(High-Low) N期前 * 100
    """
    hl = df['High'] - df['Low']
    ema_hl = hl.ewm(span=ema_period, adjust=False).mean()
    prev = ema_hl.shift(roc_period)
    cv = (ema_hl - prev) / prev * 100
    return float(cv.iloc[-1])


def get_rvol(df, period=20):
    """RVOL = 今日成交量 / period 日均量"""
    avg_vol = df['Volume'].rolling(period).mean().iloc[-1]
    if pd.isna(avg_vol) or avg_vol == 0:
        return 0.0
    return float(df['Volume'].iloc[-1] / avg_vol)


def get_call_put_walls(calls_df, puts_df):
    """依未平倉量 (Open Interest) 找出 Call Wall / Put Wall"""
    call_wall = put_wall = None
    if not calls_df.empty and calls_df['openInterest'].sum() > 0:
        call_wall = float(calls_df.loc[calls_df['openInterest'].idxmax(), 'strike'])
    if not puts_df.empty and puts_df['openInterest'].sum() > 0:
        put_wall = float(puts_df.loc[puts_df['openInterest'].idxmax(), 'strike'])
    return call_wall, put_wall


def get_call_volume_ratio(calls_df, puts_df):
    """Call/Put 成交量比 — 衡量買權追價力道"""
    call_vol = calls_df['volume'].fillna(0).sum()
    put_vol = puts_df['volume'].fillna(0).sum()
    if put_vol <= 0:
        return float('inf') if call_vol > 0 else 0.0
    return float(call_vol / put_vol)


# ======================================================
# Master Gamma Squeeze Engine
# ======================================================

def get_gamma_squeeze_score(symbol, verbose=False):
    """計算單一標的 Gamma Squeeze 機率分數 (0-100)

    Returns:
        dict: {
            'symbol', 'score', 'level', 'spot',
            'gamma_flip', 'gex_positive', 'data_quality',
            'call_wall', 'put_wall', 'above_call_wall',
            'call_volume_ratio', 'chaikin_volatility', 'rvol',
            'sentiment_score', 'sentiment_adjustment',
            'reasons'
        }
    """
    result = {
        'symbol': symbol, 'score': 0, 'level': '無',
        'spot': None, 'gamma_flip': None, 'gex_positive': False, 'data_quality': 'ERROR',
        'call_wall': None, 'put_wall': None, 'above_call_wall': False,
        'call_volume_ratio': 0.0, 'chaikin_volatility': 0.0, 'rvol': 0.0,
        'sentiment_score': 0.0, 'sentiment_adjustment': 0,
        'reasons': [],
    }

    try:
        hist = yf.Ticker(symbol).history(period="3mo")
        if hist.empty:
            result['reasons'].append("無法取得價格資料")
            return result

        spot = float(hist['Close'].iloc[-1])
        result['spot'] = spot

        # 1. Chaikin Volatility
        chaikin = get_chaikin_volatility(hist)
        result['chaikin_volatility'] = chaikin
        if chaikin > 30:
            result['score'] += 10
            result['reasons'].append(f"Chaikin Volatility {chaikin:.1f}% > 30%")

        # 2. RVOL
        rvol = get_rvol(hist)
        result['rvol'] = rvol
        if rvol > 2:
            result['score'] += 15
            result['reasons'].append(f"RVOL {rvol:.2f}x > 2x")

        # 3. GEX 翻正 (複用 gamma_flip_gate)
        try:
            gamma_flip, found, dq = get_gamma_flip(symbol, spot)
            result['gamma_flip'] = gamma_flip
            result['data_quality'] = dq

            if found and gamma_flip is not None:
                gex_positive = spot >= gamma_flip
                result['gex_positive'] = gex_positive
                if gex_positive:
                    result['score'] += 20
                    result['reasons'].append(
                        f"現價 ${spot:.2f} >= Gamma Flip ${gamma_flip:.2f} (GEX轉正)")
        except Exception as e:
            result['reasons'].append(f"Gamma Flip 計算失敗: {e}")

        # 4. Call Wall / Put Wall / Call Volume Ratio
        try:
            calls_df, puts_df, expiry = _fetch_nearest_chain(symbol)

            call_wall, put_wall = get_call_put_walls(calls_df, puts_df)
            result['call_wall'] = call_wall
            result['put_wall'] = put_wall

            if call_wall is not None and spot > call_wall:
                result['above_call_wall'] = True
                result['score'] += 35
                result['reasons'].append(
                    f"現價 ${spot:.2f} > Call Wall ${call_wall:.2f} (已突破)")

            cvr = get_call_volume_ratio(calls_df, puts_df)
            result['call_volume_ratio'] = cvr
            if cvr > 2:
                result['score'] += 20
                result['reasons'].append(f"Call/Put Volume Ratio {cvr:.2f} > 2")
        except Exception as e:
            result['reasons'].append(f"選擇權鏈資料取得失敗: {e}")

        # 5. FinBERT 新聞情緒 (加減分)
        if calculate_sentiment_score is not None:
            try:
                sentiment = calculate_sentiment_score(symbol, verbose=verbose)
                result['sentiment_score'] = sentiment.get('sentiment_score', 0.0)
                adj = sentiment.get('score_adjustment', 0)
                result['sentiment_adjustment'] = adj
                if adj != 0:
                    result['score'] += adj
                    result['reasons'].append(
                        f"FinBERT 情緒 {sentiment.get('sentiment_label', '')} ({adj:+d})")
            except Exception as e:
                result['reasons'].append(f"FinBERT 情緒分析失敗: {e}")

        result['score'] = max(0, min(100, result['score']))

        if result['score'] > 70:
            result['level'] = '已開始 Gamma Squeeze'
        elif result['score'] >= 50:
            result['level'] = '高機率形成'
        elif result['score'] >= 30:
            result['level'] = '可能形成'
        else:
            result['level'] = '無'

    except Exception as e:
        result['reasons'].append(f"計算失敗: {e}")

    return result


def format_gamma_squeeze_output(result):
    """格式化單一標的輸出"""
    lines = [f"{result['symbol']} Gamma Squeeze: {result['score']}% ({result['level']})"]

    if result['spot'] is not None:
        lines.append(f"   現價: ${result['spot']:.2f}")

    if result['gamma_flip'] is not None:
        gex_txt = "正" if result['gex_positive'] else "負"
        lines.append(f"   Gamma Flip: ${result['gamma_flip']:.2f}  (GEX: {gex_txt}, 數據品質: {result['data_quality']})")

    if result['call_wall'] is not None:
        cw_txt = f"   Call Wall: ${result['call_wall']:.2f}"
        if result['above_call_wall']:
            cw_txt += "  <- 已突破"
        lines.append(cw_txt)

    if result['put_wall'] is not None:
        lines.append(f"   Put Wall: ${result['put_wall']:.2f}")

    lines.append(f"   Call/Put Volume Ratio: {result['call_volume_ratio']:.2f}")
    lines.append(f"   Chaikin Volatility: {result['chaikin_volatility']:.1f}%")
    lines.append(f"   RVOL: {result['rvol']:.2f}x")

    if result['sentiment_adjustment'] != 0:
        lines.append(f"   FinBERT 情緒分數: {result['sentiment_score']:.2f} ({result['sentiment_adjustment']:+d})")

    if result['reasons']:
        lines.append("   觸發條件:")
        for r in result['reasons']:
            lines.append(f"     - {r}")

    return "\n".join(lines)


def scan_gamma_squeeze(symbols, verbose=False):
    """掃描多個標的，回傳依分數排序的結果列表"""
    results = [get_gamma_squeeze_score(s, verbose=verbose) for s in symbols]
    results.sort(key=lambda r: r['score'], reverse=True)
    return results


# ======================================================
# 主程序
# ======================================================

if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    symbols = [s.upper() for s in sys.argv[1:]] or ['MU', 'AVGO', 'NVDA', 'TSM']

    print("=" * 60)
    print("Gamma Squeeze Probability Engine")
    print("=" * 60)

    results = []
    for symbol in symbols:
        print()
        r = get_gamma_squeeze_score(symbol)
        results.append(r)
        print(format_gamma_squeeze_output(r))

    print()
    print("=" * 60)
    print("摘要排行 (由高到低)")
    print("=" * 60)
    for r in sorted(results, key=lambda x: x['score'], reverse=True):
        print(f"  {r['symbol']:6s} {r['score']:3d}%  {r['level']}")
