"""
Pre-market 動態止損監控
執行時機：美股盤前 (台灣時間 16:00–21:30)
用法：  python pre_market_monitor.py
"""

import os, sys, io
os.environ['MPLBACKEND'] = 'Agg'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── 持倉清單（依實際持股修改）────────────────────────────────────────────────
# symbol: 股票代碼
# name:   公司名稱
# qty:    持股數
# cost:   成本價（用於計算損益）
HOLDINGS = [
    {'symbol': 'NVDA',  'name': 'NVIDIA',          'qty': 100,  'cost': 180.00},
    {'symbol': 'GOOG',  'name': 'Google',           'qty': 50,   'cost': 170.00},
    {'symbol': 'MU',    'name': 'Micron',            'qty': 20,   'cost': 850.00},
    {'symbol': 'SNDK',  'name': 'SanDisk',           'qty': 10,   'cost': 1400.00},
    {'symbol': 'AVGO',  'name': 'Broadcom',          'qty': 30,   'cost': 400.00},
    {'symbol': 'AMD',   'name': 'AMD',               'qty': 40,   'cost': 120.00},
    {'symbol': 'TSM',   'name': 'TSMC ADR',          'qty': 60,   'cost': 150.00},
    # 台股（若需要盤前監控可加，但台股盤前資料有限）
    # {'symbol': '2454.TW', 'name': '聯發科', 'qty': 5, 'cost': 900.00},
]

ATR_PERIOD    = 14   # ATR 計算週期
ATR_MULT      = 1.5  # 止損倍數
LOOKBACK_DAYS = 20   # 最高收盤回顧天數

# ── 時間判斷 ─────────────────────────────────────────────────────────────────
def get_market_session() -> str:
    """回傳目前美東時間的市場狀態"""
    et = timezone(timedelta(hours=-4))  # EDT (夏令時)
    now_et = datetime.now(et)
    h, m = now_et.hour, now_et.minute
    total_min = h * 60 + m

    if 240 <= total_min < 570:    # 4:00–9:30
        return 'pre-market'
    elif 570 <= total_min < 960:  # 9:30–16:00
        return 'regular'
    elif 960 <= total_min < 1200: # 16:00–20:00
        return 'after-hours'
    else:
        return 'closed'

# ── 取得最新價（含盤前/盤後）────────────────────────────────────────────────
def get_latest_price(symbol: str) -> tuple[float | None, str]:
    """
    回傳 (price, source)
    source: 'pre-market' / 'regular' / 'after-hours' / 'last-close'
    """
    try:
        tk = yf.Ticker(symbol)
        fi = tk.fast_info

        # 盤前
        pre = getattr(fi, 'pre_market_price', None)
        if pre and pre > 0:
            return float(pre), 'pre-market'

        # 盤後
        post = getattr(fi, 'post_market_price', None)
        if post and post > 0:
            return float(post), 'after-hours'

        # 即時 / 最後收盤
        last = getattr(fi, 'last_price', None) or getattr(fi, 'regular_market_price', None)
        if last and last > 0:
            return float(last), 'last-close'

    except Exception:
        pass

    # fallback: 1分鐘 K 線（含盤前盤後）
    try:
        df1m = yf.download(symbol, period='1d', interval='1m',
                           prepost=True, progress=False, auto_adjust=True)
        if not df1m.empty:
            if isinstance(df1m.columns, pd.MultiIndex):
                df1m.columns = df1m.columns.droplevel(1)
            return float(df1m['Close'].iloc[-1]), 'realtime-1m'
    except Exception:
        pass

    return None, 'N/A'

# ── 計算動態止損 ─────────────────────────────────────────────────────────────
def calc_trailing_stop(symbol: str) -> dict:
    """下載日線資料，計算 ATR₁₄ 與動態止損"""
    try:
        df = yf.download(symbol, period='60d', progress=False, auto_adjust=True)
        if df.empty:
            return {}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.columns = [c.lower() for c in df.columns]

        tr = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': (df['high'] - df['close'].shift(1)).abs(),
            'lc': (df['low']  - df['close'].shift(1)).abs(),
        }).max(axis=1)
        atr14         = float(tr.rolling(ATR_PERIOD).mean().iloc[-1])
        highest_close = float(df['close'].tail(LOOKBACK_DAYS).max())
        trailing_stop = round(highest_close - ATR_MULT * atr14, 2)
        last_close    = float(df['close'].iloc[-1])

        return {
            'atr14':         round(atr14, 2),
            'highest_close': round(highest_close, 2),
            'trailing_stop': trailing_stop,
            'last_close':    round(last_close, 2),
        }
    except Exception as e:
        return {'error': str(e)}

# ── 主流程 ───────────────────────────────────────────────────────────────────
def main():
    session = get_market_session()
    et = timezone(timedelta(hours=-4))
    now_et  = datetime.now(et).strftime('%Y-%m-%d %H:%M ET')
    now_tw  = datetime.now().strftime('%Y-%m-%d %H:%M 台灣')

    print("=" * 72)
    print("  🌅 Pre-market 動態止損監控")
    print(f"  {now_tw}  /  {now_et}")
    print(f"  市場狀態: {session.upper()}")
    print("=" * 72)

    if session == 'closed':
        print("\n  ⏸  美股市場已休市，盤前監控僅在 4:00–9:30 ET（台灣 16:00–21:30）執行。")
        print("  （仍會顯示最後收盤與止損狀態供參考）\n")

    alerts   = []
    safe     = []
    errors   = []

    for h in HOLDINGS:
        sym  = h['symbol']
        name = h['name']
        qty  = h['qty']
        cost = h['cost']

        # 動態止損
        ts_data = calc_trailing_stop(sym)
        if 'error' in ts_data or not ts_data:
            errors.append({'sym': sym, 'name': name, 'msg': ts_data.get('error', '無法取得資料')})
            continue

        # 最新價
        price, src = get_latest_price(sym)
        if price is None:
            price = ts_data['last_close']
            src   = 'last-close'

        ts   = ts_data['trailing_stop']
        atr  = ts_data['atr14']
        high = ts_data['highest_close']

        breached  = price < ts
        pnl_pct   = (price - cost) / cost * 100
        pnl_usd   = (price - cost) * qty
        gap_to_ts = price - ts
        gap_pct   = gap_to_ts / price * 100

        row = {
            'sym':    sym,
            'name':   name,
            'price':  price,
            'src':    src,
            'ts':     ts,
            'atr':    atr,
            'high':   high,
            'pnl':    pnl_usd,
            'pnl_pct':pnl_pct,
            'gap':    gap_to_ts,
            'gap_pct':gap_pct,
        }

        if breached:
            alerts.append(row)
        else:
            safe.append(row)

    # ── 輸出 ──────────────────────────────────────────────────────────────────
    def fmt_row(r, is_alert: bool):
        icon   = '🚨' if is_alert else '✅'
        status = f"⚠️  已跌破止損  缺口:{r['gap']:+.2f} ({r['gap_pct']:+.1f}%)" if is_alert \
                 else f"安全  距止損: +{r['gap']:.2f} (+{r['gap_pct']:.1f}%)"
        pnl_s  = f"{'🟢' if r['pnl']>=0 else '🔴'} {r['pnl']:+,.0f} USD ({r['pnl_pct']:+.1f}%)"
        src_s  = f"[{r['src']}]"
        print(f"  {icon} {r['sym']:<8} {r['name']:<14}  "
              f"${r['price']:>9.2f} {src_s:<15}  "
              f"止損:${r['ts']:>8.2f}  {status}")
        print(f"       損益: {pnl_s}   ATR:{r['atr']:.2f}  近{LOOKBACK_DAYS}日高點:${r['high']:.2f}")

    if alerts:
        print(f"\n{'─'*72}")
        print(f"  🚨 止損警示  ({len(alerts)} 檔已跌破動態止損)  → 考慮減碼或出場")
        print(f"{'─'*72}")
        for r in sorted(alerts, key=lambda x: x['gap_pct']):
            fmt_row(r, is_alert=True)

    if safe:
        print(f"\n{'─'*72}")
        print(f"  ✅ 持倉安全  ({len(safe)} 檔未觸發止損)")
        print(f"{'─'*72}")
        for r in sorted(safe, key=lambda x: x['gap_pct']):
            fmt_row(r, is_alert=False)

    if errors:
        print(f"\n  ⚠️  資料錯誤  ({len(errors)} 檔):")
        for e in errors:
            print(f"     • {e['sym']} {e['name']}: {e['msg']}")

    print(f"\n{'─'*72}")
    print(f"  📌 提醒：動態止損為參考警示，非硬性出場線。")
    print(f"           真正出場訊號：儀表板週線MACD跌破0")
    print(f"  公式：止損 = 近{LOOKBACK_DAYS}日最高收盤 - {ATR_MULT} × ATR{ATR_PERIOD}")
    print(f"{'─'*72}\n")

    # 盤前緊急警示
    if session == 'pre-market' and alerts:
        print("🚨 " + "=" * 68)
        print("  CRITICAL: 盤前已跌破止損！建議盤前或開盤立即評估出場計畫：")
        for r in alerts:
            print(f"  • {r['sym']} 現價 ${r['price']:.2f}，止損 ${r['ts']:.2f}，"
                  f"跌破 {abs(r['gap']):.2f} ({abs(r['gap_pct']):.1f}%)")
        print("🚨 " + "=" * 68 + "\n")

if __name__ == '__main__':
    main()
