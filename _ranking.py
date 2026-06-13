"""
/ranking  —  成交金額排行榜
解析最新 taiwan_signals_output_*.txt，計算 price × volume 並排序
"""
import sys, io, os, re, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── find latest output file ───────────────────────────────────────
base = r"C:\Users\Silvi\Projects\trading-bot"
files = glob.glob(os.path.join(base, "taiwan_signals_output_*.txt"))
if not files:
    print("⚠️  找不到輸出檔，請先執行 run_all_local_tw_to_file.ps1")
    sys.exit(0)

latest = max(files, key=os.path.getmtime)
print(f"📂 解析: {os.path.basename(latest)}\n")

# ── parse all stocks ──────────────────────────────────────────────
with open(latest, encoding='utf-8', errors='ignore') as f:
    content = f.read()

stocks = []

# Split into per-stock blocks by "进度: [N/M]" markers
# Each block contains one stock's full output
blocks = re.split(r'进度:\s*\[\d+/\d+\]', content)

for block in blocks:
    # ── price ─────────────────────────────────────────────────────
    # Standard: "   当前价格: NT$23.80"  or "   当前价格: $503.89"
    pm = re.search(r'当前价格:\s*(?:NT\$|\$)([\d,]+\.?\d*)', block)
    # V2 format: "║  價格: $94.60"
    if not pm:
        pm = re.search(r'[║\|]\s*價格:\s*(?:NT\$|\$)([\d,]+\.?\d*)', block)
    price = float(pm.group(1).replace(',', '')) if pm else None

    # ── volume ────────────────────────────────────────────────────
    vm = re.search(r'今日成交量:\s*([\d,]+)', block)
    volume = int(vm.group(1).replace(',', '')) if vm else None

    # ── symbol ───────────────────────────────────────────────────
    # Standard: "   股票: 2330.TW (台積電)"
    sm = re.search(r'股票:\s*(\S[\S ]*)', block)
    # V2 box: "║  股票: 1303.TW (台積電)"
    if not sm:
        sm = re.search(r'[║\|]\s*股票:\s*(\S[\S ]*)', block)
    symbol = sm.group(1).strip().rstrip('║').strip() if sm else None

    # ── signal ───────────────────────────────────────────────────
    # Use negative lookbehind to skip "交易信號:" (MA50 intermediate)
    # Match only standalone "信号:" / "信號:" in final AI or 快速摘要 sections
    # e.g. "   信号: 观望 (WAIT)"  or  "║  信號: 強烈買入信號  ║"
    all_sigs = re.findall(r'(?<!交易)信[号號]:\s*([^\n║]+)', block)
    signal = all_sigs[-1].strip().rstrip('║').strip() if all_sigs else '—'

    if symbol and price and volume:
        money = price * volume
        stocks.append({
            'symbol': symbol,
            'price':  price,
            'volume': volume,
            'money':  money,
            'signal': signal,
        })

if not stocks:
    print("⚠️  無法解析任何股票資料，請檢查輸出檔格式")
    sys.exit(0)

# ── sort by money desc ────────────────────────────────────────────
ranked = sorted(stocks, key=lambda x: x['money'], reverse=True)

def fmt_money(m):
    if m >= 1e9:  return f"NT${m/1e9:.2f}B"
    if m >= 1e6:  return f"NT${m/1e6:.1f}M"
    if m >= 1e3:  return f"NT${m/1e3:.0f}K"
    return f"NT${m:.0f}"

def sig_emoji(s):
    if 'BUY' in s or '买入' in s or '買入' in s:  return '🟢'
    if 'SELL' in s or '卖出' in s or '賣出' in s or '看空' in s: return '🔴'
    return '🟡'

# ── print ranking table ───────────────────────────────────────────
print(f"{'='*90}")
print(f"  💰 成交金額排行榜  (共解析 {len(stocks)} 支，by {os.path.basename(latest)[:20]})")
print(f"{'='*90}")
print(f"  {'排名':>4}  {'股票':<26}  {'價格':>10}  {'成交量':>14}  {'成交金額':>12}  信號")
print(f"  {'-'*84}")

buy_list  = []
sell_list = []
wait_list = []

for rank, r in enumerate(ranked, 1):
    emoji     = sig_emoji(r['signal'])
    price_str = f"NT${r['price']:,.1f}"
    vol_str   = f"{r['volume']:,}"
    money_str = fmt_money(r['money'])
    sym       = r['symbol'][:26]

    print(f"  {rank:>4}  {sym:<26}  {price_str:>10}  {vol_str:>14}  {money_str:>12}  {emoji} {r['signal']}")

    if 'BUY' in r['signal'] or '买入' in r['signal'] or '買入' in r['signal']:
        buy_list.append(r)
    elif 'SELL' in r['signal'] or '卖出' in r['signal'] or '賣出' in r['signal'] or '看空' in r['signal']:
        sell_list.append(r)
    else:
        wait_list.append(r)

print(f"{'='*90}")

# ── buy summary ───────────────────────────────────────────────────
if buy_list:
    print(f"\n  🟢 買入信號  ({len(buy_list)} 檔，依成交金額排序):")
    for r in buy_list:
        print(f"     • {r['symbol']:<28}  {fmt_money(r['money']):>12}  NT${r['price']:,.1f}")

if sell_list:
    print(f"\n  🔴 賣出信號  ({len(sell_list)} 檔):")
    for r in sell_list[:10]:
        print(f"     • {r['symbol']:<28}  {fmt_money(r['money']):>12}  NT${r['price']:,.1f}")

if wait_list:
    print(f"\n  🟡 觀望/持有  ({len(wait_list)} 檔)")

print()
