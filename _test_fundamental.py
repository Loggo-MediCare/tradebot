import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import yfinance as yf, warnings, numpy as np
warnings.filterwarnings('ignore')

# ── /fundamental 2454 ────────────────────────────────────────────
print('=== /fundamental 2454 ===')
t = yf.Ticker('2454.TW')
info = t.info
price  = info.get('currentPrice') or info.get('regularMarketPrice', 0)
pe     = info.get('trailingPE', 0) or 0
pb     = info.get('priceToBook', 0) or 0
roe    = (info.get('returnOnEquity') or 0) * 100
margin = (info.get('grossMargins') or 0) * 100
net_m  = (info.get('profitMargins') or 0) * 100
mktcap = info.get('marketCap', 0) or 0
tgt    = info.get('targetMeanPrice', 0) or 0
rating = info.get('recommendationKey', 'N/A')
fcf    = info.get('freeCashflow', 0) or 0
div    = (info.get('dividendYield') or 0) * 100

print(f'  股票: 2454.TW 聯發科')
print(f'  當前價格:   NT${price:,.0f}  |  分析師目標: NT${tgt:,.0f}  |  評級: {rating}')
print(f'  本益比P/E:  {pe:.1f}x    |  股價淨值P/B: {pb:.1f}x')
print(f'  ROE:        {roe:.1f}%   |  毛利率: {margin:.1f}%   |  淨利率: {net_m:.1f}%')
print(f'  市值:       NT${mktcap/1e9:.0f}B')
print(f'  自由現金流: NT${fcf/1e9:.1f}B   |  殖利率: {div:.1f}%')

if fcf > 0:
    wacc = 0.10; g = 0.12; term_g = 0.03; yrs = 10
    dcf  = sum([fcf*(1+g)**i / (1+wacc)**i for i in range(1, yrs+1)])
    term = fcf*(1+g)**yrs*(1+term_g) / ((wacc-term_g)*(1+wacc)**yrs)
    shares    = info.get('sharesOutstanding', 1) or 1
    intrinsic = (dcf + term) / shares
    updown    = (intrinsic - price) / price * 100
    verdict   = '低估 🟢' if updown > 10 else ('高估 🔴' if updown < -10 else '合理 🟡')
    print(f'  DCF內在價值: NT${intrinsic:,.0f}  ({updown:+.1f}% vs 現價)  → {verdict}')

# ── /compare AMD MU INTC ─────────────────────────────────────────
print()
print('=== /compare AMD MU INTC ===')
cols = ['AMD', 'MU', 'INTC']
rows = []
for sym in cols:
    i = yf.Ticker(sym).info
    rows.append({
        'sym':    sym,
        'price':  i.get('currentPrice') or i.get('regularMarketPrice', 0),
        'pe':     i.get('trailingPE', 0) or 0,
        'pb':     i.get('priceToBook', 0) or 0,
        'roe':    (i.get('returnOnEquity') or 0) * 100,
        'margin': (i.get('grossMargins') or 0) * 100,
        'mktcap': (i.get('marketCap', 0) or 0) / 1e9,
        'rating': i.get('recommendationKey', 'NA'),
        'tgt':    i.get('targetMeanPrice', 0) or 0,
    })

hdr = f"  {'指標':<14}  {'AMD':>12}  {'MU':>12}  {'INTC':>12}"
print(hdr)
print('  ' + '-'*54)
fields = [
    ('price',  '價格($)',   lambda v: f'${v:>10.2f}'),
    ('pe',     'P/E(x)',    lambda v: f'{v:>11.1f}x'),
    ('pb',     'P/B(x)',    lambda v: f'{v:>11.1f}x'),
    ('roe',    'ROE%',      lambda v: f'{v:>11.1f}%'),
    ('margin', '毛利率%',   lambda v: f'{v:>11.1f}%'),
    ('mktcap', '市值($B)',  lambda v: f'${v:>9.0f}B'),
    ('tgt',    '目標價($)', lambda v: f'${v:>10.2f}'),
    ('rating', '評級',      lambda v: f'{str(v):>12}'),
]
for key, label, fmt in fields:
    parts = '  '.join(fmt(r[key]) for r in rows)
    print(f'  {label:<14}  {parts}')

best_pe  = min(rows, key=lambda x: x['pe'] if x['pe'] > 0 else 9999)
best_roe = max(rows, key=lambda x: x['roe'])
print(f'\n  🏆 最低P/E: {best_pe["sym"]}  |  最高ROE: {best_roe["sym"]}')
