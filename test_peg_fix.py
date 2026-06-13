"""
快速测试 PEG 比率修复
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import yfinance as yf

def get_peg_ratio(ticker_symbol):
    """获取 PEG 比率"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        # 尝试多个可能的键（US stocks 用 pegRatio，Taiwan/Asia stocks 用 trailingPegRatio）
        peg_ratio = info.get('pegRatio') or info.get('trailingPegRatio')
        return peg_ratio
    except Exception as e:
        return None

print("=" * 80)
print("测试 PEG 比率获取 - 修复后")
print("=" * 80)

# 测试 US 股票
print("\n美股测试:")
print("-" * 80)
us_stocks = ['GOOG', 'NVDA', 'TSLA']
for ticker in us_stocks:
    peg = get_peg_ratio(ticker)
    if peg:
        print(f"{ticker:10s}: PEG = {peg:.2f}")
    else:
        print(f"{ticker:10s}: PEG = N/A")

# 测试台股
print("\n台股测试:")
print("-" * 80)
tw_stocks = [
    ('2330.TW', '台積電'),
    ('2317.TW', '鴻海'),
    ('2454.TW', '聯發科')
]
for ticker, name in tw_stocks:
    peg = get_peg_ratio(ticker)
    if peg:
        print(f"{ticker:15s} {name:10s}: PEG = {peg:.2f}")
    else:
        print(f"{ticker:15s} {name:10s}: PEG = N/A")

print("\n" + "=" * 80)
print("测试完成!")
print("=" * 80)
