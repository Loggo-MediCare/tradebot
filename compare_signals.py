"""
Compare two signal output files and report any differences in signals.
Usage: python compare_signals.py <old_file> <new_file>
"""
import sys
import re

def extract_signals(filepath):
    """Extract {ticker: (price, signal)} from a signal output file."""
    results = {}
    current_ticker = None
    with open(filepath, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            # Match ticker line: "   股票: AAPL (Apple)"  or  "   股票: AMD"
            m = re.match(r'股票:\s+([A-Z0-9]+)', line)
            if m:
                current_ticker = m.group(1)
                continue
            # Match price line
            m = re.match(r'价格:\s+\$?([\d,\.]+)', line)
            if m and current_ticker:
                price = m.group(1).replace(',', '')
                if current_ticker not in results:
                    results[current_ticker] = {'price': price, 'signal': ''}
                else:
                    results[current_ticker]['price'] = price
                continue
            # Match signal line
            m = re.match(r'信号:\s+(.+)', line)
            if m and current_ticker:
                signal = m.group(1).strip()
                if current_ticker not in results:
                    results[current_ticker] = {'price': '', 'signal': signal}
                else:
                    results[current_ticker]['signal'] = signal
    return results

def signal_category(signal):
    """Simplify signal to BUY / SELL / HOLD / WAIT."""
    s = signal.upper()
    if 'BUY' in s:
        return 'BUY'
    if 'SELL' in s:
        return 'SELL'
    if 'WAIT' in s or '觀望' in s or '观望' in s:
        return 'WAIT'
    return 'HOLD'

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_signals.py <old_file> <new_file>")
        sys.exit(1)

    old_file, new_file = sys.argv[1], sys.argv[2]
    old = extract_signals(old_file)
    new = extract_signals(new_file)

    all_tickers = sorted(set(list(old.keys()) + list(new.keys())))

    changed = []
    unchanged = []
    only_old = []
    only_new = []

    for ticker in all_tickers:
        if ticker in old and ticker not in new:
            only_old.append(ticker)
        elif ticker in new and ticker not in old:
            only_new.append(ticker)
        else:
            o_sig = signal_category(old[ticker]['signal'])
            n_sig = signal_category(new[ticker]['signal'])
            o_price = old[ticker]['price']
            n_price = new[ticker]['price']
            if o_sig != n_sig:
                changed.append((ticker, o_price, old[ticker]['signal'], n_price, new[ticker]['signal']))
            else:
                unchanged.append((ticker, n_price, new[ticker]['signal']))

    print("=" * 70)
    print("信號比較報告")
    print(f"舊檔: {old_file}")
    print(f"新檔: {new_file}")
    print("=" * 70)

    if changed:
        print(f"\n🔄 信號變化 ({len(changed)} 支):")
        print(f"  {'股票':<8} {'舊價格':>10}  {'舊信號':<25} → {'新價格':>10}  {'新信號'}")
        print("  " + "-" * 65)
        for ticker, op, os, np_, ns in changed:
            print(f"  {ticker:<8} ${op:>9}  {os:<25} → ${np_:>9}  {ns}")
    else:
        print("\n✅ 無信號變化")

    print(f"\n📊 不變 ({len(unchanged)} 支):")
    for ticker, price, sig in unchanged:
        emoji = '🟢' if signal_category(sig) == 'BUY' else '🔴' if signal_category(sig) == 'SELL' else '🟡'
        print(f"  {emoji} {ticker:<8} ${price:>10}  {sig}")

    if only_old:
        print(f"\n⬅️  僅舊檔有 ({len(only_old)} 支): {', '.join(only_old)}")
    if only_new:
        print(f"\n➡️  僅新檔有 ({len(only_new)} 支): {', '.join(only_new)}")

    print(f"\n總結: {len(all_tickers)} 支股票  |  變化: {len(changed)}  |  不變: {len(unchanged)}")

if __name__ == '__main__':
    main()
