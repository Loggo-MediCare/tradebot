"""
Add Hong Kong stocks to run_all_western.py
"""
import re
from pathlib import Path

def main():
    """Add HK stocks to run_all_western.py"""

    # Hong Kong stocks to add
    hk_stocks = [
        {'file': 'get_trading_signal_02202.py', 'name': '02202.HK Vanke'},
        {'file': 'get_trading_signal_01810.py', 'name': '01810.HK Xiaomi'},
    ]

    # Read current file
    file_path = Path('.') / 'run_all_western.py'
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Add HK section after European Stocks
    pattern = r'(    # European Stocks\n    \{\'file\': \'get_trading_signal_rnmby\.py\', \'name\': \'RNMBY Rheinmetall AG\'\},\n)'

    # Create the HK entries
    hk_entries = "\n    # Hong Kong Stocks\n"
    for stock in hk_stocks:
        hk_entries += f"    {{'file': '{stock['file']}', 'name': '{stock['name']}'}},\n"

    # Insert HK stocks
    content = re.sub(
        pattern,
        r'\1' + hk_entries,
        content
    )

    # Update the count (US: 28, EU: 1, HK: 2)
    content = re.sub(
        r'\(US: 28, EU: 1\)',
        '(US: 28, EU: 1, HK: 2)',
        content
    )

    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("=" * 80)
    print("Updated run_all_western.py")
    print("=" * 80)
    print(f"Added {len(hk_stocks)} Hong Kong stocks:")
    for stock in hk_stocks:
        print(f"  + {stock['name']}")
    print("=" * 80)
    print(f"Total stocks: 31 (was 29)")
    print(f"  US: 28")
    print(f"  EU: 1")
    print(f"  HK: 2 (NEW)")
    print("=" * 80)

if __name__ == "__main__":
    main()
