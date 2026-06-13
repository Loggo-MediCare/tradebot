"""
Update run_all_western.py with newly created US stocks
"""
import re
from pathlib import Path

def main():
    """Add new US stocks to run_all_western.py"""

    # New US stocks created today (5 stocks)
    new_stocks = [
        {'file': 'get_trading_signal_amd.py', 'name': 'AMD Advanced Micro Devices'},
        {'file': 'get_trading_signal_apld.py', 'name': 'APLD Applied Digital'},
        {'file': 'get_trading_signal_gild.py', 'name': 'GILD Gilead Sciences'},
        {'file': 'get_trading_signal_mrna.py', 'name': 'MRNA Moderna'},
        {'file': 'get_trading_signal_nem.py', 'name': 'NEM Newmont'},
    ]

    # Read current file
    file_path = Path('.') / 'run_all_western.py'
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the line with tsla (last US stock before European section)
    # Insert new stocks after tsla and before the European Stocks comment

    pattern = r"(\{'file': 'get_trading_signal_tsla\.py', 'name': 'tesla'\},\s*\n\s*\n\s*# European Stocks)"

    # Create the new entries string
    new_entries = ""
    for stock in new_stocks:
        new_entries += f"    {{'file': '{stock['file']}', 'name': '{stock['name']}'}},\n"

    # Insert the new entries
    content = re.sub(
        pattern,
        f"    {{'file': 'get_trading_signal_tsla.py', 'name': 'tesla'}},\n{new_entries}\n    # European Stocks",
        content
    )

    # Update the count in the print statement
    # From "US: 23, EU: 1" to "US: 28, EU: 1"
    content = re.sub(
        r'\(US: \d+, EU: \d+\)',
        '(US: 28, EU: 1)',
        content
    )

    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("=" * 80)
    print("Updated run_all_western.py")
    print("=" * 80)
    print(f"Added {len(new_stocks)} new US stocks:")
    for stock in new_stocks:
        print(f"  + {stock['name']}")
    print("=" * 80)
    print(f"Total stocks in run_all_western.py: 29 (was 24)")
    print(f"  US stocks: 28 (was 23)")
    print(f"  EU stocks: 1")
    print("=" * 80)

if __name__ == "__main__":
    main()
