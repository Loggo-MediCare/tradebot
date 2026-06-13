"""
Update run_all_local_tw.py with newly created Taiwan stocks
"""
import re
from pathlib import Path

def main():
    """Add new Taiwan stocks to run_all_local_tw.py"""

    # New Taiwan stocks created today (16 stocks)
    new_stocks = [
        {'file': 'get_trading_signal_2357.py', 'name': '2357 華碩'},
        {'file': 'get_trading_signal_2363.py', 'name': '2363 矽統'},
        {'file': 'get_trading_signal_2367.py', 'name': '2367 燿華'},
        {'file': 'get_trading_signal_2634.py', 'name': '2634 漢翔'},
        {'file': 'get_trading_signal_3004.py', 'name': '3004 豐達科'},
        {'file': 'get_trading_signal_3022.py', 'name': '3022 威強電'},
        {'file': 'get_trading_signal_3037.py', 'name': '3037 欣興'},
        {'file': 'get_trading_signal_3135.py', 'name': '3135 台股'},
        {'file': 'get_trading_signal_3138.py', 'name': '3138 耀登'},
        {'file': 'get_trading_signal_3260.py', 'name': '3260 威剛'},
        {'file': 'get_trading_signal_3491.py', 'name': '3491 台股'},
        {'file': 'get_trading_signal_4967.py', 'name': '4967 台股'},
        {'file': 'get_trading_signal_5371.py', 'name': '5371 台股'},
        {'file': 'get_trading_signal_6446.py', 'name': '6446 藥華藥'},
        {'file': 'get_trading_signal_6668.py', 'name': '6668 台股'},
        {'file': 'get_trading_signal_8222.py', 'name': '8222 台股'},
    ]

    # Read current file
    file_path = Path('.') / 'run_all_local_tw.py'
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the SIGNAL_SCRIPTS list
    # Insert new stocks before the closing bracket

    # Find where the list ends (before the final ']')
    pattern = r'(\]\s*\n\s*def run_signal)'

    # Create the new entries string
    new_entries = ""
    for stock in new_stocks:
        new_entries += f"    {{'file': '{stock['file']}', 'name': '{stock['name']}'}},\n"

    # Insert the new entries
    content = re.sub(
        pattern,
        f"{new_entries}\\1",
        content
    )

    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("=" * 80)
    print("Updated run_all_local_tw.py")
    print("=" * 80)
    print(f"Added {len(new_stocks)} new Taiwan stocks:")
    for stock in new_stocks:
        print(f"  ✓ {stock['name']}")
    print("=" * 80)
    print(f"Total stocks in run_all_local_tw.py: {62 + len(new_stocks)} (was 62)")
    print("=" * 80)

if __name__ == "__main__":
    main()
