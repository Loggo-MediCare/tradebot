"""
Fix Hong Kong ticker symbols - remove leading zeros for Yahoo Finance
"""
import re
from pathlib import Path

def fix_hk_ticker(file_path, old_ticker, new_ticker):
    """Fix HK ticker in a file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace ticker symbols (keep quotes)
    content = content.replace(f"'{old_ticker}.HK'", f"'{new_ticker}.HK'")
    content = content.replace(f'"{old_ticker}.HK"', f'"{new_ticker}.HK"')

    # Update model filenames
    content = content.replace(f'ppo_{old_ticker}_hk', f'ppo_{new_ticker}_hk')

    # Update accuracy display
    content = content.replace(f"'{old_ticker}.HK'", f"'{new_ticker}.HK'")

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    """Fix HK tickers in all relevant files"""
    print("=" * 80)
    print("Fixing Hong Kong Ticker Symbols")
    print("=" * 80)

    fixes = [
        {
            'file': 'get_trading_signal_01810.py',
            'old': '01810',
            'new': '1810',
            'name': 'Xiaomi'
        },
        {
            'file': 'get_trading_signal_02202.py',
            'old': '02202',
            'new': '2202',
            'name': 'Vanke'
        }
    ]

    for fix in fixes:
        file_path = Path('.') / fix['file']
        if file_path.exists():
            fix_hk_ticker(file_path, fix['old'], fix['new'])
            print(f"[OK] Fixed {fix['file']}")
            print(f"     {fix['old']}.HK -> {fix['new']}.HK ({fix['name']})")
        else:
            print(f"[SKIP] {fix['file']} not found")

    # Also fix training scripts if they exist
    training_files = [
        'train_01810_hk_improved.py',
        'train_02202_hk_improved.py'
    ]

    for train_file in training_files:
        file_path = Path('.') / train_file
        if file_path.exists():
            if '01810' in train_file:
                fix_hk_ticker(file_path, '01810', '1810')
                print(f"[OK] Fixed {train_file}")
            elif '02202' in train_file:
                fix_hk_ticker(file_path, '02202', '2202')
                print(f"[OK] Fixed {train_file}")

    print("\n" + "=" * 80)
    print("Summary - Correct Ticker Formats")
    print("=" * 80)
    print("Xiaomi:  01810.HK ❌ -> 1810.HK ✅")
    print("Vanke:   02202.HK ❌ -> 2202.HK ✅")
    print("=" * 80)
    print("\nModel files needed (with correct symbols):")
    print("  - ppo_1810_hk_improved.zip")
    print("  - ppo_2202_hk_improved.zip")
    print("=" * 80)

if __name__ == "__main__":
    main()
