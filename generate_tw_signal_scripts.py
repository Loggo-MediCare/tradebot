"""
Generate get_trading_signal_*.py for new TW stocks using 2330 as template,
then add them to run_all_local_tw_to_excel.py
"""
import os
import re

TEMPLATE_FILE = 'get_trading_signal_2330.py'

# Stocks that need new signal files created
NEW_STOCKS = [
    ('2412', '2412.TW', '中華電信'),
    ('6274', '6274.TW', '台燿'),
    ('8112', '8112.TW', '至上'),
    ('2049', '2049.TW', '上銀'),
    ('1785', '1785.TW', '光洋科'),
    ('6531', '6531.TW', '愛普'),
    ('2395', '2395.TW', '研華'),
    ('4749', '4749.TW', '不二家'),
    ('3131', '3131.TW', '弘塑'),
]

# All 14 stocks to add to run_all_local_tw_to_excel.py (only add ones not already present)
ALL_14_STOCKS = [
    ('2412', '2412.TW', '中華電信'),
    ('2603', '2603.TW', '長榮海運'),
    ('6274', '6274.TW', '台燿'),
    ('8112', '8112.TW', '至上'),
    ('2382', '2382.TW', '廣達電腦'),
    ('2049', '2049.TW', '上銀'),
    ('1785', '1785.TW', '光洋科'),
    ('3017', '3017.TW', '奇鋐'),
    ('6531', '6531.TW', '愛普'),
    ('2395', '2395.TW', '研華'),
    ('3037', '3037.TW', '欣興'),
    ('4749', '4749.TW', '不二家'),
    ('3131', '3131.TW', '弘塑'),
    ('3715', '3715.TW', '定穎投控'),
]

def generate_signal_script(symbol, ticker, name, template_content):
    content = template_content

    # Replace model path: ppo_2330_tw_improved -> ppo_{symbol}_improved
    content = content.replace('ppo_2330_tw_improved', f'ppo_{symbol}_improved')

    # Replace all occurrences of '2330.TW' with new ticker
    content = content.replace("'2330.TW'", f"'{ticker}'")
    content = content.replace('"2330.TW"', f'"{ticker}"')

    # Replace standalone '2330' (not part of longer number) with symbol
    # Using word boundary approach - replace '2330' in strings carefully
    content = content.replace("'2330'", f"'{symbol}'")
    content = content.replace('"2330"', f'"{symbol}"')

    # Replace Chinese name 台積電
    content = content.replace('台積電', name)

    # Replace in print statements and comments that use bare 2330
    content = content.replace('2330 (台積電)', f'{symbol} ({name})')
    content = content.replace('(台積電)', f'({name})')

    # Replace docstring header reference
    content = content.replace('台股 2330', f'台股 {symbol}')

    # Fix the summary print line: (台積電) already replaced above
    # Fix: "股票: {result['symbol']} (台積電)" already handled

    return content

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Read template
    template_path = os.path.join(script_dir, TEMPLATE_FILE)
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()

    print("=" * 60)
    print("生成新股票信號腳本")
    print("=" * 60)

    # Generate new signal scripts
    for symbol, ticker, name in NEW_STOCKS:
        output_file = os.path.join(script_dir, f'get_trading_signal_{symbol}.py')
        if os.path.exists(output_file):
            print(f"  ⏭️  {symbol} 已存在，跳過")
            continue

        content = generate_signal_script(symbol, ticker, name, template_content)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ 生成 get_trading_signal_{symbol}.py ({name})")

    # Update run_all_local_tw_to_excel.py
    print("\n更新 run_all_local_tw_to_excel.py...")
    excel_script = os.path.join(script_dir, 'run_all_local_tw_to_excel.py')

    with open(excel_script, 'r', encoding='utf-8') as f:
        excel_content = f.read()

    added = []
    for symbol, ticker, name in ALL_14_STOCKS:
        entry = f"get_trading_signal_{symbol}.py"
        if entry in excel_content:
            print(f"  ⏭️  {symbol} 已在清單中")
            continue

        # Add before the closing bracket of SIGNAL_SCRIPTS
        new_entry = f"    {{'file': 'get_trading_signal_{symbol}.py', 'name': '{symbol} {name}'}},\n"
        # Insert before the last ] of SIGNAL_SCRIPTS list
        # Find the last entry line and insert after it
        insert_marker = "]  # END_SIGNAL_SCRIPTS"
        if insert_marker in excel_content:
            excel_content = excel_content.replace(insert_marker, new_entry + insert_marker)
        else:
            # Find the end of SIGNAL_SCRIPTS list - look for the closing ]
            # Insert after the last {'file': ...} entry
            last_entry_pattern = r"(\s*\{'file': '[^']+\.py', 'name': '[^']+'\},?\s*\]\s*)"
            matches = list(re.finditer(r"(\s*\{'file': '[^']+\.py', 'name': '[^']+'\},?)", excel_content))
            if matches:
                last_match = matches[-1]
                insert_pos = last_match.end()
                excel_content = excel_content[:insert_pos] + '\n' + new_entry.rstrip('\n') + excel_content[insert_pos:]
        added.append(symbol)
        print(f"  ✅ 添加 {symbol} ({name})")

    with open(excel_script, 'w', encoding='utf-8') as f:
        f.write(excel_content)

    print(f"\n完成! 添加了 {len(added)} 支股票到清單")

if __name__ == '__main__':
    main()
