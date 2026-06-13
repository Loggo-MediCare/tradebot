"""
Create missing signal files for newly trained stocks
"""
import shutil
from pathlib import Path

def create_signal_file(stock_code, suffix, template_file):
    """Create a signal file from template"""

    output_file = Path(f'get_trading_signal_{stock_code}.py')

    if output_file.exists():
        print(f"[SKIP] {output_file.name} already exists")
        return False

    # Read template
    with open(template_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract template stock code from filename
    template_stock = template_file.stem.replace('get_trading_signal_', '')
    template_suffix = 'TW' if '.TW' in str(template_file) else 'TWO'

    # Replace stock code
    content = content.replace(f"'{template_stock}.{template_suffix}'", f"'{stock_code}.{suffix}'")
    content = content.replace(f'"{template_stock}.{template_suffix}"', f'"{stock_code}.{suffix}"')

    # Replace model filename
    template_model = f'ppo_{template_stock.lower()}_{template_suffix.lower()}_improved'
    new_model = f'ppo_{stock_code.lower()}_{suffix.lower()}_improved'
    content = content.replace(template_model, new_model)

    # Replace in DynamicWeightCalculator calls
    content = content.replace(f"DynamicWeightCalculator('{template_stock}.{template_suffix}')",
                             f"DynamicWeightCalculator('{stock_code}.{suffix}')")
    content = content.replace(f'get_model_accuracy_display("{template_stock}.{template_suffix}")',
                             f'get_model_accuracy_display("{stock_code}.{suffix}")')
    content = content.replace(f"get_model_accuracy_display('{template_stock}.{template_suffix}')",
                             f"get_model_accuracy_display('{stock_code}.{suffix}')")

    # Write output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"[OK] Created {output_file.name}")
    return True

def main():
    """Create missing signal files"""
    print("=" * 80)
    print("Creating Missing Signal Files")
    print("=" * 80)

    # Stocks to create with their suffix
    stocks_to_create = [
        {'code': '1301', 'suffix': 'TW', 'template': 'get_trading_signal_1101.py'},
        {'code': '8069', 'suffix': 'TWO', 'template': 'get_trading_signal_6163.py'},
        {'code': '6285', 'suffix': 'TW', 'template': 'get_trading_signal_6442.py'},
    ]

    created_count = 0

    for stock_info in stocks_to_create:
        template_file = Path(stock_info['template'])

        if not template_file.exists():
            print(f"[ERROR] Template {template_file.name} not found")
            continue

        if create_signal_file(stock_info['code'], stock_info['suffix'], template_file):
            created_count += 1

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Created: {created_count} signal files")
    print("=" * 80)

if __name__ == "__main__":
    main()
