"""
Create Hong Kong stock signal files
"""
import re
from pathlib import Path

def create_hk_signal_file(hk_stock, company_name, template_file):
    """Create a Hong Kong stock signal file from template"""

    # Read template
    with open(template_file, 'r', encoding='utf-8') as f:
        content = f.read()

    template_name = template_file.stem.replace('get_trading_signal_', '')

    # Replace stock symbol (6442.TW -> XXXXX.HK)
    content = re.sub(
        r"'" + template_name + r"\.TW'",
        f"'{hk_stock}.HK'",
        content
    )
    content = re.sub(
        r'"' + template_name + r'\.TW"',
        f'"{hk_stock}.HK"',
        content
    )

    # Replace model filename
    template_model = f'ppo_{template_name}_improved'
    new_model = f'ppo_{hk_stock}_hk_improved'
    content = content.replace(template_model, new_model)
    content = content.replace(f'"{template_model}"', f'"{new_model}"')
    content = content.replace(f"'{template_model}'", f"'{new_model}'")

    # Update weight calculator and accuracy tracker
    content = re.sub(
        r"DynamicWeightCalculator\(['\"]" + template_name + r"['\"]",
        f"DynamicWeightCalculator('{hk_stock}'",
        content,
        flags=re.IGNORECASE
    )
    content = re.sub(
        r"get_model_accuracy_display\(['\"]" + template_name + r"\.TW['\"]",
        f"get_model_accuracy_display('{hk_stock}.HK'",
        content,
        flags=re.IGNORECASE
    )

    # Update company name in header
    content = re.sub(
        r'台股 \d+ \([^)]+\)',
        f'港股 {hk_stock} ({company_name})',
        content
    )

    # Write new file
    output_file = Path('.') / f'get_trading_signal_{hk_stock}.py'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

    return output_file

def main():
    """Create Hong Kong stock signal files"""
    print("=" * 80)
    print("Creating Hong Kong Stock Signal Files")
    print("=" * 80)

    # Hong Kong stocks to create
    hk_stocks = [
        {'stock': '02202', 'name': '萬科企業 (Vanke)'},
        {'stock': '01810', 'name': '小米集團 (Xiaomi)'},
    ]

    # Use Taiwan stock template as base
    template = Path('.') / 'get_trading_signal_6442.py'

    if not template.exists():
        print("[ERROR] Template file not found!")
        return

    created_count = 0

    for stock_info in hk_stocks:
        try:
            output_file = create_hk_signal_file(
                stock_info['stock'],
                stock_info['name'],
                template
            )
            print(f"[OK] Created {output_file.name} - {stock_info['name']}")
            created_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to create {stock_info['stock']}: {e}")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Created: {created_count} Hong Kong stock signal files")
    print("=" * 80)
    print("\nNote: Model files needed:")
    print("  - ppo_02202_hk_improved.zip")
    print("  - ppo_01810_hk_improved.zip")
    print("\nOnce models are trained, these signal files will work!")
    print("=" * 80)

if __name__ == "__main__":
    main()
