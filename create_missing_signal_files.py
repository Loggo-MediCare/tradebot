"""
Create get_trading_signal files for all trained models
"""
import os
import re
from pathlib import Path

def get_stock_info(model_name):
    """Extract stock info from model name"""
    # Remove ppo_ prefix and _improved suffix
    stock = model_name.replace('ppo_', '').replace('_improved', '')

    # Determine if it's US or TW stock
    if any(suffix in stock.lower() for suffix in ['_tw', '_two', '_taiwan', '_t']):
        # Taiwan stock
        base_stock = re.sub(r'_(tw|two|taiwan|t)$', '', stock, flags=re.IGNORECASE)
        market = 'TW'
        symbol_suffix = '.TW'
    elif stock.lower() in ['aapl', 'amd', 'gild', 'mrna', 'nem', 'apld', 'rhm']:
        # US stock
        base_stock = stock.upper()
        market = 'US'
        symbol_suffix = ''
    elif '_de' in stock.lower():
        # German stock
        base_stock = stock.replace('_de', '').upper()
        market = 'DE'
        symbol_suffix = '.DE'
    elif stock == 'trading_model' or stock == 'aapl_model':
        return None  # Skip generic models
    else:
        # Default to Taiwan
        base_stock = stock
        market = 'TW'
        symbol_suffix = '.TW'

    return {
        'base': base_stock,
        'market': market,
        'symbol': base_stock + symbol_suffix,
        'model_name': f'ppo_{stock}_improved'
    }

def create_signal_file(stock_info, template_file):
    """Create a signal file from template"""
    base_stock = stock_info['base']
    market = stock_info['market']
    symbol = stock_info['symbol']
    model_name = stock_info['model_name']

    # Read template
    with open(template_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Get template stock info from template filename
    template_name = template_file.stem.replace('get_trading_signal_', '')

    # Replace stock symbols and model names
    if market == 'US':
        # For US stocks, replace with uppercase symbol
        content = re.sub(
            r"'" + template_name.upper() + r"'",
            f"'{base_stock}'",
            content,
            flags=re.IGNORECASE
        )
        content = re.sub(
            r'symbol.*?=.*?["\']' + template_name.upper() + r'["\']',
            f"symbol = '{base_stock}'",
            content,
            flags=re.IGNORECASE
        )
    else:
        # For TW stocks, replace number
        content = re.sub(
            r"'" + template_name + r"\.TW'",
            f"'{symbol}'",
            content
        )
        content = re.sub(
            r'"' + template_name + r'\.TW"',
            f'"{symbol}"',
            content
        )

    # Replace model filenames
    template_model = f'ppo_{template_name}_improved'
    content = content.replace(template_model, model_name)
    content = content.replace(f'"{template_model}"', f'"{model_name}"')
    content = content.replace(f"'{template_model}'", f"'{model_name}'")

    # Update function name for weight calculator and accuracy tracker
    content = re.sub(
        r"DynamicWeightCalculator\(['\"]" + template_name + r"['\"]",
        f"DynamicWeightCalculator('{base_stock}'",
        content,
        flags=re.IGNORECASE
    )
    content = re.sub(
        r"get_model_accuracy_display\(['\"]" + template_name + r"['\"]",
        f"get_model_accuracy_display('{symbol}'",
        content,
        flags=re.IGNORECASE
    )

    # Write new file
    output_file = Path('.') / f'get_trading_signal_{base_stock.lower()}.py'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

    return output_file

def main():
    """Main function"""
    print("=" * 80)
    print("Creating get_trading_signal files for trained models")
    print("=" * 80)

    # Find all model files
    current_dir = Path('.')
    model_files = list(current_dir.glob('ppo_*.zip'))

    # Get existing signal files
    existing_files = {f.stem.replace('get_trading_signal_', '').lower()
                     for f in current_dir.glob('get_trading_signal_*.py')
                     if '_bak' not in f.name}

    # Templates
    tw_template = Path('.') / 'get_trading_signal_6442.py'  # Taiwan stock template
    us_template = Path('.') / 'get_trading_signal_nvda.py'  # US stock template

    if not tw_template.exists() or not us_template.exists():
        print("[ERROR] Template files not found!")
        return

    print(f"Found {len(model_files)} model files")
    print(f"Found {len(existing_files)} existing signal files\n")

    created_count = 0
    skipped_count = 0
    error_count = 0

    for model_file in sorted(model_files):
        model_name = model_file.stem
        stock_info = get_stock_info(model_name)

        if not stock_info:
            continue

        base_stock_lower = stock_info['base'].lower()

        # Check if signal file already exists
        if base_stock_lower in existing_files:
            skipped_count += 1
            continue

        try:
            # Choose template based on market
            template = us_template if stock_info['market'] == 'US' else tw_template

            # Create signal file
            output_file = create_signal_file(stock_info, template)
            print(f"[OK] Created {output_file.name} (from {template.name})")
            created_count += 1

        except Exception as e:
            print(f"[ERROR] Failed to create signal for {stock_info['base']}: {e}")
            error_count += 1

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Model files:        {len(model_files)}")
    print(f"Created:            {created_count}")
    print(f"Already existed:    {skipped_count}")
    print(f"Errors:             {error_count}")
    print("=" * 80)

    if created_count > 0:
        print(f"\n[OK] Successfully created {created_count} new signal files!")
    else:
        print("\n[INFO] No new signal files needed")

if __name__ == "__main__":
    main()
