"""
Batch create trading signal files for newly trained stocks
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import re

# Define stock configurations: (filename, ticker, company_name, currency_symbol, model_file)
STOCKS = [
    ('get_trading_signal_2451.py', '2451.TW', '創見資訊', 'NT$', 'ppo_2451_tw_improved.zip'),
    ('get_trading_signal_omer.py', 'OMER', 'Omeros Corporation', '$', 'ppo_omer_improved.zip'),
    ('get_trading_signal_3661.py', '3661.TW', '世芯-KY', 'NT$', 'ppo_3661_tw_improved.zip'),
    ('get_trading_signal_6781.py', '6781.TW', 'AES-KY', 'NT$', 'ppo_6781_tw_improved.zip'),
    ('get_trading_signal_alab.py', 'ALAB', 'Astera Labs Inc', '$', 'ppo_alab_improved.zip'),
    ('get_trading_signal_rhm.py', 'RHM.DE', 'Rheinmetall AG', '€', 'ppo_rhm_de_improved.zip'),
    ('get_trading_signal_nat.py', 'NAT', 'Nordic American Tankers', '$', 'ppo_nat_improved.zip'),
    ('get_trading_signal_7769.py', '7769.TW', '霖揚', 'NT$', 'ppo_7769_tw_improved.zip'),
    ('get_trading_signal_3653.py', '3653.TW', '健策', 'NT$', 'ppo_3653_tw_improved.zip'),
    ('get_trading_signal_2360.py', '2360.TW', '致茂', 'NT$', 'ppo_2360_tw_improved.zip'),
    ('get_trading_signal_htgc.py', 'HTGC', 'Hercules Capital Inc', '$', 'ppo_htgc_improved.zip'),
]

def update_signal_file(filename, ticker, company_name, currency, model_file):
    """Update a signal file with the correct ticker and company name"""
    print(f"Updating {filename}...")

    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace all NVDA references with the new ticker
    content = re.sub(r'nvda \(NVIDIA\)', f'{ticker.split(".")[0]} ({company_name})', content, flags=re.IGNORECASE)
    content = re.sub(r"'NVDA'", f"'{ticker}'", content)
    content = re.sub(r'"NVDA"', f'"{ticker}"', content)
    content = re.sub(r'NVDA', ticker, content)

    # Replace model path
    content = re.sub(r'ppo_nvda_improved\.zip', model_file, content)

    # Replace currency symbol (only in price displays)
    content = re.sub(r'当前价格: \$', f'当前价格: {currency}', content)
    content = re.sub(r'价格: \$', f'价格: {currency}', content)

    # Update company name in final output
    content = re.sub(r'\(NVIDIA\)', f'({company_name})', content)

    # Write back
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"[OK] {filename} updated successfully")

if __name__ == '__main__':
    print("=" * 70)
    print("Batch Creating Trading Signal Files")
    print("=" * 70)

    for filename, ticker, company, currency, model in STOCKS:
        try:
            update_signal_file(filename, ticker, company, currency, model)
        except Exception as e:
            print(f"[ERROR] Error updating {filename}: {e}")

    print("\n" + "=" * 70)
    print("[OK] All signal files updated!")
    print("=" * 70)
