"""
Check which trained stocks have get_trading_signal files
"""
import os
from pathlib import Path

# All stocks trained today (from the 8 batches)
trained_stocks = [
    # Batch 1
    '3260', '6239', 'apld', '02202', '2357', 'nem', '3037', '6446',
    # Batch 2
    '2362', '2313', '2367', '2454', '2049', '3138', '3491', '4916',
    '4540', '6285', '3711', '7717', '2002', '6770', '1605', '8112',
    '2014', '2368', '3028', '3019', '2451', '3060', '6962', '2889',
    '5386', '2010', '2633',
    # Batch 3
    '2303', '1303', '2887', '2883', '2610', '1301', '1802', '1326',
    '2337', '2382', '1314', '2308', '6548', '2618', '2855',
    # Batch 4
    '6949', '2408', '3081', '8046', '3189', '5864', '5347', '2344',
    '2485', '4533', '7709', '3006', '4534', '8299',
    # Batch 5
    '1717', '3049', '8096', '3149', '3532', '1789', '3691',
    # Batch 6
    '6415', '2740', '7788', '5245', '6510', '8088', '3221', '7610',
    '9103', '6588', '7777', '6265',
    # Batch 7
    '6988', '1809', '6485', '6443', '2431', '4561', '6683', '2484',
    '5289', '5228', '4958', '6515', '8438', '8436', '2486', '6492',
    '3693', '3051', '5340',
    # Batch 8
    '4967', '4563', '4973', '3135', '5328', '3432', '2233', '3555',
    '6236', '8104', '8155', '6442', '6015', '6862', '3576',
]

# Remove duplicates and sort
trained_stocks = sorted(list(set(trained_stocks)))

print("=" * 80)
print("Checking trained stocks for get_trading_signal files")
print("=" * 80)
print(f"Total unique stocks trained: {len(trained_stocks)}\n")

# Find all existing get_trading_signal files
current_dir = Path('.')
existing_files = list(current_dir.glob('get_trading_signal_*.py'))
existing_stocks = set()

for file in existing_files:
    # Extract stock symbol from filename
    name = file.stem.replace('get_trading_signal_', '')
    # Skip backup files
    if '_bak' not in name and name != 'template':
        existing_stocks.add(name)

print(f"Existing get_trading_signal files: {len(existing_stocks)}\n")

# Check which trained stocks are missing signal files
missing_stocks = []
for stock in trained_stocks:
    stock_lower = stock.lower()
    # Check for exact match or with market suffix
    if stock_lower not in existing_stocks:
        missing_stocks.append(stock)

if missing_stocks:
    print("=" * 80)
    print(f"Missing get_trading_signal files for {len(missing_stocks)} stocks:")
    print("=" * 80)
    for stock in missing_stocks:
        print(f"  - {stock}")
else:
    print("[OK] All trained stocks have get_trading_signal files!")

# Find model files to verify which ones actually exist
print("\n" + "=" * 80)
print("Checking for trained model files (ppo_*.zip)")
print("=" * 80)

model_files = list(current_dir.glob('ppo_*.zip'))
trained_models = set()
for model_file in model_files:
    # Extract stock symbol from model filename
    name = model_file.stem.replace('ppo_', '').replace('_improved', '')
    trained_models.add(name)

print(f"Found {len(model_files)} model files\n")

# Check which models have signal files
models_without_signals = []
for model in trained_models:
    if model.lower() not in existing_stocks:
        models_without_signals.append(model)

if models_without_signals:
    print("=" * 80)
    print(f"Models without signal files: {len(models_without_signals)}")
    print("=" * 80)
    for model in sorted(models_without_signals):
        print(f"  - {model}")
else:
    print("[OK] All models have corresponding signal files!")

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"Trained stocks (from batches): {len(trained_stocks)}")
print(f"Existing signal files:         {len(existing_stocks)}")
print(f"Model files found:             {len(model_files)}")
print(f"Missing signal files:          {len(missing_stocks)}")
print(f"Models without signals:        {len(models_without_signals)}")
print("=" * 80)
