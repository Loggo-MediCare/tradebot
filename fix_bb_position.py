"""
Fix missing bb_position in all training files
"""
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

train_files = [
    'train_1519_taiwan_improved.py',
    'train_2317_taiwan_improved.py',
    'train_2330_taiwan_improved.py',
    'train_2337_taiwan_improved.py',
    'train_2344_taiwan_improved.py',
    'train_2408_taiwan_improved.py',
    'train_3711_taiwan_improved.py',
    'train_3715_taiwan_improved.py',
    'train_4991_taiwan_improved.py',
    'train_6175_taiwan_improved.py',
    'train_6269_taiwan_improved.py',
    'train_6515_taiwan_improved.py',
    'train_6770_taiwan_improved.py',
    'train_8131_taiwan_improved.py',
    'train_aapl_improved.py',
    'train_avgo_improved.py',
    'train_goog_improved.py',
    'train_mu_improved.py',
    'train_nvda_improved.py',
]

print("Fixing bb_position in all training files...")

for filename in train_files:
    filepath = os.path.join(r'C:\Users\Silvi\Projects\trading-bot', filename)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    if "df['bb_position']" in content:
        print(f"OK {filename}: bb_position already exists")
    else:
        print(f"FIXING {filename}: adding bb_position")

        # Add bb_position after bb_lower
        if "df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)" in content:
            content = content.replace(
                "df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)",
                "df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)\n    bb_range = df['bb_upper'] - df['bb_lower']\n    df['bb_position'] = ((df['close'] - df['bb_lower']) / bb_range * 100).fillna(50)"
            )

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  -> Fixed {filename}")

print("\nDone!")
