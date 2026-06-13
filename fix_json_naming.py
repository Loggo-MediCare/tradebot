"""
Fix feature importance JSON file naming
Change from XXXX_TW_feature_importance.json to XXXX.TW_feature_importance.json
"""
import os
import re
from pathlib import Path

def fix_json_filenames():
    """Rename JSON files to use dots instead of underscores in ticker"""

    print("=" * 80)
    print("Fixing Feature Importance JSON File Names")
    print("=" * 80)

    # Pattern to match files like: 1101_TW_feature_importance.json
    # We want to rename them to: 1101.TW_feature_importance.json
    pattern = re.compile(r'^(\d+|[A-Z]+)_(TW|TWO|HK)_feature_importance\.json$')

    files = list(Path('.').glob('*_feature_importance.json'))
    renamed_count = 0
    skipped_count = 0

    for file_path in files:
        filename = file_path.name
        match = pattern.match(filename)

        if match:
            ticker = match.group(1)
            suffix = match.group(2)
            new_filename = f"{ticker}.{suffix}_feature_importance.json"

            # Check if file already has correct name (with dot)
            if filename == new_filename:
                print(f"[SKIP] {filename} - already correct")
                skipped_count += 1
                continue

            # Check if target file already exists
            if Path(new_filename).exists():
                print(f"[SKIP] {filename} - target {new_filename} already exists")
                skipped_count += 1
                continue

            # Rename the file
            try:
                os.rename(filename, new_filename)
                print(f"[OK] {filename} -> {new_filename}")
                renamed_count += 1
            except Exception as e:
                print(f"[ERROR] Failed to rename {filename}: {e}")
        else:
            print(f"[SKIP] {filename} - doesn't match pattern")
            skipped_count += 1

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Renamed: {renamed_count} files")
    print(f"Skipped: {skipped_count} files")
    print(f"Total: {len(files)} files")
    print("=" * 80)

if __name__ == "__main__":
    fix_json_filenames()
