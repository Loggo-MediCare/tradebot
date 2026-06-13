"""
Add sma_200 calculation to add_technical_indicators function
"""
import re
import sys
import io
from pathlib import Path

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def add_sma_200_to_function(file_path):
    """Add sma_200 calculation to add_technical_indicators function"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        original_lines = lines.copy()
        modified = False

        # Find the add_technical_indicators function
        in_function = False
        for i, line in enumerate(lines):
            # Check if we're entering the function
            if 'def add_technical_indicators' in line:
                in_function = True

            # Look for sma_50 line within the function
            if in_function and "df['sma_50'] = df['close'].rolling(50).mean()" in line:
                # Check if next line already has sma_200
                if i + 1 < len(lines) and 'sma_200' in lines[i + 1]:
                    return False, "Already has sma_200 in function"

                # Insert sma_200 after sma_50
                indent = '    '  # Same indent as other df assignments
                new_line = f"{indent}df['sma_200'] = df['close'].rolling(200).mean()  # 添加200日均线\n"
                lines.insert(i + 1, new_line)
                modified = True
                break

        # Only write if content changed
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return True, "Added sma_200 to function"
        else:
            return False, "Pattern not found or already exists"

    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Main function"""
    print("=" * 80)
    print("Adding sma_200 to add_technical_indicators function")
    print("=" * 80)

    # Find all get_trading*.py files
    current_dir = Path('.')
    files = list(current_dir.glob('get_trading_signal_*.py'))

    if not files:
        print("[ERROR] No get_trading_signal_*.py files found")
        return

    print(f"Found {len(files)} files to update\n")

    success_count = 0
    skip_count = 0
    error_count = 0

    for file_path in sorted(files):
        print(f"Updating {file_path.name}...", end=" ")
        success, message = add_sma_200_to_function(file_path)

        if success:
            print(f"[OK] {message}")
            success_count += 1
        else:
            if "Already has" in message or "already exists" in message:
                print(f"[SKIP] {message}")
                skip_count += 1
            else:
                print(f"[WARN] {message}")
                skip_count += 1

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total files:    {len(files)}")
    print(f"Updated:        {success_count}")
    print(f"Already OK:     {skip_count}")
    print(f"Errors:         {error_count}")
    print("=" * 80)

    if success_count > 0:
        print(f"\n[OK] Successfully updated {success_count} files!")
    elif skip_count == len(files):
        print("\n[OK] All files already have sma_200 in add_technical_indicators!")
    else:
        print("\n[INFO] Some files may need manual review")

if __name__ == "__main__":
    main()
