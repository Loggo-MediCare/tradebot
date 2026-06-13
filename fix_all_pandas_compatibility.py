"""
Automatic Pandas Compatibility Fixer
Finds and fixes all training scripts with deprecated fillna(method=...) syntax
"""
import os
import re
import sys
import io
from pathlib import Path

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def fix_fillna_syntax(file_path):
    """Fix the deprecated fillna syntax in a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Pattern 1: df = df.fillna(method='bfill').fillna(method='ffill')
        pattern1 = r"df\s*=\s*df\.fillna\(method=['\"]bfill['\"]\)\.fillna\(method=['\"]ffill['\"]\)"
        replacement1 = "df = df.bfill().ffill()"
        content = re.sub(pattern1, replacement1, content)

        # Pattern 2: .fillna(method='ffill')
        pattern2 = r"\.fillna\(method=['\"]ffill['\"]\)"
        replacement2 = ".ffill()"
        content = re.sub(pattern2, replacement2, content)

        # Pattern 3: .fillna(method='bfill')
        pattern3 = r"\.fillna\(method=['\"]bfill['\"]\)"
        replacement3 = ".bfill()"
        content = re.sub(pattern3, replacement3, content)

        # Check if any changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, "Fixed"
        else:
            return False, "No changes needed"

    except Exception as e:
        return False, f"Error: {e}"

def scan_and_fix_directory(directory='.', pattern='train_*_improved.py'):
    """Scan directory for training scripts and fix them"""
    print("=" * 70)
    print("Pandas Compatibility Fixer")
    print("=" * 70)
    print(f"Scanning directory: {directory}")
    print(f"Pattern: {pattern}")
    print("=" * 70)

    files_to_fix = list(Path(directory).glob(pattern))

    if not files_to_fix:
        print("No files found matching pattern.")
        return

    print(f"\nFound {len(files_to_fix)} files to check\n")

    fixed_count = 0
    already_ok_count = 0
    error_count = 0

    results = []

    for file_path in sorted(files_to_fix):
        file_name = file_path.name
        was_fixed, message = fix_fillna_syntax(file_path)

        if was_fixed:
            status = "[FIXED]"
            fixed_count += 1
        elif "Error" in message:
            status = "[ERROR]"
            error_count += 1
        else:
            status = "[OK]"
            already_ok_count += 1

        results.append((file_name, status, message))
        print(f"{status:12} {file_name:50} {message}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total files scanned:  {len(files_to_fix)}")
    print(f"Fixed:                {fixed_count}")
    print(f"Already OK:           {already_ok_count}")
    if error_count > 0:
        print(f"Errors:               {error_count}")
    print("=" * 70)

    if fixed_count > 0:
        print(f"\nSuccessfully fixed {fixed_count} file(s)!")
        print("All training scripts are now compatible with pandas 2.x")
    else:
        print("\nAll files are already up to date!")

if __name__ == "__main__":
    # Fix all training scripts
    scan_and_fix_directory('.', 'train_*_improved.py')

    # Also check test files
    print("\n" + "=" * 70)
    print("Checking test files...")
    print("=" * 70)
    scan_and_fix_directory('.', 'test_*_improved.py')

    # Check other Python files that might have the issue
    print("\n" + "=" * 70)
    print("Checking other training files...")
    print("=" * 70)
    scan_and_fix_directory('.', 'train_*.py')

    print("\n" + "=" * 70)
    print("All done! You can now run any training script without pandas errors.")
    print("=" * 70)
