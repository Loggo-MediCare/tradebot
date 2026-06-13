"""
Verify syntax of all get_trading_signal_*.py files
"""
import py_compile
import sys
import io
from pathlib import Path

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def verify_syntax(file_path):
    """Verify Python syntax of a file"""
    try:
        py_compile.compile(str(file_path), doraise=True)
        return True, "OK"
    except py_compile.PyCompileError as e:
        return False, str(e)

def main():
    """Main function"""
    print("=" * 80)
    print("Verifying syntax of all get_trading_signal_*.py files")
    print("=" * 80)

    # Find all get_trading*.py files
    current_dir = Path('.')
    files = list(current_dir.glob('get_trading_signal_*.py'))

    if not files:
        print("[ERROR] No get_trading_signal_*.py files found")
        return

    print(f"Found {len(files)} files to verify\n")

    success_count = 0
    error_count = 0
    errors = []

    for file_path in sorted(files):
        print(f"Verifying {file_path.name}...", end=" ")
        success, message = verify_syntax(file_path)

        if success:
            print(f"[OK]")
            success_count += 1
        else:
            print(f"[ERROR]")
            error_count += 1
            errors.append((file_path.name, message))

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total files:    {len(files)}")
    print(f"Valid syntax:   {success_count}")
    print(f"Syntax errors:  {error_count}")
    print("=" * 80)

    if error_count > 0:
        print("\n" + "=" * 80)
        print("Files with errors:")
        print("=" * 80)
        for filename, error in errors:
            print(f"\n{filename}:")
            print(f"  {error}")
        print("\n[ERROR] Some files have syntax errors!")
        sys.exit(1)
    else:
        print("\n[OK] All files have valid Python syntax!")

if __name__ == "__main__":
    main()
