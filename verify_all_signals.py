"""
验证所有信号文件的语法正确性
"""
import glob
import py_compile
import sys

def verify_file(file_path):
    """验证单个文件的语法"""
    try:
        py_compile.compile(file_path, doraise=True)
        return True, None
    except SyntaxError as e:
        return False, str(e)

def main():
    signal_files = glob.glob('get_trading_signal_*.py')

    print(f"Verifying {len(signal_files)} signal files...")
    print("=" * 80)

    errors = []
    success_count = 0

    for file_path in signal_files:
        is_valid, error = verify_file(file_path)
        if is_valid:
            success_count += 1
            print(f"OK: {file_path}")
        else:
            errors.append((file_path, error))
            print(f"ERROR: {file_path}")
            print(f"  {error}")

    print("=" * 80)
    print(f"\nVerification complete:")
    print(f"  Success: {success_count}/{len(signal_files)}")
    print(f"  Errors: {len(errors)}/{len(signal_files)}")

    if errors:
        print("\nFiles with errors:")
        for file_path, error in errors:
            print(f"  - {file_path}")
        sys.exit(1)
    else:
        print("\nAll files passed syntax check!")
        sys.exit(0)

if __name__ == '__main__':
    main()
