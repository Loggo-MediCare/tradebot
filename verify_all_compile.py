"""
Verify all signal files compile without syntax errors
"""
import py_compile
from pathlib import Path

def verify_compile(file_path):
    """Check if a Python file compiles"""
    try:
        py_compile.compile(str(file_path), doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        return False, str(e)

def main():
    """Verify all signal files"""
    print("=" * 80)
    print("Verifying All Signal Files Compile")
    print("=" * 80)

    signal_files = sorted(Path('.').glob('get_trading_signal_*.py'))

    success_count = 0
    failed_files = []

    for file_path in signal_files:
        success, error = verify_compile(file_path)

        if success:
            print(f"[OK] {file_path.name}")
            success_count += 1
        else:
            print(f"[ERROR] {file_path.name}")
            print(f"        {error}")
            failed_files.append(file_path.name)

    print("\n" + "=" * 80)
    print("Verification Results")
    print("=" * 80)
    print(f"Success: {success_count}/{len(signal_files)}")
    print(f"Failed:  {len(failed_files)}/{len(signal_files)}")

    if failed_files:
        print("\nFailed files:")
        for filename in failed_files:
            print(f"  - {filename}")
        print("=" * 80)
        return False
    else:
        print("\n>>> ALL FILES COMPILE SUCCESSFULLY!")
        print("=" * 80)
        return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
