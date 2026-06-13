"""
Fix matplotlib backend issue in all get_trading_signal_*.py files
Adds MPLBACKEND='Agg' environment variable before other imports
"""
import os
import glob

def fix_file(filepath):
    """Add MPLBACKEND setting after TF environment variables"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already fixed
    if "MPLBACKEND" in content:
        print(f"  [SKIP] {os.path.basename(filepath)} - already has MPLBACKEND")
        return False

    # Find the line with TF_ENABLE_ONEDNN_OPTS and add MPLBACKEND after it
    old_pattern = "os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'"
    new_pattern = """os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['MPLBACKEND'] = 'Agg'  # Fix Tcl/Tk error on Windows"""

    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  [FIXED] {os.path.basename(filepath)}")
        return True
    else:
        # Try alternative pattern
        old_pattern2 = 'os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"'
        new_pattern2 = '''os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["MPLBACKEND"] = "Agg"  # Fix Tcl/Tk error on Windows'''

        if old_pattern2 in content:
            content = content.replace(old_pattern2, new_pattern2)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  [FIXED] {os.path.basename(filepath)}")
            return True

        print(f"  [SKIP] {os.path.basename(filepath)} - pattern not found")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Fixing matplotlib backend in signal files...")
    print("=" * 60)

    # Find all signal files
    signal_files = glob.glob("get_trading_signal_*.py")

    fixed = 0
    skipped = 0

    for filepath in sorted(signal_files):
        if fix_file(filepath):
            fixed += 1
        else:
            skipped += 1

    print("=" * 60)
    print(f"Done! Fixed: {fixed}, Skipped: {skipped}")
    print("=" * 60)
