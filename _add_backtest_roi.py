"""
Batch-patch all get_trading_signal_*.py to add PPO backtest ROI display.

Changes per file:
  1. Add import at top (after existing imports)
  2. Insert backtest calc right after action_value is assigned
  3. Replace bare print("模型输出动作值...") line with print_ppo_action_line()
"""

import glob, re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

IMPORT_LINE = "from backtest_utils import calculate_ppo_backtest_roi, print_ppo_action_line\n"

# Two lines to insert right after  action_value = float(...)
BACKTEST_CALC = (
    "    # PPO backtest ROI\n"
    "    _ppo_roi, _bh_roi = calculate_ppo_backtest_roi(model, df)\n"
)

# Pattern that matches the old bare print line
OLD_PRINT_RE = re.compile(
    r'(\s*)print\(f"模型输出动作值: \{action_value:\+\.4f\}"\)\n'
)
NEW_PRINT = (
    r'\1print_ppo_action_line(action_value, _ppo_roi, _bh_roi)\n'
)

scripts = sorted(glob.glob("get_trading_signal_*.py"))
patched = 0
skipped = 0

for path in scripts:
    try:
        src = open(path, encoding='utf-8').read()
    except Exception as e:
        print(f"  ⚠️  read error {path}: {e}")
        continue

    # Skip if already patched
    if "from backtest_utils import" in src:
        skipped += 1
        continue

    # ── 1. Add import after the last consecutive import/from line ──────────
    lines = src.splitlines(keepends=True)
    last_import_idx = -1
    for i, ln in enumerate(lines):
        # Only top-level imports (no leading whitespace)
        if re.match(r'^(import |from \S+ import)', ln):
            last_import_idx = i
    if last_import_idx == -1:
        print(f"  ⚠️  no import block found: {path}")
        continue
    lines.insert(last_import_idx + 1, IMPORT_LINE)
    src = ''.join(lines)

    # ── 2. Insert backtest calc after action_value assignment ──────────────
    # Match the assignment line (handles both array and scalar)
    assign_re = re.compile(
        r'(    action_value = float\(action\[0\]\) if isinstance\(action, np\.ndarray\) else float\(action\)\n)'
    )
    if assign_re.search(src):
        src = assign_re.sub(r'\1' + BACKTEST_CALC, src, count=1)
    else:
        # fallback: simpler assignment pattern
        assign_re2 = re.compile(r'(    action_value = float\([^)]+\)\n)')
        if assign_re2.search(src):
            src = assign_re2.sub(r'\1' + BACKTEST_CALC, src, count=1)
        else:
            print(f"  ⚠️  action_value assignment not found: {path}")

    # ── 3. Replace old print line ──────────────────────────────────────────
    if OLD_PRINT_RE.search(src):
        src = OLD_PRINT_RE.sub(NEW_PRINT, src, count=1)
    else:
        # Some scripts use slightly different spacing/quotes — try looser match
        loose = re.compile(r'(\s*)print\(f.模型输出动作值.*action_value.*\)\n')
        if loose.search(src):
            src = loose.sub(r'\1print_ppo_action_line(action_value, _ppo_roi, _bh_roi)\n', src, count=1)
        else:
            print(f"  ℹ️  print line not found (may use fmt.print_metric): {path}")

    open(path, 'w', encoding='utf-8').write(src)
    patched += 1
    print(f"  ✅ {path}")

print(f"\nDone — patched {patched}, skipped (already done) {skipped}, total {len(scripts)}")
