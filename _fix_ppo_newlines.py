"""
Fix literal newlines in PPO-converted signal files.
re.sub interpreted \\n in replacement strings as real newlines → SyntaxError.
Also fixes double-indented '# PPO 環境與預測' comment line.
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

STOCKS = [
    '1301','1326','1560','1569','1595','2357','2404','2426','2485','2486',
    '3004','3135','3163','3363','3380','3455','3535','3563','3576','3583',
    '3615','3665','3680','4564','4577','4720','4768','4772','4951','4989',
    '4991','5351','6108','6138','6139','6176','6187','6213','6220','6230',
    '6234','6438','6526','6531','6605','6672','6788','6789','6830','6877',
    '6937','7703','7751','8027','8028','8064','8069','8071','8103','8147',
    '8438','8927',
]

# ── Broken → Correct replacement pairs ───────────────────────────────────────
# Note: Python string literals below contain REAL newlines where the file has them.
# The 'correct' strings use \\n which writes \n (the escape) to the file.

FIXES = [
    # 1. PPO AI predict print  (Type A + B)
    (
        'print("\n🧠 PPO AI 模型分析中...")',
        'print("\\n🧠 PPO AI 模型分析中...")'
    ),
    # 2. Technical indicators header  (Type B)
    (
        'print("\n" + "=" * 80 + "\n📊 技術指標\n" + "=" * 80)',
        'print("\\n" + "=" * 80 + "\\n📊 技術指標\\n" + "=" * 80)'
    ),
    # 3. Pattern score adjustment  (Type B)
    (
        'print("\n型態評分調整: "',
        'print("\\n型態評分調整: "'
    ),
    # 4. Trading signal header  (Type B)
    (
        'print("\n" + "=" * 80 + "\n🎯 交易信號\n" + "=" * 80)',
        'print("\\n" + "=" * 80 + "\\n🎯 交易信號\\n" + "=" * 80)'
    ),
    # 5. Model loading print  (Type A)
    (
        'print(f"\n📦 加载 PPO 模型: {model_path}")',
        'print(f"\\n📦 加载 PPO 模型: {model_path}")'
    ),
    # 6. Double-indented comment from X-match not capturing leading spaces
    (
        '        # PPO 環境與預測\n    env = ImprovedTradingEnv(df)',
        '    # PPO 環境與預測\n    env = ImprovedTradingEnv(df)'
    ),
]

fixed = 0
unchanged = 0

for code in STOCKS:
    fname = f"get_trading_signal_{code}.py"
    fpath = os.path.join(SCRIPT_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  SKIP (not found): {fname}")
        continue

    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content
    for broken, correct in FIXES:
        content = content.replace(broken, correct)

    if content != original:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ fixed: {fname}")
        fixed += 1
    else:
        print(f"  -- no change: {fname}")
        unchanged += 1

print(f"\nDone: {fixed} fixed, {unchanged} unchanged")
