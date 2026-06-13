"""
Fix the 44 signal scripts that have 'from backtest_utils import' but are
missing the actual calculate_ppo_backtest_roi() call + print.

Groups handled:
  A  — av = float(action_raw[0]) ...   Q值/Q 值 print  (25+2 scripts)
  B  — av = float(ppo_act[0])          PPO动作值 print  (8 scripts)
  C  — action_val = float(action[0])   动作值 print     (7 scripts)
  D  — 2417 ensemble                   模型輸出動作值   (1 script)
  E  — DQN-only 3046/5285/6834         skip             (3 scripts)
"""
import re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

SCRIPTS_44 = [
    'get_trading_signal_1582.py','get_trading_signal_1595.py','get_trading_signal_1710.py',
    'get_trading_signal_1717.py','get_trading_signal_2317aiwan.py','get_trading_signal_2355.py',
    'get_trading_signal_2417.py','get_trading_signal_2442.py','get_trading_signal_2455.py',
    'get_trading_signal_2472.py','get_trading_signal_2481.py','get_trading_signal_2851.py',
    'get_trading_signal_301005.py','get_trading_signal_3023.py','get_trading_signal_3033.py',
    'get_trading_signal_3036.py','get_trading_signal_3046.py','get_trading_signal_3094.py',
    'get_trading_signal_3105.py','get_trading_signal_3152.py','get_trading_signal_3167.py',
    'get_trading_signal_3357.py','get_trading_signal_3491.py','get_trading_signal_3491o.py',
    'get_trading_signal_3504.py','get_trading_signal_3563.py','get_trading_signal_3694.py',
    'get_trading_signal_4933.py','get_trading_signal_5285.py','get_trading_signal_5328.py',
    'get_trading_signal_6451.py','get_trading_signal_6643.py','get_trading_signal_6706.py',
    'get_trading_signal_6742.py','get_trading_signal_6834.py','get_trading_signal_6980.py',
    'get_trading_signal_7610.py','get_trading_signal_7734.py','get_trading_signal_8044.py',
    'get_trading_signal_8131aiwan.py','get_trading_signal_8291.py','get_trading_signal_8473.py',
    'get_trading_signal_9888.py','get_trading_signal_rhm_de.py',
]

# DQN-only scripts — no PPO predict branch, skip
DQN_ONLY = {'get_trading_signal_3046.py', 'get_trading_signal_5285.py', 'get_trading_signal_6834.py'}

patched = skipped = errors = 0

for path in SCRIPTS_44:
    try:
        src = open(path, encoding='utf-8').read()
    except FileNotFoundError:
        print(f'  ❌ MISSING: {path}')
        errors += 1
        continue

    if '_ppo_roi, _bh_roi = calculate_ppo_backtest_roi' in src:
        print(f'  ✅ already done: {path}')
        skipped += 1
        continue

    if path in DQN_ONLY:
        print(f'  ⏭️  DQN-only, skip: {path}')
        skipped += 1
        continue

    original = src
    changed = False

    # ── GROUP B: ppo_act pattern ─────────────────────────────────────────────
    if 'av = float(ppo_act[0])' in src:
        # Insert calc after av = float(ppo_act[0])
        src = src.replace(
            '        av = float(ppo_act[0])\n',
            '        av = float(ppo_act[0])\n'
            '        _ppo_roi, _bh_roi = calculate_ppo_backtest_roi(model, df)\n',
            1
        )
        # Replace PPO动作值 print line
        src = re.sub(
            r'(\s+)print\(f"PPO动作值: \{confidence:.3f\}.*?\)\n',
            r'\1print_ppo_action_line(av, _ppo_roi, _bh_roi)\n',
            src, count=1
        )
        changed = True

    # ── GROUP C: action_val pattern ──────────────────────────────────────────
    elif re.search(r'action_val = float\(action', src):
        # Insert calc after action_val = float(...)
        src = re.sub(
            r'(    action_val = float\([^\n]+\n)',
            r'\1    _ppo_roi, _bh_roi = calculate_ppo_backtest_roi(model, df)\n',
            src, count=1
        )
        # Replace 动作值 print line
        src = re.sub(
            r'(\s+)print\(f"动作值: \{action_val:\+\.4f\}"\)\n',
            r'\1print_ppo_action_line(action_val, _ppo_roi, _bh_roi)\n',
            src, count=1
        )
        changed = True

    # ── GROUP D: 2417 ensemble ───────────────────────────────────────────────
    elif path == 'get_trading_signal_2417.py':
        # Insert calc after action_value = float(np.mean(action_values))
        src = re.sub(
            r'(    action_value = float\(np\.mean\(action_values\)\)\n)',
            r'\1    _ppo_roi, _bh_roi = calculate_ppo_backtest_roi(\n'
            r'        next((m for m in models if hasattr(m, "policy")), None), df)\n',
            src, count=1
        )
        # Replace 模型輸出動作值 print
        src = re.sub(
            r'(\s+)print\(f"   模型輸出動作值: \{action_value:\+\.4f\}"\)\n',
            r'\1print_ppo_action_line(action_value, _ppo_roi, _bh_roi)\n',
            src, count=1
        )
        changed = True

    # ── GROUP A: action_raw pattern ──────────────────────────────────────────
    elif re.search(r'av = float\(action_raw\[0\]\)', src):
        # 1. Init vars before the PPO branch
        ppo_if_re = re.compile(r'(    if _mtype == .PPO.:\n|    if model_type == .PPO.:\n)')
        m = ppo_if_re.search(src)
        if m:
            init_line = '    _ppo_roi, _bh_roi, _av = None, None, None\n'
            src = src[:m.start()] + init_line + src[m.start():]

        # 2. Add calc + _av capture after av = float(action_raw[0]) line
        src = re.sub(
            r'(        av = float\(action_raw\[0\]\) if hasattr\(action_raw, .__len__.\) else float\(action_raw\)\n)',
            r'\1        _ppo_roi, _bh_roi = calculate_ppo_backtest_roi(model, df)\n'
            r'        _av = av\n',
            src, count=1
        )

        # 3. After Q值 or Q 値 print line, insert conditional PPO action print
        q_re = re.compile(
            r'(    print\(f"Q.値: Hold=\{q_values\[0\].*?\n)',
            re.DOTALL
        )
        m2 = q_re.search(src)
        if m2:
            insert = (
                '    if _av is not None:\n'
                '        print_ppo_action_line(_av, _ppo_roi, _bh_roi)\n'
            )
            src = src[:m2.end()] + insert + src[m2.end():]
        changed = True

    if changed and src != original:
        open(path, 'w', encoding='utf-8').write(src)
        print(f'  ✅ patched [{path}]')
        patched += 1
    elif changed:
        print(f'  ⚠️  no change made: {path}')
        errors += 1
    else:
        print(f'  ⚠️  no pattern matched: {path}')
        errors += 1

print(f'\nDone — patched {patched}, skipped {skipped}, errors/warnings {errors}')
