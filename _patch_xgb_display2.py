import sys
sys.stdout.reconfigure(encoding='utf-8')

patches = {
    'get_trading_signal_00981A.py': [
        (
            "print(f\"買入概率: {xgb_prob*100:.1f}%  ({'看多' if xgb_pred == 1 else '看空'})\")",
            "print(f\"今日買入機率 P(buy): {xgb_prob*100:.1f}%（P(not buy): {(1-xgb_prob)*100:.1f}%）  {'看多' if xgb_pred == 1 else '看空'}\")"
        ),
        (
            "print(f\"   XGBoost 買入概率: {xgb_prob*100:.1f}%\")",
            "print(f\"   今日買入機率 P(buy): {xgb_prob*100:.1f}%（P(not buy): {(1-xgb_prob)*100:.1f}%）\")"
        ),
    ],
    'get_trading_signal_2882.py': [
        (
            "print(f\"買入概率: {xgb_prob*100:.1f}%  [{bar}]  ({'看多 📈' if xgb_pred == 1 else '看空 📉'})\")",
            "print(f\"今日買入機率 P(buy): {xgb_prob*100:.1f}%（P(not buy): {(1-xgb_prob)*100:.1f}%）  [{bar}]  {'看多 📈' if xgb_pred == 1 else '看空 📉'}\")"
        ),
        (
            "print(f\"   XGBoost 買入概率: {xgb_prob*100:.1f}%\" if xgb_prob is not None else \"\")",
            "print(f\"   今日買入機率 P(buy): {xgb_prob*100:.1f}%（P(not buy): {(1-xgb_prob)*100:.1f}%）\" if xgb_prob is not None else \"\")"
        ),
    ],
}

for fname, replacements in patches.items():
    txt = open(fname, encoding='utf-8').read()
    changed = 0
    for old, new in replacements:
        if old in txt:
            txt = txt.replace(old, new)
            changed += 1
        else:
            print(f'  ⚠️  {fname}: pattern not found: {old[:60]}')
    if changed:
        open(fname, 'w', encoding='utf-8').write(txt)
        print(f'  ✅ {fname}: {changed} replacements')

print('Done')
