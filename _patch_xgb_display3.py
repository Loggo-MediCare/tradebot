"""
Batch-patch all get_trading_signal_*.py scripts:
- Only files where buy_prob comes from XGBoost predict_proba (proba[1] or prediction_proba[1])
- Skip PPO-derived files (buy_prob = action_val * 50 + 50)
- Update all buy_prob display lines to P(buy)/P(not buy) format
"""
import sys, glob, re
sys.stdout.reconfigure(encoding='utf-8')

# String replacements: (old, new)
# Note: files with sell_prob already defined use {sell_prob:.1f} instead of {100-buy_prob:.1f}
REPLACEMENTS_WITH_SELL = [
    (
        "print(f\"預測結果: {'買入機率' if prediction == 1 else '不買入'} (買入: {buy_prob:.1f}%, 不買入: {sell_prob:.1f}%)\")",
        "print(f\"預測結果: {'買入機率' if prediction == 1 else '不買入'} — 今日買入機率 P(buy): {buy_prob:.1f}%（P(not buy): {sell_prob:.1f}%）\")"
    ),
]

REPLACEMENTS_COMMON = [
    (
        "print(f\"\\n預測: {'買入機率' if pred == 1 else '不買入'} (買入: {buy_prob:.1f}%, 不買入: {proba[0]*100:.1f}%)\")",
        "print(f\"\\n預測: {'買入機率' if pred == 1 else '不買入'} — 今日買入機率 P(buy): {buy_prob:.1f}%（P(not buy): {proba[0]*100:.1f}%）\")"
    ),
    (
        "print(f\"買入概率: {buy_prob:.1f}%\")",
        "print(f\"今日買入機率 P(buy): {buy_prob:.1f}%（P(not buy): {100-buy_prob:.1f}%）\")"
    ),
    (
        "print(f\"   信心度: {buy_prob:.1f}%\")",
        "print(f\"   今日買入機率 P(buy): {buy_prob:.1f}%（P(not buy): {100-buy_prob:.1f}%）\")"
    ),
    (
        "print(f\"   信心度: {buy_prob:.1f}%，謹慎操作\")",
        "print(f\"   今日買入機率 P(buy): {buy_prob:.1f}%（P(not buy): {100-buy_prob:.1f}%）（謹慎操作）\")"
    ),
    (
        "print(f\"   買入信心度低: {buy_prob:.1f}%\")",
        "print(f\"   今日買入機率 P(buy): {buy_prob:.1f}%（P(not buy): {100-buy_prob:.1f}%）（信心偏低）\")"
    ),
    (
        "print(f\"   買入信心度不足: {buy_prob:.1f}%\")",
        "print(f\"   今日買入機率 P(buy): {buy_prob:.1f}%（P(not buy): {100-buy_prob:.1f}%）（信心不足）\")"
    ),
    (
        "print(f\"  XGBoost ({XGB_ACC}%): 買入機率={buy_prob:.1f}%  →  {xgb_signal}\")",
        "print(f\"  XGBoost ({XGB_ACC}%): 今日買入機率 P(buy)={buy_prob:.1f}%（P(not buy): {100-buy_prob:.1f}%）  →  {xgb_signal}\")"
    ),
]

# Multi-line pattern for the "買入信心度不足 + continuation" pattern
MULTI_OLD = '        print(f"   買入信心度不足: {buy_prob:.1f}%" +\n'
MULTI_NEW = '        print(f"   今日買入機率 P(buy): {buy_prob:.1f}%（P(not buy): {100-buy_prob:.1f}%）（信心不足）" +\n'

updated_files = 0
total_replacements = 0

for f in sorted(glob.glob('get_trading_signal_*.py')):
    txt = open(f, encoding='utf-8', errors='ignore').read()

    # Skip PPO-derived scripts
    if 'buy_prob = action_val' in txt:
        continue

    # Skip already updated
    if 'P(buy)' in txt:
        continue

    # Only process files with XGBoost buy_prob
    if 'buy_prob' not in txt:
        continue

    new_txt = txt
    file_changes = 0

    # Apply multi-line fix first
    if MULTI_OLD in new_txt:
        new_txt = new_txt.replace(MULTI_OLD, MULTI_NEW)
        file_changes += 1

    # Apply common replacements
    for old, new in REPLACEMENTS_COMMON:
        if old in new_txt:
            new_txt = new_txt.replace(old, new)
            file_changes += 1

    # Apply sell_prob-specific replacement
    for old, new in REPLACEMENTS_WITH_SELL:
        if old in new_txt:
            new_txt = new_txt.replace(old, new)
            file_changes += 1

    if file_changes > 0 and new_txt != txt:
        open(f, 'w', encoding='utf-8').write(new_txt)
        print(f'  ✅ {f} ({file_changes} replacements)')
        updated_files += 1
        total_replacements += file_changes

print(f'\nDone: {updated_files} files updated, {total_replacements} total replacements')
