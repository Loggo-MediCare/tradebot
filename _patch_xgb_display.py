import sys
sys.stdout.reconfigure(encoding='utf-8')

files = [
    'get_trading_signal_1326.py','get_trading_signal_2303.py','get_trading_signal_2308.py',
    'get_trading_signal_2337.py','get_trading_signal_2344.py','get_trading_signal_2368.py',
    'get_trading_signal_2449.py','get_trading_signal_2609.py','get_trading_signal_2890.py',
    'get_trading_signal_2892.py','get_trading_signal_3034.py','get_trading_signal_3036.py',
    'get_trading_signal_3189.py','get_trading_signal_3231.py','get_trading_signal_3443.py',
]

OLD = "print(f\"XGBoost買入概率:{xgb_prob*100:.1f}%  {'看多📈'if xgb_pred==1 else'看空📉'}\")"
NEW = "print(f\"今日買入機率 P(buy): {xgb_prob*100:.1f}%（P(not buy): {(1-xgb_prob)*100:.1f}%）  {'看多📈'if xgb_pred==1 else'看空📉'}\")"

updated = 0
for f in files:
    txt = open(f, encoding='utf-8').read()
    if OLD in txt:
        open(f, 'w', encoding='utf-8').write(txt.replace(OLD, NEW))
        print(f'  ✅ {f}')
        updated += 1
    else:
        print(f'  ❌ {f}: old string not found')

print(f'\nDone: {updated}/{len(files)} updated')
