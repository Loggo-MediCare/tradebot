import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ENTRIES = [
    ('6138', '茂達'), ('3264', '欣銓'), ('6147', '頎邦'), ('6257', '矽格'),
    ('8064', '東捷'), ('3455', '由田'), ('3450', '聯鈞'), ('4979', '華星光'),
    ('2426', '鼎元'), ('4966', '譜瑞-KY'), ('3581', '博磊'), ('3163', '波若威'),
    ('3265', '台星科'), ('3535', '晶彩科'), ('3714', '富采'), ('2340', '台亞'),
    ('3587', '閎康'), ('8028', '昇陽半導體'), ('3498', '陽程'), ('3680', '家登'),
    ('1773', '勝一'), ('2059', '川湖'), ('2417', '圓剛'), ('1717', '長興'),
    ('2442', '新美齊'),
]

ANCHOR = "{'file': 'get_trading_signal_8996.py', 'name': '8996 高力'},"

with open('run_all_local_tw.py', 'r', encoding='utf-8') as f:
    content = f.read()

added = 0
for code, name in ENTRIES:
    pattern = f"get_trading_signal_{code}.py"
    if pattern not in content:
        new_entry = f"    {{'file': 'get_trading_signal_{code}.py', 'name': '{code} {name}'}},"
        content = content.replace(ANCHOR, ANCHOR + '\n' + new_entry)
        added += 1
        print(f'Added {code} {name}')
    else:
        print(f'Skip {code} (exists)')

with open('run_all_local_tw.py', 'w', encoding='utf-8') as f:
    f.write(content)

print(f'\nDone. {added} entries added.')
