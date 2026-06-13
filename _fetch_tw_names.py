import requests, re, json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

headers = {'User-Agent': 'Mozilla/5.0'}
r = requests.get('https://isin.twse.com.tw/isin/C_public.jsp?strMode=2', headers=headers, timeout=20)
text_twse = r.content.decode('big5', errors='replace')
r2 = requests.get('https://isin.twse.com.tw/isin/C_public.jsp?strMode=4', headers=headers, timeout=20)
text_tpex = r2.content.decode('big5', errors='replace')

results = {}
idsp = chr(12288)

for src in [text_twse, text_tpex]:
    for m in re.finditer(r'>(\d{4})' + idsp + r'([^<\n\r]{1,20})<', src):
        code = m.group(1)
        name = m.group(2).strip()
        # Filter out warrants
        if any(c in name for c in ['購', '售', '認購', '認售']):
            continue
        results[code] = name

targets = ['3046','2455','6706','3036','7610','6834','6742','5285','3105','3694','3563','3504','5328','3094','3357','3033','4933','2472','1710','8473','6451','3023','3167','2851','1582','6980','2481','2355','1595','6643','3152','8043','8044','8291','7734']
out = {}
for t in targets:
    out[t] = results.get(t, 'NOT FOUND')
    print(t + '|' + out[t])

with open('_tw_names_temp.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
