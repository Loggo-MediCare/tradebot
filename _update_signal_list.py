"""
Updates run_all_local_tw_to_excel.py:
  1. Removes 25 duplicate entries from SIGNAL_SCRIPTS
  2. Appends 155 missing TW stock entries
"""
import re, sys
sys.stdout.reconfigure(encoding='utf-8')

NEW_ENTRIES = [
    {'file': 'get_trading_signal_01810.py', 'name': '01810 小米集團'},
    {'file': 'get_trading_signal_02202.py', 'name': '02202 萬科企業'},
    {'file': 'get_trading_signal_1425.py',  'name': '1425 宇隆'},
    {'file': 'get_trading_signal_1471.py',  'name': '1471 首利'},
    {'file': 'get_trading_signal_1504.py',  'name': '1504 東元'},
    {'file': 'get_trading_signal_1513.py',  'name': '1513 中興電'},
    {'file': 'get_trading_signal_1529.py',  'name': '1529 樂士'},
    {'file': 'get_trading_signal_1582.py',  'name': '1582 信錦'},
    {'file': 'get_trading_signal_1711.py',  'name': '1711 永光'},
    {'file': 'get_trading_signal_1802.py',  'name': '1802 台玻'},
    {'file': 'get_trading_signal_2301.py',  'name': '2301 光寶科'},
    {'file': 'get_trading_signal_2324.py',  'name': '2324 仁寶'},
    {'file': 'get_trading_signal_2331.py',  'name': '2331 精英'},
    {'file': 'get_trading_signal_2356.py',  'name': '2356 英業達'},
    {'file': 'get_trading_signal_2369.py',  'name': '2369 菱生'},
    {'file': 'get_trading_signal_2375.py',  'name': '2375 凱美'},
    {'file': 'get_trading_signal_2377.py',  'name': '2377 微星科技'},
    {'file': 'get_trading_signal_2380.py',  'name': '2380 虹光'},
    {'file': 'get_trading_signal_2399.py',  'name': '2399 映泰'},
    {'file': 'get_trading_signal_2417.py',  'name': '2417 圓剛'},
    {'file': 'get_trading_signal_2431.py',  'name': '2431 聯昌'},
    {'file': 'get_trading_signal_2442.py',  'name': '2442 新美齊'},
    {'file': 'get_trading_signal_2464.py',  'name': '2464 盟立'},
    {'file': 'get_trading_signal_2467.py',  'name': '2467 志聖'},
    {'file': 'get_trading_signal_2472.py',  'name': '2472 立隆電'},
    {'file': 'get_trading_signal_2478.py',  'name': '2478 大毅'},
    {'file': 'get_trading_signal_2484.py',  'name': '2484 希華晶體'},
    {'file': 'get_trading_signal_2489.py',  'name': '2489 瑞軒'},
    {'file': 'get_trading_signal_2492.py',  'name': '2492 華新科'},
    {'file': 'get_trading_signal_2505.py',  'name': '2505 國揚'},
    {'file': 'get_trading_signal_2609.py',  'name': '2609 陽明'},
    {'file': 'get_trading_signal_2610.py',  'name': '2610 中華航空'},
    {'file': 'get_trading_signal_2618.py',  'name': '2618 長榮航空'},
    {'file': 'get_trading_signal_2645.py',  'name': '2645 漢翔'},
    {'file': 'get_trading_signal_2810.py',  'name': '2810 大成鋼'},
    {'file': 'get_trading_signal_2880.py',  'name': '2880 華南金'},
    {'file': 'get_trading_signal_2881.py',  'name': '2881 富邦金'},
    {'file': 'get_trading_signal_2882.py',  'name': '2882 國泰金'},
    {'file': 'get_trading_signal_2886.py',  'name': '2886 兆豐金'},
    {'file': 'get_trading_signal_2892.py',  'name': '2892 第一金'},
    {'file': 'get_trading_signal_2912.py',  'name': '2912 統一超商'},
    {'file': 'get_trading_signal_3016.py',  'name': '3016 嘉晶'},
    {'file': 'get_trading_signal_3036.py',  'name': '3036 文曄'},
    {'file': 'get_trading_signal_3042.py',  'name': '3042 晶技'},
    {'file': 'get_trading_signal_3049.py',  'name': '3049 和鑫'},
    {'file': 'get_trading_signal_3051.py',  'name': '3051 力特'},
    {'file': 'get_trading_signal_3057.py',  'name': '3057 喬鼎'},
    {'file': 'get_trading_signal_3090.py',  'name': '3090 日電貿'},
    {'file': 'get_trading_signal_3092.py',  'name': '3092 鴻碩'},
    {'file': 'get_trading_signal_3149.py',  'name': '3149 正達'},
    {'file': 'get_trading_signal_3167.py',  'name': '3167 渼洋'},
    {'file': 'get_trading_signal_3209.py',  'name': '3209 全科'},
    {'file': 'get_trading_signal_3221.py',  'name': '3221 台嘉碩'},
    {'file': 'get_trading_signal_3234.py',  'name': '3234 光環'},
    {'file': 'get_trading_signal_3293.py',  'name': '3293 鈊象'},
    {'file': 'get_trading_signal_3308.py',  'name': '3308 聯德'},
    {'file': 'get_trading_signal_3338.py',  'name': '3338 泰碩'},
    {'file': 'get_trading_signal_3357.py',  'name': '3357 臺慶科'},
    {'file': 'get_trading_signal_3360.py',  'name': '3360 尚志'},
    {'file': 'get_trading_signal_3374.py',  'name': '3374 精材'},
    {'file': 'get_trading_signal_3402.py',  'name': '3402 聯德控股'},
    {'file': 'get_trading_signal_3432.py',  'name': '3432 台端'},
    {'file': 'get_trading_signal_3485.py',  'name': '3485 敘豐'},
    {'file': 'get_trading_signal_3498.py',  'name': '3498 陽程'},
    {'file': 'get_trading_signal_3532.py',  'name': '3532 台勝科'},
    {'file': 'get_trading_signal_3630.py',  'name': '3630 新巨'},
    {'file': 'get_trading_signal_3645.py',  'name': '3645 達邁'},
    {'file': 'get_trading_signal_3690.py',  'name': '3690 美律'},
    {'file': 'get_trading_signal_3706.py',  'name': '3706 神達'},
    {'file': 'get_trading_signal_3707.py',  'name': '3707 漢磊'},
    {'file': 'get_trading_signal_4142.py',  'name': '4142 國光生技'},
    {'file': 'get_trading_signal_4167.py',  'name': '4167 松瑞藥'},
    {'file': 'get_trading_signal_4533.py',  'name': '4533 協易機'},
    {'file': 'get_trading_signal_4541.py',  'name': '4541 全球傳動'},
    {'file': 'get_trading_signal_4542.py',  'name': '4542 達方'},
    {'file': 'get_trading_signal_4760.py',  'name': '4760 勤凱'},
    {'file': 'get_trading_signal_4900.py',  'name': '4900 富爾特'},
    {'file': 'get_trading_signal_4908.py',  'name': '4908 前鼎'},
    {'file': 'get_trading_signal_4916.py',  'name': '4916 事欣科'},
    {'file': 'get_trading_signal_4919.py',  'name': '4919 新唐科技'},
    {'file': 'get_trading_signal_4927.py',  'name': '4927 泰鼎'},
    {'file': 'get_trading_signal_4966.py',  'name': '4966 譜瑞-KY'},
    {'file': 'get_trading_signal_4973.py',  'name': '4973 廣宇'},
    {'file': 'get_trading_signal_4979.py',  'name': '4979 華星光通'},
    {'file': 'get_trading_signal_5011.py',  'name': '5011 久陽'},
    {'file': 'get_trading_signal_5269.py',  'name': '5269 祥碩'},
    {'file': 'get_trading_signal_5274.py',  'name': '5274 信驊'},
    {'file': 'get_trading_signal_5386.py',  'name': '5386 雷凌科技'},
    {'file': 'get_trading_signal_5475.py',  'name': '5475 德英電子'},
    {'file': 'get_trading_signal_6104.py',  'name': '6104 創源'},
    {'file': 'get_trading_signal_6127.py',  'name': '6127 九豪'},
    {'file': 'get_trading_signal_6133.py',  'name': '6133 金橋'},
    {'file': 'get_trading_signal_6134.py',  'name': '6134 萬旭'},
    {'file': 'get_trading_signal_6135.py',  'name': '6135 新麗'},
    {'file': 'get_trading_signal_6155.py',  'name': '6155 鈦昇'},
    {'file': 'get_trading_signal_6166.py',  'name': '6166 增你強'},
    {'file': 'get_trading_signal_6173.py',  'name': '6173 信昌電'},
    {'file': 'get_trading_signal_6175.py',  'name': '6175 立敦'},
    {'file': 'get_trading_signal_6205.py',  'name': '6205 詮欣'},
    {'file': 'get_trading_signal_6217.py',  'name': '6217 中探針'},
    {'file': 'get_trading_signal_6224.py',  'name': '6224 聚鼎'},
    {'file': 'get_trading_signal_6263.py',  'name': '6263 普萊德'},
    {'file': 'get_trading_signal_6265.py',  'name': '6265 方土昇'},
    {'file': 'get_trading_signal_6282.py',  'name': '6282 康舒'},
    {'file': 'get_trading_signal_6284.py',  'name': '6284 佳邦'},
    {'file': 'get_trading_signal_6344.py',  'name': '6344 萬年清'},
    {'file': 'get_trading_signal_6415.py',  'name': '6415 矽力-KY'},
    {'file': 'get_trading_signal_6423.py',  'name': '6423 精測'},
    {'file': 'get_trading_signal_6426.py',  'name': '6426 統新'},
    {'file': 'get_trading_signal_6456.py',  'name': '6456 GreenPower'},
    {'file': 'get_trading_signal_6457.py',  'name': '6457 醣聯'},
    {'file': 'get_trading_signal_6485.py',  'name': '6485 點序'},
    {'file': 'get_trading_signal_6530.py',  'name': '6530 創威'},
    {'file': 'get_trading_signal_6570.py',  'name': '6570 維田'},
    {'file': 'get_trading_signal_6574.py',  'name': '6574 霖揚'},
    {'file': 'get_trading_signal_6584.py',  'name': '6584 申豐'},
    {'file': 'get_trading_signal_6592.py',  'name': '6592 和潤企業'},
    {'file': 'get_trading_signal_6603.py',  'name': '6603 富奇想'},
    {'file': 'get_trading_signal_6658.py',  'name': '6658 聯策'},
    {'file': 'get_trading_signal_6691.py',  'name': '6691 長天科技'},
    {'file': 'get_trading_signal_6727.py',  'name': '6727 亞泰金屬'},
    {'file': 'get_trading_signal_6829.py',  'name': '6829 勝麗'},
    {'file': 'get_trading_signal_6831.py',  'name': '6831 騰雲'},
    {'file': 'get_trading_signal_6834.py',  'name': '6834 新纖維'},
    {'file': 'get_trading_signal_6835.py',  'name': '6835 複合互連'},
    {'file': 'get_trading_signal_6862.py',  'name': '6862 勝麗-KY'},
    {'file': 'get_trading_signal_6903.py',  'name': '6903 信強'},
    {'file': 'get_trading_signal_6944.py',  'name': '6944 譜力'},
    {'file': 'get_trading_signal_6949.py',  'name': '6949 連展投控'},
    {'file': 'get_trading_signal_6958.py',  'name': '6958 無限金融'},
    {'file': 'get_trading_signal_6988.py',  'name': '6988 正基'},
    {'file': 'get_trading_signal_6994.py',  'name': '6994 環科'},
    {'file': 'get_trading_signal_7728.py',  'name': '7728 中磊'},
    {'file': 'get_trading_signal_7744.py',  'name': '7744 致伸'},
    {'file': 'get_trading_signal_7795.py',  'name': '7795 億泰'},
    {'file': 'get_trading_signal_7805.py',  'name': '7805 台灣外包'},
    {'file': 'get_trading_signal_8033.py',  'name': '8033 雷虎'},
    {'file': 'get_trading_signal_8038.py',  'name': '8038 長園科'},
    {'file': 'get_trading_signal_8039.py',  'name': '8039 台虹'},
    {'file': 'get_trading_signal_8042.py',  'name': '8042 金山電'},
    {'file': 'get_trading_signal_8043.py',  'name': '8043 蜜望寶'},
    {'file': 'get_trading_signal_8074.py',  'name': '8074 特技'},
    {'file': 'get_trading_signal_8086.py',  'name': '8086 宏捷科'},
    {'file': 'get_trading_signal_8271.py',  'name': '8271 宇瞻'},
    {'file': 'get_trading_signal_8292.py',  'name': '8292 邦特'},
    {'file': 'get_trading_signal_8341.py',  'name': '8341 日友'},
    {'file': 'get_trading_signal_8358.py',  'name': '8358 金居'},
    {'file': 'get_trading_signal_8377.py',  'name': '8377 建漢'},
    {'file': 'get_trading_signal_8431.py',  'name': '8431 匯僑'},
    {'file': 'get_trading_signal_8450.py',  'name': '8450 霸機科'},
    {'file': 'get_trading_signal_8462.py',  'name': '8462 柯訊'},
    {'file': 'get_trading_signal_9136.py',  'name': '9136 巨騰'},
    {'file': 'get_trading_signal_9888.py',  'name': '9888 碧桂園服務'},
    {'file': 'get_trading_signal_9933.py',  'name': '9933 中鼎'},
    {'file': 'get_trading_signal_9984.py',  'name': '9984 軟銀'},
]

# ── Read current file ────────────────────────────────────────────────────────
with open('run_all_local_tw_to_excel.py', encoding='utf-8') as f:
    txt = f.read()

# ── Step 1: De-duplicate SIGNAL_SCRIPTS ─────────────────────────────────────
# Find every {'file': '...', 'name': '...'} entry and keep first occurrence
entry_pattern = re.compile(
    r"(\s*\{'file':\s*'[^']+',\s*'name':\s*'[^']+'\},?\n?)",
    re.MULTILINE
)
seen = set()
def dedup(m):
    entry = m.group(0)
    fm = re.search(r"'file':\s*'([^']+)'", entry)
    if not fm:
        return entry
    key = fm.group(1)
    if key in seen:
        return ''   # remove duplicate
    seen.add(key)
    return entry

new_txt = entry_pattern.sub(dedup, txt)

# ── Step 2: Build new batch block ────────────────────────────────────────────
lines = []
lines.append("    # ── Batch 9 — 2026-06-09 (previously missing scripts) ──────────────────────")
for e in NEW_ENTRIES:
    lines.append(f"    {{'file': '{e['file']}', 'name': '{e['name']}'}},")
batch_block = '\n'.join(lines) + '\n'

# Insert just before the closing ] of SIGNAL_SCRIPTS
# Find the last entry + closing bracket
insert_marker = re.search(r"(\s*\{'file':[^}]+\},?\s*\]\s*\ndef get_ticker)", new_txt, re.DOTALL)
if not insert_marker:
    # fallback: find the ] that closes SIGNAL_SCRIPTS
    idx = new_txt.rfind("\n]")
    new_txt = new_txt[:idx] + '\n' + batch_block + new_txt[idx:]
else:
    pos = insert_marker.start(0)
    # find the \n] before "def get_ticker"
    bracket_pos = new_txt.rfind('\n]', 0, insert_marker.end())
    new_txt = new_txt[:bracket_pos] + '\n' + batch_block + new_txt[bracket_pos:]

with open('run_all_local_tw_to_excel.py', 'w', encoding='utf-8') as f:
    f.write(new_txt)

# ── Verify ───────────────────────────────────────────────────────────────────
from collections import Counter
import re as _re
all_listed = _re.findall(r"'file':\s*'(get_trading_signal_[^']+\.py)'", new_txt)
dupes = [(f, c) for f, c in Counter(all_listed).items() if c > 1]
print(f"Total entries after update: {len(all_listed)}")
print(f"Unique entries: {len(set(all_listed))}")
print(f"Remaining duplicates: {len(dupes)}")
if dupes:
    for d, c in dupes:
        print(f"  STILL DUP: {d} ({c}x)")
else:
    print("No duplicates remaining.")
print(f"New batch added: {len(NEW_ENTRIES)} stocks")
