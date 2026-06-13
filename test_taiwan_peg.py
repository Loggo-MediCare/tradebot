"""
测试台股 PEG 比率获取
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import yfinance as yf
import json

# 测试 2330.TW (台積電)
print("=" * 80)
print("测试 2330.TW 台積電")
print("=" * 80)

ticker = yf.Ticker("2330.TW")

# 方法1: 检查所有可用的 info 数据
print("\n所有可用的键值:")
print("-" * 80)
info = ticker.info

# 寻找所有包含 'peg' 或 'PEG' 的键
peg_keys = [key for key in info.keys() if 'peg' in key.lower()]
print(f"\n包含 'peg' 的键: {peg_keys}")

# 打印所有可能相关的键值
potential_keys = ['pegRatio', 'trailingPegRatio', 'forwardPegRatio', 'fiveYearAvgPEG']
print("\n尝试常见的 PEG 键:")
for key in potential_keys:
    value = info.get(key, 'N/A')
    print(f"  {key}: {value}")

# 方法2: 检查 statistics
print("\n\n检查其他可能包含 PEG 的数据:")
print("-" * 80)

# 保存完整的 info 到文件以便查看
with open('2330_tw_info.json', 'w', encoding='utf-8') as f:
    # 过滤掉无法序列化的值
    serializable_info = {}
    for k, v in info.items():
        try:
            json.dumps(v)
            serializable_info[k] = v
        except:
            serializable_info[k] = str(v)
    json.dump(serializable_info, f, ensure_ascii=False, indent=2)

print("完整 info 数据已保存到 2330_tw_info.json")

# 方法3: 尝试直接访问网页数据
print("\n\n尝试从网页获取数据:")
print("-" * 80)
try:
    # yfinance 有时会从不同的端点获取数据
    import requests
    from bs4 import BeautifulSoup

    url = "https://finance.yahoo.com/quote/2330.TW/key-statistics"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        print(f"成功访问网页: {url}")
        # 简单检查是否包含 PEG
        if 'PEG' in response.text or 'peg' in response.text:
            print("✓ 网页包含 PEG 数据")
        else:
            print("✗ 网页未找到 PEG 数据")
    else:
        print(f"无法访问网页，状态码: {response.status_code}")
except Exception as e:
    print(f"访问网页失败: {e}")

print("\n" + "=" * 80)
