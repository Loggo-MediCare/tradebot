"""
移除不相關股票的1303關聯性檢查
只保留半導體/記憶體相關股票的1303檢查
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import re

# 應該保留1303關聯的股票（記憶體/半導體相關）
KEEP_1303_CORRELATION = {
    'mu',      # Micron - DRAM/NAND製造商
    'smci',    # SuperMicro - 伺服器（使用大量記憶體）
    'nvda',    # NVIDIA - GPU（使用HBM記憶體）
    'tsm',     # 台積電
    '2330',    # 台積電（台股代號）
    'axon',    # 也許保留？（原本就有的）
}

# 應該移除1303關聯的股票
REMOVE_FILES = [
    'get_trading_signal_amd.py',      # CPU/GPU，不是DRAM
    'get_trading_signal_amzn.py',     # 電商
    'get_trading_signal_arm.py',      # ARM架構授權
    'get_trading_signal_gild.py',     # 生技製藥
    'get_trading_signal_googl.py',    # 軟體/搜尋引擎
    'get_trading_signal_hsai.py',     # AI
    'get_trading_signal_intc.py',     # CPU，雖有記憶體但不是主業
    'get_trading_signal_invz.py',     # 激光雷達
    'get_trading_signal_jazz.py',     # 生技製藥
    'get_trading_signal_jazz_xgb_backup.py',
    'get_trading_signal_meta.py',     # 社交媒體
    'get_trading_signal_mpwr.py',     # 電源管理IC
    'get_trading_signal_orcl.py',     # 資料庫軟體
    'get_trading_signal_rdw.py',      # 零售
    'get_trading_signal_txn.py',      # 類比/嵌入式IC
    # 台股也移除（與1303同為台股，關聯性不明確）
    'get_trading_signal_1301.py',
    'get_trading_signal_2357.py',
    'get_trading_signal_3135.py',
    'get_trading_signal_3135_xgb.py',
    'get_trading_signal_3363.py',
    'get_trading_signal_8069.py',
]

def remove_1303_correlation(file_path):
    """移除1303關聯性檢查"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 找到並移除1303相關區塊
        # Pattern: 從 "# 1. 检查与1303的相关性" 到下一個 "# 2."
        pattern = r'        # 1\. 检查与1303的相关性.*?(?=        # 2\.)'

        new_content = re.sub(pattern, '        # 1303相关性检查已移除 - 与台湾DRAM厂商无直接关联\n\n', content, flags=re.DOTALL)

        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        return False

    except Exception as e:
        print(f"錯誤 {file_path}: {e}")
        return False

# 處理所有需要移除的檔案
total_modified = 0

print("開始移除不相關的1303關聯性檢查...")
print("=" * 80)

for file_path in REMOVE_FILES:
    if remove_1303_correlation(file_path):
        print(f"✅ {file_path}")
        total_modified += 1
    else:
        print(f"⚪ {file_path}: 無需修改或失敗")

print("=" * 80)
print(f"完成! 共修改 {total_modified} 個檔案")
print("\n保留1303關聯的股票: MU, SMCI, NVDA, TSM, 2330, AXON")
