"""
为所有训练文件添加 JSON 保存功能，用于动态调整交易信号权重
"""
import os
import re
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

train_files = [
    'train_1519_taiwan_improved.py',
    'train_2317_taiwan_improved.py',
    'train_2330_taiwan_improved.py',
    'train_2337_taiwan_improved.py',
    'train_2344_taiwan_improved.py',
    'train_2408_taiwan_improved.py',
    'train_3711_taiwan_improved.py',
    'train_3715_taiwan_improved.py',
    'train_4991_taiwan_improved.py',
    'train_6175_taiwan_improved.py',
    'train_6269_taiwan_improved.py',
    'train_6515_taiwan_improved.py',
    'train_6770_taiwan_improved.py',
    'train_8131_taiwan_improved.py',
    'train_aapl_improved.py',
    'train_avgo_improved.py',
    'train_goog_improved.py',
    'train_mu_improved.py',
    'train_nvda_improved.py',
]

# 需要添加的代码段（在 return 语句之前）
json_save_code = """
    # 保存为 JSON 文件供交易信号使用
    import json
    json_data = {
        'ticker': ticker,
        'analysis_date': datetime.now().strftime('%Y-%m-%d'),
        'model_accuracy': float(accuracy),
        'feature_importance': {
            row['Feature']: float(row['Importance'])
            for _, row in feature_importance_df.iterrows()
        }
    }
    json_filename = f'{ticker}_feature_importance.json'
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"✅ 特徵重要性數據已保存: {json_filename}")

"""

def update_train_file(filename):
    """为训练文件添加 JSON 保存功能"""
    filepath = os.path.join(r'C:\Users\Silvi\Projects\trading-bot', filename)

    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filename}")
        return False

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        modified = False

        # 检查是否已经有 JSON 保存代码
        if 'json_filename = ' in content and '_feature_importance.json' in content:
            print(f"⏭️  跳过 (已有JSON保存): {filename}")
            return True

        # 在 analyze_feature_importance 函数的 return 之前添加 JSON 保存
        # 查找函数中的 return feature_importance_df
        pattern = r'(    plt\.close\(\)\s*\n)(\s*return feature_importance_df)'

        if re.search(pattern, content):
            content = re.sub(
                pattern,
                r'\1' + json_save_code + r'\2',
                content
            )
            modified = True

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"✅ 更新成功: {filename}")
            return True
        else:
            print(f"⚠️  未找到插入点: {filename}")
            return False

    except Exception as e:
        print(f"❌ 更新失败 {filename}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("为训练文件添加 JSON 保存功能")
    print("=" * 70)
    print(f"\n将更新 {len(train_files)} 个文件...")
    print("\n新增功能:")
    print("  1. ✅ 保存特征重要性为 JSON 文件")
    print("  2. ✅ 包含股票代码、日期、准确率")
    print("  3. ✅ 供交易信号动态调整权重使用")
    print("\n" + "=" * 70)
    print("\n开始更新...\n")

    success_count = 0
    for filename in train_files:
        if update_train_file(filename):
            success_count += 1

    print("\n" + "=" * 70)
    print(f"完成! 成功更新 {success_count}/{len(train_files)} 个文件")
    print("=" * 70)
