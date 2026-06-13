"""
批量运行所有训练文件，生成特征重要性 JSON 文件
=========================================================
这个脚本会依次运行所有 train_*_improved.py 文件
每个文件会生成对应的 {ticker}_feature_importance.json

注意：
- 完整训练需要很长时间 (每个股票可能需要数小时)
- 建议在后台运行或分批执行
- 可以随时中断，已完成的 JSON 文件会保存
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import subprocess
from datetime import datetime

# 所有训练文件
training_files = [
    # 台股
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
    # 美股
    'train_aapl_improved.py',
    'train_avgo_improved.py',
    'train_goog_improved.py',
    'train_mu_improved.py',
    'train_nvda_improved.py',
]

# 预期生成的 JSON 文件
expected_json_files = {
    'train_1519_taiwan_improved.py': '1519.TW_feature_importance.json',
    'train_2317_taiwan_improved.py': '2317.TW_feature_importance.json',
    'train_2330_taiwan_improved.py': '2330.TW_feature_importance.json',
    'train_2337_taiwan_improved.py': '2337.TW_feature_importance.json',
    'train_2344_taiwan_improved.py': '2344.TW_feature_importance.json',
    'train_2408_taiwan_improved.py': '2408.TW_feature_importance.json',
    'train_3711_taiwan_improved.py': '3711.TW_feature_importance.json',
    'train_3715_taiwan_improved.py': '3715.TW_feature_importance.json',
    'train_4991_taiwan_improved.py': '4991.TW_feature_importance.json',
    'train_6175_taiwan_improved.py': '6175.TW_feature_importance.json',
    'train_6269_taiwan_improved.py': '6269.TW_feature_importance.json',
    'train_6515_taiwan_improved.py': '6515.TW_feature_importance.json',
    'train_6770_taiwan_improved.py': '6770.TW_feature_importance.json',
    'train_8131_taiwan_improved.py': '8131.TW_feature_importance.json',
    'train_aapl_improved.py': 'AAPL_feature_importance.json',
    'train_avgo_improved.py': 'AVGO_feature_importance.json',
    'train_goog_improved.py': 'GOOG_feature_importance.json',
    'train_mu_improved.py': 'MU_feature_importance.json',
    'train_nvda_improved.py': 'NVDA_feature_importance.json',
}

def check_existing_json():
    """检查已存在的 JSON 文件"""
    project_dir = r'C:\Users\Silvi\Projects\trading-bot'
    existing = []
    missing = []

    for train_file, json_file in expected_json_files.items():
        json_path = os.path.join(project_dir, json_file)
        if os.path.exists(json_path):
            existing.append(json_file)
        else:
            missing.append((train_file, json_file))

    return existing, missing

def run_training_file(train_file):
    """运行单个训练文件"""
    project_dir = r'C:\Users\Silvi\Projects\trading-bot'
    filepath = os.path.join(project_dir, train_file)

    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {train_file}")
        return False

    print(f"\n{'='*70}")
    print(f"🚀 开始训练: {train_file}")
    print(f"{'='*70}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        # 运行训练文件
        import sys
        result = subprocess.run(
            [sys.executable, filepath],
            cwd=project_dir,
            capture_output=False,  # 实时显示输出
            text=True,
            timeout=7200  # 2小时超时
        )

        if result.returncode == 0:
            # 检查 JSON 文件是否生成
            json_file = expected_json_files.get(train_file)
            if json_file:
                json_path = os.path.join(project_dir, json_file)
                if os.path.exists(json_path):
                    print(f"\n✅ 训练完成并生成 JSON: {json_file}")
                    return True
                else:
                    print(f"\n⚠️  训练完成但未生成 JSON: {json_file}")
                    return False
            else:
                print(f"\n✅ 训练完成: {train_file}")
                return True
        else:
            print(f"\n❌ 训练失败 (返回码 {result.returncode}): {train_file}")
            return False

    except subprocess.TimeoutExpired:
        print(f"\n⏰ 训练超时 (2小时): {train_file}")
        return False
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    print("=" * 70)
    print("批量生成特征重要性 JSON 文件")
    print("=" * 70)
    print(f"总共 {len(training_files)} 个训练文件\n")

    # 检查现有 JSON
    existing, missing = check_existing_json()

    print(f"📊 当前状态:")
    print(f"  ✅ 已存在 JSON: {len(existing)} 个")
    if existing:
        for json_file in existing:
            print(f"     • {json_file}")

    print(f"\n  ⏳ 需要生成 JSON: {len(missing)} 个")
    if missing:
        for train_file, json_file in missing:
            print(f"     • {json_file} (需要运行 {train_file})")

    if not missing:
        print("\n🎉 所有 JSON 文件已存在，无需重新训练!")
        print("=" * 70)
        sys.exit(0)

    # 询问用户是否继续
    print("\n" + "=" * 70)
    print("⚠️  警告：完整训练需要很长时间!")
    print(f"   预计时间: {len(missing)} 个股票 × 30-120 分钟/股票")
    print("   建议: 可以随时按 Ctrl+C 中断，已完成的文件会保存")
    print("=" * 70)

    try:
        user_input = input("\n是否开始训练? (y/n): ").strip().lower()
        if user_input != 'y':
            print("\n❌ 用户取消")
            sys.exit(0)
    except:
        # 如果无法获取输入，自动继续
        print("\n▶️  自动继续...")

    # 开始批量训练
    print("\n" + "=" * 70)
    print("🏁 开始批量训练")
    print("=" * 70)

    success_count = 0
    failed_files = []

    for train_file, json_file in missing:
        try:
            if run_training_file(train_file):
                success_count += 1
            else:
                failed_files.append(train_file)
        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断训练")
            print(f"已完成: {success_count}/{len(missing)} 个文件")
            break
        except Exception as e:
            print(f"\n❌ 未预期的错误: {e}")
            failed_files.append(train_file)

    # 最终报告
    print("\n" + "=" * 70)
    print("📊 训练完成报告")
    print("=" * 70)
    print(f"成功: {success_count}/{len(missing)} 个文件")

    if failed_files:
        print(f"\n失败的文件:")
        for f in failed_files:
            print(f"  ❌ {f}")

    # 再次检查 JSON 状态
    existing_final, missing_final = check_existing_json()
    print(f"\n当前 JSON 文件状态:")
    print(f"  ✅ 已存在: {len(existing_final)} 个")
    print(f"  ⏳ 仍需生成: {len(missing_final)} 个")

    print("=" * 70)
