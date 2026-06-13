"""
批量訓練所有 XGBoost 模型
"""
import subprocess
import sys
import time

# All training scripts to run
training_scripts = [
    'train_1471_xgboost.py',  # Already trained
    'train_3030_tw_xgboost.py',
    'train_3092_tw_xgboost.py',
    'train_3221_two_xgboost.py',
    'train_3234_two_xgboost.py',
    'train_3535_tw_xgboost.py',
    'train_3563_tw_xgboost.py',
    'train_3576_tw_xgboost.py',
    'train_3615_two_xgboost.py',
    'train_3665_tw_xgboost.py',
    'train_4564_tw_xgboost.py',
    'train_4577_two_xgboost.py',
    'train_4768_two_xgboost.py',
    'train_4989_tw_xgboost.py',
    'train_4991_two_xgboost.py',
    'train_6220_two_xgboost.py',
    'train_6230_tw_xgboost.py',
    'train_6442_tw_xgboost.py',
    'train_6526_tw_xgboost.py',
    'train_6789_tw_xgboost.py',
    'train_6830_tw_xgboost.py',
    'train_6877_two_xgboost.py',
    'train_8438_tw_xgboost.py',
    'train_8927_two_xgboost.py',
]

print(f"Starting batch training for {len(training_scripts)} models...")
print("=" * 80)

results = []
start_time = time.time()

for i, script in enumerate(training_scripts, 1):
    print(f"\n[{i}/{len(training_scripts)}] Running: {script}")
    print("-" * 80)

    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout
            encoding='utf-8',
            errors='ignore'
        )

        if result.returncode == 0:
            # Extract accuracy from output
            output_lines = result.stdout.split('\n')
            test_acc = None
            for line in output_lines:
                if '測試準確度:' in line:
                    try:
                        test_acc = line.split(':')[1].strip().replace('%', '')
                    except:
                        pass

            results.append({
                'script': script,
                'status': 'SUCCESS',
                'accuracy': test_acc
            })
            print(f"SUCCESS - Accuracy: {test_acc}%")
        else:
            results.append({
                'script': script,
                'status': 'FAILED',
                'error': result.stderr[:200]
            })
            print(f"FAILED - Error: {result.stderr[:200]}")

    except subprocess.TimeoutExpired:
        results.append({
            'script': script,
            'status': 'TIMEOUT'
        })
        print("TIMEOUT - Exceeded 5 minutes")
    except Exception as e:
        results.append({
            'script': script,
            'status': 'ERROR',
            'error': str(e)
        })
        print(f"ERROR - {str(e)}")

elapsed_time = time.time() - start_time

# Final summary
print("\n" + "=" * 80)
print("BATCH TRAINING COMPLETE")
print("=" * 80)
print(f"Total scripts: {len(training_scripts)}")
print(f"Successful: {sum(1 for r in results if r['status'] == 'SUCCESS')}")
print(f"Failed: {sum(1 for r in results if r['status'] not in ['SUCCESS'])}")
print(f"Total time: {elapsed_time/60:.1f} minutes")
print("")

# Detailed results
print("DETAILED RESULTS:")
print("-" * 80)
for r in results:
    status_icon = "OK" if r['status'] == 'SUCCESS' else "!!"
    acc = r.get('accuracy', 'N/A')
    print(f"[{status_icon}] {r['script']:<35} {r['status']:<10} {acc}")

print("\n" + "=" * 80)
print("All models trained!")
print("=" * 80)
