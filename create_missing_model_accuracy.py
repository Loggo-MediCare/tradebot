"""
Create missing model_accuracy JSON files from feature_importance data
"""
import json
import os
from pathlib import Path
from datetime import datetime

def create_model_accuracy_from_feature_importance():
    """Create model_accuracy files for stocks that have feature_importance but no model_accuracy"""

    print("=" * 80)
    print("Creating Missing Model Accuracy Files")
    print("=" * 80)

    # Find all feature_importance files
    feature_files = list(Path('.').glob('*_feature_importance.json'))

    created_count = 0
    skipped_count = 0

    for feat_file in feature_files:
        # Extract ticker from filename
        ticker = feat_file.stem.replace('_feature_importance', '')

        # Check if model_accuracy file already exists
        model_acc_file = Path(f'model_accuracy_{ticker.replace(".", "_")}.json')

        if model_acc_file.exists():
            print(f"[SKIP] {ticker} - model_accuracy file already exists")
            skipped_count += 1
            continue

        # Read feature_importance file
        try:
            with open(feat_file, 'r', encoding='utf-8') as f:
                feat_data = json.load(f)

            # Get model accuracy from feature importance
            model_acc = feat_data.get('model_accuracy')

            if model_acc is None:
                print(f"[SKIP] {ticker} - no model_accuracy in feature_importance")
                skipped_count += 1
                continue

            # Convert to percentage (0-100)
            if model_acc < 1:
                model_acc = model_acc * 100

            # Create model_accuracy data
            model_acc_data = {
                'symbol': ticker,
                'model_type': 'PPO',
                'training_accuracy': None,
                'validation_accuracy': None,
                'backtest_accuracy': round(model_acc, 1),
                'win_rate': None,
                'sharpe_ratio': None,
                'total_signals': 0,
                'correct_signals': 0,
                'live_accuracy': None,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'history': [],
                'notes': f'Generated from feature_importance data on {datetime.now().strftime("%Y-%m-%d")}'
            }

            # Save model_accuracy file
            with open(model_acc_file, 'w', encoding='utf-8') as f:
                json.dump(model_acc_data, f, ensure_ascii=False, indent=2)

            print(f"[OK] Created {model_acc_file.name} - Accuracy: {model_acc:.1f}%")
            created_count += 1

        except Exception as e:
            print(f"[ERROR] Failed to process {ticker}: {e}")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Created: {created_count} files")
    print(f"Skipped: {skipped_count} files")
    print(f"Total feature_importance files: {len(feature_files)}")
    print("=" * 80)

if __name__ == "__main__":
    create_model_accuracy_from_feature_importance()
