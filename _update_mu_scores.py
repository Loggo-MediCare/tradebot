import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from roi_control import print_roi
from model_accuracy_tracker import ModelAccuracyTracker, get_model_accuracy_display, get_best_model_type

updates = {
    ('MU', 'XGBoost'): (202.97, 100.0),
    ('MU', 'Hybrid'):  (258.14, 100.0),
}
for (sym, mtype), (roi, wr) in updates.items():
    t = ModelAccuracyTracker(sym, mtype)
    score = ModelAccuracyTracker.roi_to_score(roi)
    t.update_training_stats(backtest_acc=score, win_rate=wr)
    print_roi(f'{mtype}: ROI {roi:+.2f}% -> AI score {score:.1f}/100')

print()
print(get_model_accuracy_display('MU'))
w, p, d, x, h = get_best_model_type('MU')
print(f'PPO:{p}  DQN:{d}  XGB:{x}  HYB:{h}')
