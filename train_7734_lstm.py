"""
7734.TWO (印能科技) — TensorFlow LSTM Model
Compares LSTM vs XGBoost (37.5%) vs PPO (23.17%)
Limited data: ~439 rows — uses heavy regularisation + early stopping
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys, io, json, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from datetime import datetime

TICKER = '7734.TWO'
LOOKBACK = 10      # sequence length (days fed into LSTM)
THRESHOLD = 0.02   # 5-day return > 2% = BUY label

print("=" * 60)
print(f"  {TICKER} (印能科技) — TensorFlow LSTM Training")
print("=" * 60)

# ── 1. Download & indicators ──────────────────────────────────
df = yf.download(TICKER, start='2015-01-01', end='2025-12-31', progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)
df = df.rename(columns={'Close':'close','Volume':'volume',
                         'Open':'open','High':'high','Low':'low'}).reset_index()
print(f"Raw data: {len(df)} days")

df['sma_10']  = df['close'].rolling(10).mean()
df['sma_30']  = df['close'].rolling(30).mean()
df['ema_12']  = df['close'].ewm(span=12).mean()
df['ema_26']  = df['close'].ewm(span=26).mean()
d = df['close'].diff()
g = d.where(d > 0, 0).rolling(14).mean()
l = (-d.where(d < 0, 0)).rolling(14).mean()
df['rsi']         = 100 - (100 / (1 + g / (l + 1e-10)))
df['macd']        = df['ema_12'] - df['ema_26']
df['macd_signal'] = df['macd'].ewm(span=9).mean()
df['bb_mid']  = df['close'].rolling(20).mean()
df['bb_std']  = df['close'].rolling(20).std()
df['bb_pos']  = ((df['close'] - (df['bb_mid'] - 2*df['bb_std'])) /
                  (4 * df['bb_std'] + 1e-10) * 100).fillna(50)
lo14 = df['low'].rolling(14).min(); hi14 = df['high'].rolling(14).max()
df['K']       = ((df['close'] - lo14) / (hi14 - lo14 + 1e-10) * 100).fillna(50)
df['vol_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)
df['ret_5d']  = df['close'].pct_change(5) * 100

df['future_ret'] = df['close'].shift(-5) / df['close'] - 1
df['target']     = (df['future_ret'] > THRESHOLD).astype(int)

FEATURES = ['close','rsi','macd','macd_signal','bb_pos','K',
            'sma_10','sma_30','vol_ratio','ret_5d']

df = df.dropna(subset=FEATURES + ['target'])
print(f"Clean rows: {len(df)}")

# ── 2. Scale ──────────────────────────────────────────────────
scaler = MinMaxScaler()
feature_data = scaler.fit_transform(df[FEATURES].values)
labels = df['target'].values

# ── 3. Build sequences ────────────────────────────────────────
X, y = [], []
for i in range(LOOKBACK, len(feature_data)):
    X.append(feature_data[i - LOOKBACK:i])
    y.append(labels[i])
X, y = np.array(X), np.array(y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"Train sequences: {len(X_train)} | Test sequences: {len(X_test)}")
print(f"Buy label ratio — train: {y_train.mean():.2f}  test: {y_test.mean():.2f}")

# ── 4. Build LSTM ─────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(42)
np.random.seed(42)

model = Sequential([
    LSTM(32, input_shape=(LOOKBACK, len(FEATURES)),
         return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    LSTM(16, dropout=0.2, recurrent_dropout=0.2),
    BatchNormalization(),
    Dense(8, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel architecture:")
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=20,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=10, min_lr=1e-5, verbose=0),
]

print("\nTraining LSTM...")
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=16,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=0,
    class_weight={0: 1.0, 1: 1.5}   # slight weight on BUY to handle imbalance
)

stopped_epoch = len(history.history['loss'])
print(f"Stopped at epoch {stopped_epoch}")

# ── 5. Evaluate ───────────────────────────────────────────────
y_prob  = model.predict(X_test, verbose=0).flatten()

# Find optimal threshold on train set to balance precision/recall
train_prob = model.predict(X_train, verbose=0).flatten()
best_thresh, best_f1 = 0.5, 0.0
from sklearn.metrics import f1_score
for t in np.arange(0.2, 0.8, 0.05):
    preds_t = (train_prob >= t).astype(int)
    if preds_t.sum() > 0:
        f1 = f1_score(y_train, preds_t, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

print(f"\nOptimal threshold (F1-based): {best_thresh:.2f}")

y_pred  = (y_prob >= best_thresh).astype(int)
lstm_acc = accuracy_score(y_test, y_pred)

train_pred = (train_prob >= best_thresh).astype(int)
train_acc  = accuracy_score(y_train, train_pred)

print(f"Train accuracy: {train_acc*100:.2f}%")
print(f"Test  accuracy: {lstm_acc*100:.2f}%")
print(f"Buy predictions: {y_pred.sum()}/{len(y_pred)}")
print(f"Note: {y_test.sum()}/{len(y_test)} actual BUYs in test set")

# ── 6. Save model & accuracy ──────────────────────────────────
model.save('lstm_7734_two_model.keras')
print("\nModel saved: lstm_7734_two_model.keras")

acc_data = {
    'symbol': TICKER,
    'model_type': 'LSTM',
    'training_accuracy': float(train_acc * 100),
    'validation_accuracy': float(lstm_acc * 100),
    'backtest_accuracy':   float(lstm_acc * 100),
    'epochs_trained': stopped_epoch,
    'lookback_days': LOOKBACK,
    'features': FEATURES,
    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}
with open('model_accuracy_7734_TWO_lstm.json', 'w', encoding='utf-8') as f:
    json.dump(acc_data, f, ensure_ascii=False, indent=2)

# ── 7. Comparison ─────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  7734.TWO (印能科技) MODEL COMPARISON")
print(f"{'='*60}")
results = [
    ('XGBoost', 37.50),
    ('PPO',     23.17),
    ('LSTM',    lstm_acc * 100),
]
results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
for rank, (name, acc) in enumerate(results_sorted, 1):
    marker = ' ← WINNER' if rank == 1 else ''
    print(f"  #{rank} {name:<10} {acc:>6.2f}%{marker}")
print(f"{'='*60}")

winner = results_sorted[0][0]
print(f"\nRecommendation: use {winner} model for 7734.TWO signal script")
if lstm_acc * 100 < 45:
    print("⚠️  All models below 45% — treat 印能科技 signals as low-confidence.")
    print("   Root cause: only ~439 rows of history. Best to wait for more data.")
