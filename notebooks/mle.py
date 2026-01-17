import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF info logs

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(r"C:\Users\SReeJAL\OneDrive\Desktop\Quant_Assignment\part 4\nifty_strategy_backtest.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# ===============================
# GENERATE TARGET (TRADE PROFITABILITY)
# ===============================
df['target'] = (df['strategy_return'].shift(-1) > 0).astype(int)
df = df.iloc[:-1]  # drop last row (no next return)

# ===============================
# FEATURES
# ===============================
feature_cols = [
    'EMA_5', 'EMA_15', 'position', 'regime',
    'Average_IV', 'IV_Spread', 'PCR_OI', 'Gamma_Exposure',
    'Vega', 'Futures_Basis', 'spot_return'
]
X = df[feature_cols].fillna(0)
y = df['target']

# ===============================
# SPLIT TRAIN / TEST
# ===============================
split_idx = int(len(df) * 0.7)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# ===============================
# MODEL A: XGBOOST
# ===============================
xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    objective='binary:logistic', random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# ===============================
# MODEL B: LSTM
# ===============================
seq_len = 10

# Create sequences for LSTM
X_seq, y_seq = [], []
for i in range(seq_len, len(X)):
    X_seq.append(X.iloc[i-seq_len:i].values)
    y_seq.append(y.iloc[i])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# Split sequences into train/test
split_seq = int(len(X_seq) * 0.7)
X_train_seq, X_test_seq = X_seq[:split_seq], X_seq[split_seq:]
y_train_seq, y_test_seq = y_seq[:split_seq], y_seq[split_seq:]

# Build LSTM model
lstm_model = Sequential([
    LSTM(64, input_shape=(seq_len, X_seq.shape[2])),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train LSTM
lstm_model.fit(
    X_train_seq, y_train_seq,
    epochs=50, batch_size=64,
    validation_split=0.2,
    callbacks=[es], verbose=1
)

# LSTM predictions
lstm_pred_proba = lstm_model.predict(X_test_seq).flatten()

# ===============================
# ALIGN ML PREDICTIONS WITH TEST DATA
# ===============================

# Calculate the actual start index for the common test set
# XGBoost's test set (X_test) starts at df.iloc[split_idx]
# The targets for LSTM's test set (X_test_seq) start at df.iloc[seq_len + split_seq]
# To compare both models on the same data, we take the later start point.
common_test_start_idx = max(split_idx, seq_len + split_seq)

# Create the common test DataFrame
df_test = df.iloc[common_test_start_idx:].copy()

# Calculate XGBoost predictions for this common test set
# xgb_pred_proba was generated for df.iloc[split_idx:].
# We need to slice it to match df_test's start index.
xgb_offset = common_test_start_idx - split_idx
aligned_xgb_pred_proba = xgb_pred_proba[xgb_offset:xgb_offset + len(df_test)]

# Assign predictions to the common test DataFrame
df_test['lstm_pred'] = lstm_pred_proba
df_test['xgb_pred'] = aligned_xgb_pred_proba

# ===============================
# ML-ENHANCED BACKTEST
# ===============================
df_test['position_xgb'] = df_test['position'] * (df_test['xgb_pred'] > 0.5).astype(int)
df_test['position_lstm'] = df_test['position'] * (df_test['lstm_pred'] > 0.5).astype(int)

# Strategy returns
df_test['return'] = df_test['spot_close'].pct_change()
df_test['strategy_return_xgb'] = df_test['position_xgb'].shift(1) * df_test['return']
df_test['strategy_return_lstm'] = df_test['position_lstm'].shift(1) * df_test['return']

# ===============================
# PERFORMANCE METRICS FUNCTION
# ===============================
def performance_metrics(returns):
    total_return = (1 + returns).prod() - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(252*78) if returns.std() != 0 else np.nan
    downside = returns[returns < 0]
    sortino = returns.mean() / downside.std() * np.sqrt(252*78) if len(downside) > 0 else np.nan
    cum_returns = (1 + returns).cumprod()
    max_drawdown = (cum_returns.cummax() - cum_returns).max()
    calmar = total_return / max_drawdown if max_drawdown != 0 else np.nan
    wins = returns[returns > 0].count()
    losses = returns[returns < 0].count()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else np.nan
    pf = returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if abs(returns[returns < 0].sum()) > 0 else np.nan
    return {
        'Total Return': total_return,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Calmar Ratio': calmar,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Profit Factor': pf,
        'Total Trades': wins + losses,
        'Average Trade Duration': len(returns) / (wins + losses) if (wins + losses) > 0 else np.nan
    }

# ===============================
# METRICS COMPARISON
# ===============================
baseline_metrics = performance_metrics(df_test['strategy_return'])
xgb_metrics      = performance_metrics(df_test['strategy_return_xgb'])
lstm_metrics     = performance_metrics(df_test['strategy_return_lstm'])

print("===== BASELINE EMA STRATEGY =====")
for k,v in baseline_metrics.items():
    print(f"{k}: {v:.4f}")

print("\n===== XGBOOST ML-ENHANCED STRATEGY =====")
for k,v in xgb_metrics.items():
    print(f"{k}: {v:.4f}")

print("\n===== LSTM ML-ENHANCED STRATEGY =====")
for k,v in lstm_metrics.items():
    print(f"{k}: {v:.4f}")

# ===============================
# SAVE RESULTS
# ===============================
df_test.to_csv("nifty_ml_enhanced_backtest.csv")
print("\nML-enhanced backtest completed. File saved: nifty_ml_enhanced_backtest.csv")
