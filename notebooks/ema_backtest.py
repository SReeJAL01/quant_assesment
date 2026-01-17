import pandas as pd
import numpy as np

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(r"C:\Users\SReeJAL\OneDrive\Desktop\Quant_Assignment\part 3\nifty_hmm_regimes.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Ensure EMAs exist
if 'EMA_5' not in df.columns or 'EMA_15' not in df.columns:
    df['EMA_5'] = df['spot_close'].ewm(span=5, adjust=False).mean()
    df['EMA_15'] = df['spot_close'].ewm(span=15, adjust=False).mean()

# ===============================
# GENERATE SIGNALS
# ===============================
df['position'] = 0

# Signal logic
for i in range(1, len(df)):
    # LONG ENTRY
    if (df['EMA_5'].iloc[i-1] < df['EMA_15'].iloc[i-1]) and (df['EMA_5'].iloc[i] > df['EMA_15'].iloc[i]) and (df['regime'].iloc[i] == 1):
        df.loc[df.index[i], 'position'] = 1
    # LONG EXIT
    elif (df['EMA_5'].iloc[i-1] > df['EMA_15'].iloc[i-1]) and (df['EMA_5'].iloc[i] < df['EMA_15'].iloc[i]) and (df['position'].iloc[i-1] == 1):
        df.loc[df.index[i], 'position'] = 0
    # SHORT ENTRY
    elif (df['EMA_5'].iloc[i-1] > df['EMA_15'].iloc[i-1]) and (df['EMA_5'].iloc[i] < df['EMA_15'].iloc[i]) and (df['regime'].iloc[i] == -1):
        df.loc[df.index[i], 'position'] = -1
    # SHORT EXIT
    elif (df['EMA_5'].iloc[i-1] < df['EMA_15'].iloc[i-1]) and (df['EMA_5'].iloc[i] > df['EMA_15'].iloc[i]) and (df['position'].iloc[i-1] == -1):
        df.loc[df.index[i], 'position'] = 0
    else:
        df.loc[df.index[i], 'position'] = df['position'].iloc[i-1]


# ===============================
# CALCULATE RETURNS
# ===============================
df['return'] = df['spot_close'].pct_change()
df['strategy_return'] = df['position'].shift(1) * df['return']  # enter at next candle

# ===============================
# SPLIT TRAINING/TESTING
# ===============================
split_idx = int(len(df) * 0.7)
train = df.iloc[:split_idx]
test  = df.iloc[split_idx:]

# ===============================
# PERFORMANCE METRICS FUNCTION
# ===============================
def performance_metrics(returns):
    total_return = (1 + returns).prod() - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(252*78)  # 252 trading days, 78 5-min per day
    downside = returns[returns < 0]
    sortino = returns.mean() / downside.std() * np.sqrt(252*78) if len(downside) > 0 else np.nan
    cum_returns = (1 + returns).cumprod()
    max_drawdown = (cum_returns.cummax() - cum_returns).max()
    calmar = total_return / max_drawdown if max_drawdown != 0 else np.nan
    wins = returns[returns > 0].count()
    losses = returns[returns < 0].count()
    win_rate = wins / (wins + losses) if (wins+losses)>0 else np.nan
    pf = returns[returns>0].sum() / abs(returns[returns<0].sum()) if abs(returns[returns<0].sum())>0 else np.nan
    return {
        'Total Return': total_return,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Calmar Ratio': calmar,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Profit Factor': pf,
        'Total Trades': wins+losses,
        'Average Trade Duration': len(returns)/ (wins+losses) if (wins+losses)>0 else np.nan
    }

# ===============================
# CALCULATE METRICS
# ===============================
train_metrics = performance_metrics(train['strategy_return'])
test_metrics  = performance_metrics(test['strategy_return'])

print("===== TRAINING METRICS =====")
for k,v in train_metrics.items():
    print(f"{k}: {v:.4f}")
print("\n===== TESTING METRICS =====")
for k,v in test_metrics.items():
    print(f"{k}: {v:.4f}")

# ===============================
# SAVE STRATEGY RESULTS
# ===============================
df.to_csv("nifty_strategy_backtest.csv")
print("\nStrategy backtest completed. File saved: nifty_strategy_backtest.csv")
