import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(r"C:\Users\SReeJAL\OneDrive\Desktop\Quant_Assignment\part 2\nifty_features_5min.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# ===============================
# SELECT FEATURES FOR HMM
# ===============================
features = [
    'Average_IV', 
    'IV_Spread', 
    'PCR_OI', 
    'call_delta', 
    'Gamma_Exposure', 
    'Vega',           # if missing, will fill with zeros
    'Futures_Basis', 
    'spot_return'
]

# Fill missing Vega if needed
for f in features:
    if f not in df.columns:
        df[f] = 0

X = df[features]

# ===============================
# CLEAN & PREPROCESS FEATURES
# ===============================

# 1️⃣ Fill NaNs
X = X.fillna(0)

# 2️⃣ Remove constant columns (std = 0)
X = X.loc[:, X.std() > 0]

# 3️⃣ Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# TRAIN/TEST SPLIT
# ===============================
train_len = int(len(X_scaled) * 0.7)
X_train = X_scaled[:train_len]
X_test  = X_scaled[train_len:]

# ===============================
# FIT HMM
# ===============================
model = GaussianHMM(n_components=3, covariance_type='full', n_iter=1000, random_state=42)
model.fit(X_train)

# Predict hidden states for all data
states = model.predict(X_scaled)
df['regime_state'] = states

# Map states to trend (+1 = up, 0 = sideways, -1 = down)
state_returns = df.groupby('regime_state')['spot_return'].mean().sort_values()
state_mapping = {
    state_returns.index[0]: -1,  # lowest mean return
    state_returns.index[1]: 0,   # middle
    state_returns.index[2]: 1    # highest mean return
}
df['regime'] = df['regime_state'].map(state_mapping)

# ===============================
# SAVE REGIME DATA
# ===============================
df.to_csv("nifty_hmm_regimes.csv")
print("HMM regime classification completed. File: nifty_hmm_regimes.csv")

# ===============================
# VISUALIZATION
# ===============================

# 1️⃣ Spot Price Chart with Regimes
plt.figure(figsize=(16,6))
colors = {1:'green', 0:'gray', -1:'red'}
plt.plot(df.index, df['spot_close'], color='black', label='Spot Price')
for regime in [-1,0,1]:
    mask = df['regime']==regime
    plt.scatter(df.index[mask], df['spot_close'][mask], color=colors[regime], s=10, label=f'Regime {regime}')
plt.title("Spot Price with HMM Regimes")
plt.legend()
plt.show()

# 2️⃣ Transition Matrix Heatmap
trans_mat = model.transmat_
plt.figure(figsize=(6,5))
sns.heatmap(trans_mat, annot=True, cmap='Blues', fmt=".2f")
plt.title("HMM Transition Matrix")
plt.xlabel("To State")
plt.ylabel("From State")
plt.show()

# 3️⃣ Regime Statistics
regime_stats = df.groupby('regime')[features + ['spot_return']].mean()
print("Regime statistics:\n", regime_stats)

# 4️⃣ Regime Duration Histogram
df['regime_change'] = df['regime'].ne(df['regime'].shift()).cumsum()
durations = df.groupby(['regime','regime_change']).size().reset_index(name='duration')
plt.figure(figsize=(10,5))
sns.histplot(data=durations, x='duration', hue='regime', multiple='stack', bins=50)
plt.title("Regime Duration Histogram")
plt.xlabel("Duration (number of 5-min intervals)")
plt.show()
