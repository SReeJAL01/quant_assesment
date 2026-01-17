import pandas as pd
import numpy as np
import mibian
import math

# ===============================
# LOAD MERGED DATA
# ===============================
df = pd.read_csv(r"C:\Users\SReeJAL\OneDrive\Desktop\Quant_Assignment\part 1\deliverables\nifty_merged_5min.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# ===============================
# 2.1 EMA INDICATORS
# ===============================
df['EMA_5']  = df['spot_close'].ewm(span=5, adjust=False).mean()
df['EMA_15'] = df['spot_close'].ewm(span=15, adjust=False).mean()

# ===============================
# 2.3 Derived Features
# ===============================

# Identify ATM strike
# ATM strike = closest strike to spot price at that timestamp
def get_atm_strike(sub):
    return sub.iloc[(sub['opt_strike'] - sub['spot_close']).abs().argsort()[:1]]['opt_strike'].values[0]

atm_call_iv = []
atm_put_iv  = []
atm_call_oi = []
atm_put_oi  = []
atm_call_delta = []
atm_put_delta = []

r = 6.5  # Risk-free rate in %

# Loop through timestamps
for ts in df.index:
    try:
        row = df.loc[ts]
        spot_price = row['spot_close']
        
        # Filter ATM call and put options
        opt_ts = df.loc[[ts], ['opt_strike','opt_type','opt_ltp','opt_iv','opt_oi']]
        atm_strike = opt_ts.iloc[(opt_ts['opt_strike'] - spot_price).abs().argsort()[:1]]['opt_strike'].values[0]
        
        call = opt_ts[(opt_ts['opt_strike']==atm_strike) & (opt_ts['opt_type']=='CE')].iloc[0]
        put  = opt_ts[(opt_ts['opt_strike']==atm_strike) & (opt_ts['opt_type']=='PE')].iloc[0]
        
        # Store IV and OI
        atm_call_iv.append(call['opt_iv'])
        atm_put_iv.append(put['opt_iv'])
        atm_call_oi.append(call['opt_oi'])
        atm_put_oi.append(put['opt_oi'])
        
        # ===============================
        # 2.2 Greeks using Mibian
        # ===============================
        # T in days = assume 1 day ~ 78 5-min intervals (adjust if you have exact expiry)
        T_days = 5  # placeholder for intraday expiry, adjust if needed
        c = mibian.BS([spot_price, call['opt_strike'], r, T_days], callPrice=call['opt_ltp'])
        p = mibian.BS([spot_price, put['opt_strike'], r, T_days], putPrice=put['opt_ltp'])
        
        atm_call_delta.append(c.callDelta)
        atm_put_delta.append(p.putDelta)
        
    except:
        atm_call_iv.append(np.nan)
        atm_put_iv.append(np.nan)
        atm_call_oi.append(np.nan)
        atm_put_oi.append(np.nan)
        atm_call_delta.append(np.nan)
        atm_put_delta.append(np.nan)

df['ATM_call_IV']  = atm_call_iv
df['ATM_put_IV']   = atm_put_iv
df['ATM_call_OI']  = atm_call_oi
df['ATM_put_OI']   = atm_put_oi
df['call_delta']   = atm_call_delta
df['put_delta']    = atm_put_delta

# Derived Features
df['Average_IV'] = (df['ATM_call_IV'] + df['ATM_put_IV']) / 2
df['IV_Spread']  = df['ATM_call_IV'] - df['ATM_put_IV']

df['PCR_OI'] = df['ATM_put_OI'] / df['ATM_call_OI']
df['PCR_Volume'] = df['opt_volume'] / df['opt_volume']  # simplified, replace with call/put volume if available

df['Futures_Basis'] = (df['fut_close'] - df['spot_close']) / df['spot_close']
df['spot_return']   = np.log(df['spot_close'] / df['spot_close'].shift(1))
df['fut_return']    = np.log(df['fut_close'] / df['fut_close'].shift(1))

df['Delta_Neutral_Ratio'] = np.abs(df['call_delta']) / np.abs(df['put_delta'])
df['Gamma_Exposure'] = df['spot_close'] * df['call_delta'] * df['ATM_call_OI']  # approximate

# ===============================
# SAVE FINAL FEATURE SET
# ===============================
df.to_csv("nifty_features_5min.csv")
print("Task 2 completed: nifty_features_5min.csv created")
