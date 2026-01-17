import pandas as pd
import numpy as np

# ===============================
# LOAD DATA
# ===============================
spot = pd.read_csv("nifty_spot_5min.csv")
fut  = pd.read_csv("nifty_futures_5min.csv")
opt  = pd.read_csv("nifty_options_5min.csv")

# ===============================
# NORMALIZE TIMESTAMP
# ===============================
def fix_timestamp(df):
    df.rename(columns={'date': 'timestamp'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)
    return df

spot = fix_timestamp(spot)
fut  = fix_timestamp(fut)
opt  = fix_timestamp(opt)

# ===============================
# HANDLE MISSING VALUES
# ===============================
spot.ffill(inplace=True)
fut.ffill(inplace=True)

opt.fillna({
    'volume': 0,
    'oi': 0,
    'iv': opt['iv'].median()
}, inplace=True)

# ===============================
# REMOVE OUTLIERS (IQR METHOD)
# ===============================
def remove_outliers(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) &
                (df[col] <= Q3 + 1.5 * IQR)]
    return df

spot = remove_outliers(spot, ['open', 'high', 'low', 'close'])
fut  = remove_outliers(fut,  ['open', 'high', 'low', 'close'])

# ===============================
# ATM STRIKE CALCULATION
# ===============================
def get_atm_strike(ts, strike):
    if ts in spot.index:
        return abs(strike - spot.loc[ts, 'close'])
    return np.nan

opt['strike_diff'] = opt.apply(
    lambda x: get_atm_strike(x.name, x['strike']),
    axis=1
)

opt.dropna(inplace=True)

# ===============================
# FILTER ATM ±1 ±2 STRIKES
# (NIFTY STRIKE GAP = 50)
# ===============================
def filter_strikes(group):
    group = group.sort_values('strike_diff')
    atm = group.iloc[0]['strike']
    return group[
        (group['strike'] >= atm - 100) &
        (group['strike'] <= atm + 100)
    ]

opt = opt.groupby(opt.index, group_keys=False).apply(filter_strikes)
opt.drop(columns=['strike_diff'], inplace=True)

# ===============================
# ALIGN TIMESTAMPS
# ===============================
common_index = spot.index.intersection(fut.index).intersection(opt.index)

spot = spot.loc[common_index]
fut  = fut.loc[common_index]
opt  = opt.loc[common_index]

# ===============================
# SAVE CLEANED DATA
# ===============================
spot.to_csv("clean_nifty_spot_5min.csv")
fut.to_csv("clean_nifty_futures_5min.csv")
opt.to_csv("clean_nifty_options_5min.csv")

# ===============================
# MERGE DATA
# ===============================
merged = (
    spot.add_prefix("spot_")
    .join(fut.add_prefix("fut_"))
    .join(opt.add_prefix("opt_"))
)

merged.to_csv("nifty_merged_5min.csv")

# ===============================
# CLEANING REPORT
# ===============================
with open("data_cleaning_report.txt", "w") as f:
    f.write("DATA CLEANING REPORT\n")
    f.write("====================\n\n")
    f.write("• Timestamp normalized from 'date'\n")
    f.write("• Missing values handled via forward fill\n")
    f.write("• Outliers removed using IQR method\n")
    f.write("• Futures treated as continuous proxy (expiry data unavailable)\n")
    f.write("• ATM strike computed dynamically using spot close\n")
    f.write("• Options filtered to ATM ±1 ±2 strikes\n")
    f.write("• All datasets aligned on 5-minute timestamps\n")

print("Data cleaning and merging completed successfully")
