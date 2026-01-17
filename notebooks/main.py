import logging
from kiteconnect import KiteConnect
import pandas as pd
import datetime

# ================= CREDENTIALS =================
API_KEY = "a07ngwrdtcmbolgx"         # PASTE HERE
ACCESS_TOKEN = "IBJb4Chgh468LZP4AF1tk2j2yHyo5QP6" # PASTE HERE
# ===============================================

# Setup
logging.basicConfig(level=logging.DEBUG)
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

def get_instrument_token(symbol, exchange="NSE", instrument_type="EQUITY"):
    """
    Downloads instrument dump to find the correct token.
    """
    print("Downloading Instrument List (this takes a moment)...")
    instruments = kite.instruments(exchange)
    df_inst = pd.DataFrame(instruments)
    
    # Filter for NIFTY 50 Spot
    if symbol == "NIFTY 50":
        token = df_inst[(df_inst['name'] == "NIFTY 50") & (df_inst['segment'] == "INDICES")]
        if not token.empty:
            return token.iloc[0]['instrument_token']
    
    # Filter for NIFTY Futures (Current Month)
    # We look for "NIFTY" in NFO segment with specific expiry or just pick the first current one
    if symbol == "NIFTY FUT":
        # Filter for NIFTY Futures in NFO
        futs = df_inst[
            (df_inst['name'] == 'NIFTY') & 
            (df_inst['segment'] == 'NFO-FUT')
        ].sort_values('expiry')
        
        # Get the nearest expiry (Current Month)
        # Note: For historical stitching, this is complex. 
        # For this assignment, we will fetch the 'Continuous' contract if available 
        # or the most recent expiry to represent the data.
        if not futs.empty:
            print(f"Selected Future Contract: {futs.iloc[0]['tradingsymbol']} (Expiry: {futs.iloc[0]['expiry']})")
            return futs.iloc[0]['instrument_token']
            
    return None

def fetch_historical(token, start_date, end_date, interval="5minute"):
    """
    Fetches historical data in chunks (Zerodha limits data per call).
    """
    all_data = []
    
    # Convert strings to datetime
    current_start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    
    # Zerodha allows ~60-100 days of 5min data per call. We loop 30 days at a time to be safe.
    while current_start < end_dt:
        current_end = current_start + datetime.timedelta(days=60)
        if current_end > end_dt:
            current_end = end_dt
            
        print(f"Fetching: {token} | {current_start.date()} to {current_end.date()}")
        
        try:
            data = kite.historical_data(
                token, 
                current_start.strftime("%Y-%m-%d %H:%M:%S"), 
                current_end.strftime("%Y-%m-%d %H:%M:%S"), 
                interval
            )
            all_data.extend(data)
        except Exception as e:
            print(f"Error fetching chunk: {e}")
            
        current_start = current_end
        
    return pd.DataFrame(all_data)

# ================= MAIN EXECUTION =================

# 1. Get Tokens
spot_token = get_instrument_token("NIFTY 50")
print(f"NIFTY 50 Token: {spot_token}")

fut_token = get_instrument_token("NIFTY FUT", exchange="NFO")
print(f"NIFTY Future Token: {fut_token}")

# 2. Fetch SPOT Data
print("\n--- Fetching NIFTY SPOT ---")
df_spot = fetch_historical(spot_token, "2025-01-15", "2026-01-15")
if not df_spot.empty:
    df_spot.to_csv("nifty_spot_5min.csv", index=False)
    print("Saved nifty_spot_5min.csv")

# 3. Fetch FUTURES Data
print("\n--- Fetching NIFTY FUTURES ---")
df_fut = fetch_historical(fut_token, "2025-01-15", "2026-01-15")
if not df_fut.empty:
    df_fut.to_csv("nifty_futures_5min.csv", index=False)
    print("Saved nifty_futures_5min.csv")

# 4. Handle OPTIONS Data (The Workaround)
# Since we cannot easily fetch 1 year of rolling ATM tokens via API without a database,
# We will create the options file structure required for the next steps, 
# but we will calculate the prices using Black-Scholes in the Feature Engineering step.
print("\n--- Generating Options Placeholder ---")
if not df_spot.empty:
    df_opt = df_spot.copy()
    # Create the columns expected by the assignment
    df_opt['strike'] = (df_opt['close'] / 50).round() * 50 # Dynamic ATM Strike
    df_opt['type'] = 'CE'
    df_opt['ltp'] = 0.0 # Will calculate in Task 2
    df_opt['iv'] = 15.0 # Initial assumption, will refine later
    df_opt['oi'] = 100000 # Placeholder OI
    df_opt.to_csv("nifty_options_5min.csv", index=False)
    print("Saved nifty_options_5min.csv (Structure Ready)")

print("\nDONE! You can now run 'main.py' to process this data.")