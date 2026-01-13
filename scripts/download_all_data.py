#!/usr/bin/env python3
"""
=============================================================================
PIKACHU - DOWNLOAD ALL DATA (2015-2025)
=============================================================================
Script pour telecharger toutes les donnees necessaires au projet RL Gold Trading.

Sources:
- Yahoo Finance: Prix, ETFs, Indices
- FRED: Donnees macro (TIPS, Fed, Inflation)
- CFTC: COT Data (Commitment of Traders)

Usage:
    python scripts/download_all_data.py

Requirements:
    pip install yfinance fredapi pandas numpy requests tqdm
=============================================================================
"""

import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Try to import optional packages
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("WARNING: yfinance not installed. Run: pip install yfinance")

try:
    from fredapi import Fred
    HAS_FRED = True
except ImportError:
    HAS_FRED = False
    print("WARNING: fredapi not installed. Run: pip install fredapi")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Dates
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

# Directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# FRED API Key (get yours free at: https://fred.stlouisfed.org/docs/api/api_key.html)
FRED_API_KEY = os.environ.get("FRED_API_KEY", None)


# =============================================================================
# YAHOO FINANCE DATA
# =============================================================================

YAHOO_SYMBOLS = {
    # Main asset
    "XAUUSD": "GC=F",  # Gold Futures (proxy for spot)
    
    # Cross-assets metals
    "XAGUSD": "SI=F",  # Silver Futures
    
    # Forex (via futures)
    "USDJPY": "JPY=X",  # USD/JPY
    "USDCHF": "CHF=X",  # USD/CHF
    "EURUSD": "EUR=X",  # EUR/USD
    
    # Indices
    "DXY": "DX-Y.NYB",  # Dollar Index
    "VIX": "^VIX",      # Volatility Index
    "GVZ": "^GVZ",      # Gold Volatility Index
    
    # US Treasuries
    "US10Y": "^TNX",    # 10-Year Treasury Yield
    "US02Y": "^IRX",    # 2-Year Treasury Yield (proxy)
    "US30Y": "^TYX",    # 30-Year Treasury Yield
    
    # ETFs for flows/correlations
    "GLD": "GLD",       # SPDR Gold Trust
    "IAU": "IAU",       # iShares Gold Trust
    "SPY": "SPY",       # S&P 500 ETF
    "TLT": "TLT",       # 20+ Year Treasury ETF
    "UUP": "UUP",       # Dollar Bullish ETF
    
    # Commodities
    "OIL": "CL=F",      # Crude Oil Futures
    "COPPER": "HG=F",   # Copper Futures
}

# Timeframes for OHLCV data
TIMEFRAMES = {
    "1d": "1d",
    "1h": "1h",
    "4h": "1h",  # We'll resample 1h to 4h
    "15m": "15m",
}


def download_yahoo_data(symbol: str, ticker: str, interval: str = "1d") -> pd.DataFrame:
    """
    Download data from Yahoo Finance.
    
    Args:
        symbol: Our internal symbol name
        ticker: Yahoo Finance ticker
        interval: Data interval (1d, 1h, 15m, etc.)
    
    Returns:
        DataFrame with OHLCV data
    """
    if not HAS_YFINANCE:
        print(f"  [SKIP] yfinance not installed")
        return pd.DataFrame()
    
    try:
        # For intraday data, Yahoo limits history
        if interval in ["1h", "15m"]:
            # Yahoo limits: 1h = 730 days, 15m = 60 days
            # We need to download in chunks for longer history
            if interval == "1h":
                days_per_chunk = 700
            else:
                days_per_chunk = 59
            
            all_data = []
            current_end = datetime.strptime(END_DATE, "%Y-%m-%d")
            start_limit = datetime.strptime(START_DATE, "%Y-%m-%d")
            
            while current_end > start_limit:
                current_start = current_end - timedelta(days=days_per_chunk)
                if current_start < start_limit:
                    current_start = start_limit
                
                try:
                    df = yf.download(
                        ticker,
                        start=current_start.strftime("%Y-%m-%d"),
                        end=current_end.strftime("%Y-%m-%d"),
                        interval=interval,
                        progress=False,
                        auto_adjust=True
                    )
                    if not df.empty:
                        all_data.append(df)
                except Exception as e:
                    pass
                
                current_end = current_start - timedelta(days=1)
                time.sleep(0.5)  # Rate limiting
            
            if all_data:
                data = pd.concat(all_data)
                data = data.sort_index()
                data = data[~data.index.duplicated(keep='first')]
            else:
                data = pd.DataFrame()
        else:
            # Daily data - full history available
            data = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                interval=interval,
                progress=False,
                auto_adjust=True
            )
        
        if data.empty:
            print(f"  [WARN] No data for {symbol} ({ticker})")
            return pd.DataFrame()
        
        # Clean column names
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        data.columns = [c.lower() for c in data.columns]
        
        # Add symbol column
        data['symbol'] = symbol
        
        return data
        
    except Exception as e:
        print(f"  [ERROR] {symbol}: {e}")
        return pd.DataFrame()


def download_all_yahoo_data():
    """Download all Yahoo Finance data."""
    print("\n" + "="*60)
    print("DOWNLOADING YAHOO FINANCE DATA")
    print("="*60)
    
    # Daily data for all symbols
    print("\n[1/3] Downloading DAILY data...")
    daily_data = {}
    
    for symbol, ticker in tqdm(YAHOO_SYMBOLS.items(), desc="Daily"):
        df = download_yahoo_data(symbol, ticker, "1d")
        if not df.empty:
            daily_data[symbol] = df
            # Save individual file
            filepath = DATA_DIR / f"{symbol}_D1.csv"
            df.to_csv(filepath)
        time.sleep(0.3)
    
    print(f"  Downloaded {len(daily_data)} symbols")
    
    # Hourly data (only for main symbols - limited history)
    print("\n[2/3] Downloading HOURLY data (limited to ~2 years)...")
    hourly_symbols = ["XAUUSD", "XAGUSD", "DXY", "GLD"]
    
    for symbol in tqdm(hourly_symbols, desc="Hourly"):
        if symbol in YAHOO_SYMBOLS:
            ticker = YAHOO_SYMBOLS[symbol]
            df = download_yahoo_data(symbol, ticker, "1h")
            if not df.empty:
                # Save H1
                filepath = DATA_DIR / f"{symbol}_H1.csv"
                df.to_csv(filepath)
                
                # Resample to H4
                df_h4 = df.resample('4h').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                df_h4['symbol'] = symbol
                filepath_h4 = DATA_DIR / f"{symbol}_H4.csv"
                df_h4.to_csv(filepath_h4)
        time.sleep(0.5)
    
    # 15-minute data (very limited - ~60 days)
    print("\n[3/3] Downloading 15-MINUTE data (limited to ~60 days)...")
    m15_symbols = ["XAUUSD"]
    
    for symbol in tqdm(m15_symbols, desc="M15"):
        if symbol in YAHOO_SYMBOLS:
            ticker = YAHOO_SYMBOLS[symbol]
            df = download_yahoo_data(symbol, ticker, "15m")
            if not df.empty:
                filepath = DATA_DIR / f"{symbol}_M15.csv"
                df.to_csv(filepath)
        time.sleep(0.5)
    
    return daily_data


# =============================================================================
# FRED DATA (Macro)
# =============================================================================

FRED_SERIES = {
    # Real Interest Rates (TIPS)
    "TIPS_10Y": "DFII10",          # 10-Year TIPS yield
    "TIPS_5Y": "DFII5",            # 5-Year TIPS yield
    "TIPS_20Y": "DFII20",          # 20-Year TIPS yield
    
    # Breakeven Inflation
    "BREAKEVEN_10Y": "T10YIE",     # 10-Year Breakeven Inflation
    "BREAKEVEN_5Y": "T5YIE",       # 5-Year Breakeven Inflation
    "BREAKEVEN_5Y5Y": "T5YIFR",    # 5-Year, 5-Year Forward Inflation
    
    # Fed Policy
    "FED_FUNDS": "FEDFUNDS",       # Federal Funds Rate
    "FED_FUNDS_UPPER": "DFEDTARU", # Fed Funds Upper Bound
    "FED_FUNDS_LOWER": "DFEDTARL", # Fed Funds Lower Bound
    
    # Fed Balance Sheet
    "FED_ASSETS": "WALCL",         # Fed Total Assets
    
    # Treasury Yields
    "TREASURY_10Y": "DGS10",       # 10-Year Treasury
    "TREASURY_2Y": "DGS2",         # 2-Year Treasury
    "TREASURY_30Y": "DGS30",       # 30-Year Treasury
    "TREASURY_3M": "DGS3MO",       # 3-Month Treasury
    
    # Credit Spreads
    "HY_SPREAD": "BAMLH0A0HYM2",   # High Yield Spread
    "IG_SPREAD": "BAMLC0A4CBBB",   # Investment Grade Spread
    
    # Economic Indicators
    "CPI": "CPIAUCSL",             # CPI All Urban Consumers
    "CORE_CPI": "CPILFESL",        # Core CPI
    "UNEMPLOYMENT": "UNRATE",       # Unemployment Rate
    "GDP": "GDP",                   # GDP
    
    # Money Supply
    "M2": "M2SL",                  # M2 Money Stock
}


def download_fred_data():
    """Download all FRED data."""
    print("\n" + "="*60)
    print("DOWNLOADING FRED DATA (Macro)")
    print("="*60)
    
    if not HAS_FRED:
        print("[SKIP] fredapi not installed")
        return {}
    
    if not FRED_API_KEY:
        print("[WARN] FRED_API_KEY not set. Get your free key at:")
        print("       https://fred.stlouisfed.org/docs/api/api_key.html")
        print("       Then set: export FRED_API_KEY='your_key'")
        print("")
        print("       Trying without API key (may fail)...")
    
    try:
        fred = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else Fred()
    except Exception as e:
        print(f"[ERROR] Could not connect to FRED: {e}")
        return {}
    
    fred_data = {}
    
    for name, series_id in tqdm(FRED_SERIES.items(), desc="FRED"):
        try:
            data = fred.get_series(
                series_id,
                observation_start=START_DATE,
                observation_end=END_DATE
            )
            
            if data is not None and len(data) > 0:
                df = pd.DataFrame({name: data})
                df.index.name = 'date'
                fred_data[name] = df
                
                # Save individual file
                filepath = DATA_DIR / f"FRED_{name}.csv"
                df.to_csv(filepath)
            
        except Exception as e:
            print(f"  [WARN] {name}: {e}")
        
        time.sleep(0.2)  # Rate limiting
    
    print(f"  Downloaded {len(fred_data)} series")
    
    # Combine all FRED data into one file
    if fred_data:
        combined = pd.concat(fred_data.values(), axis=1)
        combined.to_csv(DATA_DIR / "FRED_all.csv")
        print(f"  Combined file saved: FRED_all.csv")
    
    return fred_data


# =============================================================================
# COT DATA (CFTC)
# =============================================================================

def download_cot_data():
    """Download COT (Commitment of Traders) data from CFTC."""
    print("\n" + "="*60)
    print("DOWNLOADING COT DATA (CFTC)")
    print("="*60)
    
    # COT data URLs (Disaggregated Futures Only)
    base_url = "https://www.cftc.gov/files/dea/history"
    
    # Years to download
    years = range(2015, 2026)
    
    all_cot_data = []
    
    for year in tqdm(years, desc="COT Years"):
        try:
            # Try different formats
            urls_to_try = [
                f"{base_url}/fut_disagg_txt_{year}.zip",
                f"{base_url}/deafut_txt_{year}.zip",
            ]
            
            for url in urls_to_try:
                try:
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        # Save zip file
                        zip_path = DATA_DIR / f"cot_{year}.zip"
                        with open(zip_path, 'wb') as f:
                            f.write(response.content)
                        
                        # Extract and read
                        import zipfile
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(DATA_DIR / "cot_temp")
                        
                        # Find and read the text file
                        temp_dir = DATA_DIR / "cot_temp"
                        for file in temp_dir.glob("*.txt"):
                            try:
                                df = pd.read_csv(file)
                                # Filter for Gold
                                gold_df = df[df['Market_and_Exchange_Names'].str.contains('GOLD', case=False, na=False)]
                                if not gold_df.empty:
                                    all_cot_data.append(gold_df)
                            except:
                                pass
                        
                        # Cleanup
                        import shutil
                        if temp_dir.exists():
                            shutil.rmtree(temp_dir)
                        zip_path.unlink()
                        
                        break
                except:
                    continue
                    
        except Exception as e:
            print(f"  [WARN] {year}: {e}")
        
        time.sleep(0.5)
    
    if all_cot_data:
        cot_df = pd.concat(all_cot_data, ignore_index=True)
        
        # Clean and save
        if 'Report_Date_as_YYYY-MM-DD' in cot_df.columns:
            cot_df['date'] = pd.to_datetime(cot_df['Report_Date_as_YYYY-MM-DD'])
        elif 'As_of_Date_In_Form_YYMMDD' in cot_df.columns:
            cot_df['date'] = pd.to_datetime(cot_df['As_of_Date_In_Form_YYMMDD'], format='%y%m%d')
        
        cot_df = cot_df.sort_values('date')
        cot_df.to_csv(DATA_DIR / "COT_Gold.csv", index=False)
        print(f"  Downloaded {len(cot_df)} COT reports for Gold")
        return cot_df
    else:
        print("  [WARN] Could not download COT data automatically")
        print("  Manual download: https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm")
        return pd.DataFrame()


# =============================================================================
# GLD HOLDINGS (ETF Flows)
# =============================================================================

def download_gld_holdings():
    """Download GLD holdings data for ETF flows."""
    print("\n" + "="*60)
    print("DOWNLOADING GLD HOLDINGS (ETF Flows)")
    print("="*60)
    
    # The official source is SPDR website, but we can approximate from Yahoo
    # For accurate holdings, scrape: https://www.spdrgoldshares.com/usa/historical-data/
    
    if not HAS_YFINANCE:
        print("[SKIP] yfinance not installed")
        return pd.DataFrame()
    
    try:
        gld = yf.Ticker("GLD")
        
        # Get historical data
        hist = gld.history(start=START_DATE, end=END_DATE)
        
        if not hist.empty:
            # Calculate proxy for holdings changes using volume
            hist['volume_sma20'] = hist['Volume'].rolling(20).mean()
            hist['volume_ratio'] = hist['Volume'] / hist['volume_sma20']
            
            # High volume days often correlate with large inflows/outflows
            hist['flow_proxy'] = np.where(
                hist['Close'] > hist['Close'].shift(1),
                hist['volume_ratio'],  # Positive flow on up days
                -hist['volume_ratio']  # Negative flow on down days
            )
            
            hist.to_csv(DATA_DIR / "GLD_holdings_proxy.csv")
            print(f"  Downloaded GLD data with flow proxy ({len(hist)} days)")
            print("  Note: For accurate holdings, manually download from SPDR website")
            
            return hist
    
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    return pd.DataFrame()


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_data():
    """Validate downloaded data."""
    print("\n" + "="*60)
    print("VALIDATING DOWNLOADED DATA")
    print("="*60)
    
    validation_results = []
    
    # Check all CSV files
    for filepath in DATA_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(filepath, nrows=5)
            rows = len(pd.read_csv(filepath))
            
            validation_results.append({
                'file': filepath.name,
                'rows': rows,
                'columns': len(df.columns),
                'status': 'OK' if rows > 100 else 'LOW'
            })
        except Exception as e:
            validation_results.append({
                'file': filepath.name,
                'rows': 0,
                'columns': 0,
                'status': f'ERROR: {e}'
            })
    
    # Print results
    print("\nDownloaded files:")
    print("-" * 60)
    
    for r in sorted(validation_results, key=lambda x: x['file']):
        status_icon = "✓" if r['status'] == 'OK' else "!" if r['status'] == 'LOW' else "✗"
        print(f"  {status_icon} {r['file']:<30} {r['rows']:>8} rows")
    
    print("-" * 60)
    total_files = len(validation_results)
    ok_files = sum(1 for r in validation_results if r['status'] == 'OK')
    print(f"Total: {ok_files}/{total_files} files OK")
    
    return validation_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function to download all data."""
    print("="*60)
    print("PIKACHU - DATA DOWNLOAD SCRIPT")
    print("="*60)
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Output directory: {DATA_DIR}")
    print("")
    
    start_time = time.time()
    
    # 1. Yahoo Finance data
    yahoo_data = download_all_yahoo_data()
    
    # 2. FRED data
    fred_data = download_fred_data()
    
    # 3. COT data
    cot_data = download_cot_data()
    
    # 4. GLD Holdings
    gld_data = download_gld_holdings()
    
    # 5. Validate
    validation = validate_data()
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"Time elapsed: {elapsed:.1f} seconds")
    print(f"Data directory: {DATA_DIR}")
    print("")
    print("Next steps:")
    print("  1. Get FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html")
    print("  2. Set environment variable: export FRED_API_KEY='your_key'")
    print("  3. Re-run script to download FRED data")
    print("  4. For accurate GLD holdings, download from SPDR website")
    print("  5. For MT5 data, use the mt5_loader.py script")
    print("")


if __name__ == "__main__":
    main()
