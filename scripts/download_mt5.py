#!/usr/bin/env python3
"""
=============================================================================
PIKACHU - METATRADER 5 DATA DOWNLOADER
=============================================================================
Telecharge les donnees historiques depuis MetaTrader 5.
Compatible avec compte demo FTMO ou tout autre broker.

Donnees telechargeables:
- XAUUSD (Gold) : M15, H1, H4, D1
- XAGUSD (Silver) : M15, H1, H4, D1
- USDJPY, USDCHF, EURUSD : D1

Usage:
    python scripts/download_mt5.py

Requirements:
    pip install MetaTrader5 pandas numpy tqdm
    
Note: MetaTrader5 fonctionne uniquement sur Windows!
=============================================================================
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

# Check if we're on Windows
if sys.platform != 'win32':
    print("ERROR: MetaTrader5 only works on Windows!")
    print("For Linux/Mac, use Wine or a Windows VM.")
    sys.exit(1)

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 not installed!")
    print("Run: pip install MetaTrader5")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Dates
START_DATE = datetime(2015, 1, 1)
END_DATE = datetime(2025, 12, 31)

# Output directory
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Symbols to download
SYMBOLS_CONFIG = {
    # Main asset - all timeframes
    "XAUUSD": {
        "timeframes": ["M15", "H1", "H4", "D1"],
        "description": "Gold"
    },
    # Silver - all timeframes (cross-asset important)
    "XAGUSD": {
        "timeframes": ["M15", "H1", "H4", "D1"],
        "description": "Silver"
    },
    # Forex - daily only
    "USDJPY": {
        "timeframes": ["D1"],
        "description": "USD/JPY"
    },
    "USDCHF": {
        "timeframes": ["D1"],
        "description": "USD/CHF"
    },
    "EURUSD": {
        "timeframes": ["D1"],
        "description": "EUR/USD"
    },
}

# MT5 Timeframe mapping
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}


# =============================================================================
# MT5 CONNECTION
# =============================================================================

def connect_mt5(login: int = None, password: str = None, server: str = None) -> bool:
    """
    Connect to MetaTrader 5.
    
    Args:
        login: MT5 account number (optional, uses default if not provided)
        password: MT5 password (optional)
        server: MT5 server name (optional)
    
    Returns:
        True if connected successfully
    """
    # Initialize MT5
    if not mt5.initialize():
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return False
    
    # If credentials provided, login
    if login and password and server:
        authorized = mt5.login(login, password=password, server=server)
        if not authorized:
            print(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return False
    
    # Print connection info
    terminal_info = mt5.terminal_info()
    account_info = mt5.account_info()
    
    print("="*60)
    print("MT5 CONNECTED SUCCESSFULLY")
    print("="*60)
    print(f"Terminal: {terminal_info.name}")
    print(f"Company: {terminal_info.company}")
    print(f"Account: {account_info.login}")
    print(f"Server: {account_info.server}")
    print(f"Balance: {account_info.balance} {account_info.currency}")
    print(f"Leverage: 1:{account_info.leverage}")
    print("="*60)
    
    return True


def disconnect_mt5():
    """Disconnect from MetaTrader 5."""
    mt5.shutdown()
    print("\nMT5 disconnected.")


# =============================================================================
# DATA DOWNLOAD
# =============================================================================

def get_symbol_info(symbol: str) -> dict:
    """Get symbol information from MT5."""
    info = mt5.symbol_info(symbol)
    
    if info is None:
        # Try alternative symbol names
        alternatives = {
            "XAUUSD": ["GOLD", "XAUUSDm", "XAUUSD.r", "XAUUSD_", "XAUUSD.a", "XAUUSDc"],
            "XAGUSD": ["SILVER", "XAGUSDm", "XAGUSD.r", "XAGUSD_", "XAGUSD.a", "XAGUSDc"],
            "USDJPY": ["USDJPYm", "USDJPY.r", "USDJPY_", "USDJPY.a"],
            "USDCHF": ["USDCHFm", "USDCHF.r", "USDCHF_", "USDCHF.a"],
            "EURUSD": ["EURUSDm", "EURUSD.r", "EURUSD_", "EURUSD.a"],
        }
        
        for alt in alternatives.get(symbol, []):
            info = mt5.symbol_info(alt)
            if info is not None:
                print(f"  Using alternative symbol: {alt}")
                return {"symbol": alt, "info": info}
        
        return None
    
    return {"symbol": symbol, "info": info}


def download_symbol_timeframe(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Download data for a symbol and timeframe.
    
    Args:
        symbol: Instrument symbol
        timeframe: Timeframe string (M15, H1, H4, D1)
    
    Returns:
        DataFrame with OHLCV data
    """
    # Get MT5 timeframe constant
    tf_mt5 = TIMEFRAME_MAP.get(timeframe)
    if tf_mt5 is None:
        print(f"  [ERROR] Unknown timeframe: {timeframe}")
        return pd.DataFrame()
    
    # Check symbol exists
    symbol_info = get_symbol_info(symbol)
    if symbol_info is None:
        print(f"  [ERROR] Symbol not found: {symbol}")
        return pd.DataFrame()
    
    actual_symbol = symbol_info["symbol"]
    
    # Enable symbol if needed
    if not symbol_info["info"].visible:
        if not mt5.symbol_select(actual_symbol, True):
            print(f"  [ERROR] Failed to select symbol: {actual_symbol}")
            return pd.DataFrame()
    
    # Download data
    rates = mt5.copy_rates_range(
        actual_symbol,
        tf_mt5,
        START_DATE,
        END_DATE
    )
    
    if rates is None or len(rates) == 0:
        error = mt5.last_error()
        print(f"  [ERROR] No data: {error}")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    
    # Convert time to datetime
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df = df.set_index('datetime')
    
    # Rename columns
    df = df.rename(columns={
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'tick_volume': 'volume',
        'spread': 'spread',
        'real_volume': 'real_volume'
    })
    
    # Select columns
    df = df[['open', 'high', 'low', 'close', 'volume', 'spread']]
    
    # Add symbol column
    df['symbol'] = symbol
    
    return df


def download_all_data():
    """Download all configured symbols and timeframes."""
    print("\n" + "="*60)
    print("DOWNLOADING DATA FROM MT5")
    print("="*60)
    print(f"Period: {START_DATE.date()} to {END_DATE.date()}")
    print(f"Output: {DATA_DIR}")
    print("")
    
    results = {}
    
    for symbol, config in SYMBOLS_CONFIG.items():
        print(f"\n[{symbol}] {config['description']}")
        print("-" * 40)
        
        results[symbol] = {}
        
        for tf in config["timeframes"]:
            print(f"  Downloading {tf}...", end=" ")
            
            df = download_symbol_timeframe(symbol, tf)
            
            if df.empty:
                print("FAILED")
                continue
            
            # Save to CSV
            filename = f"{symbol}_{tf}.csv"
            filepath = DATA_DIR / filename
            df.to_csv(filepath)
            
            results[symbol][tf] = len(df)
            print(f"OK ({len(df):,} candles)")
    
    return results


# =============================================================================
# VALIDATION
# =============================================================================

def validate_data():
    """Validate downloaded data."""
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)
    
    required_files = [
        ("XAUUSD_D1.csv", 2500),   # ~10 years of daily
        ("XAUUSD_H4.csv", 15000),  # ~10 years of 4H
        ("XAUUSD_H1.csv", 60000),  # ~10 years of 1H
        ("XAUUSD_M15.csv", 200000), # ~10 years of M15
        ("XAGUSD_D1.csv", 2500),   # Silver daily
        ("XAGUSD_H4.csv", 15000),  # Silver 4H
        ("XAGUSD_H1.csv", 60000),  # Silver 1H
        ("XAGUSD_M15.csv", 200000), # Silver M15
    ]
    
    all_ok = True
    
    for filename, min_rows in required_files:
        filepath = DATA_DIR / filename
        
        if not filepath.exists():
            print(f"  ✗ {filename} - MISSING")
            all_ok = False
            continue
        
        df = pd.read_csv(filepath)
        rows = len(df)
        
        if rows < min_rows:
            print(f"  ! {filename} - {rows:,} rows (expected >{min_rows:,})")
        else:
            print(f"  ✓ {filename} - {rows:,} rows")
    
    return all_ok


# =============================================================================
# DATA QUALITY CHECK
# =============================================================================

def check_data_quality(symbol: str = "XAUUSD"):
    """Check data quality for a symbol."""
    print("\n" + "="*60)
    print(f"DATA QUALITY CHECK - {symbol}")
    print("="*60)
    
    for tf in ["M15", "H1", "H4", "D1"]:
        filepath = DATA_DIR / f"{symbol}_{tf}.csv"
        
        if not filepath.exists():
            continue
        
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        print(f"\n{tf}:")
        print(f"  Period: {df.index[0]} to {df.index[-1]}")
        print(f"  Rows: {len(df):,}")
        
        # Check for gaps
        if tf == "D1":
            expected_freq = pd.Timedelta(days=1)
        elif tf == "H4":
            expected_freq = pd.Timedelta(hours=4)
        elif tf == "H1":
            expected_freq = pd.Timedelta(hours=1)
        elif tf == "M15":
            expected_freq = pd.Timedelta(minutes=15)
        
        # Count gaps (excluding weekends for daily)
        gaps = df.index.to_series().diff()
        
        if tf == "D1":
            # Allow gaps of 1-3 days (weekends)
            large_gaps = gaps[gaps > pd.Timedelta(days=4)]
        else:
            # For intraday, allow weekend gaps
            large_gaps = gaps[gaps > pd.Timedelta(days=3)]
        
        if len(large_gaps) > 0:
            print(f"  Large gaps: {len(large_gaps)}")
            for idx, gap in large_gaps.head(3).items():
                print(f"    - {idx}: {gap}")
        else:
            print(f"  Gaps: None (clean data)")
        
        # Price statistics
        print(f"  Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
        print(f"  Avg spread: {df['spread'].mean():.1f} points")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function."""
    print("="*60)
    print("PIKACHU - MT5 DATA DOWNLOADER")
    print("="*60)
    print("")
    print("This script downloads historical data from MetaTrader 5.")
    print("Make sure MT5 is running and connected to your broker.")
    print("")
    
    # Connect to MT5
    if not connect_mt5():
        print("\nFailed to connect to MT5!")
        print("\nTroubleshooting:")
        print("1. Make sure MetaTrader 5 is installed and running")
        print("2. Login to your demo account in MT5")
        print("3. Allow automated trading (Tools > Options > Expert Advisors)")
        print("4. Run this script again")
        return
    
    try:
        # Download data
        start_time = time.time()
        results = download_all_data()
        
        # Validate
        validate_data()
        
        # Quality check
        check_data_quality("XAUUSD")
        check_data_quality("XAGUSD")
        
        # Summary
        elapsed = time.time() - start_time
        
        print("\n" + "="*60)
        print("DOWNLOAD COMPLETE")
        print("="*60)
        print(f"Time: {elapsed:.1f} seconds")
        print(f"Data saved to: {DATA_DIR}")
        print("")
        print("Summary:")
        for symbol, tfs in results.items():
            if tfs:
                total = sum(tfs.values())
                print(f"  {symbol}: {total:,} candles")
        
    finally:
        disconnect_mt5()


def print_available_symbols():
    """Print all available symbols in MT5."""
    if not connect_mt5():
        return
    
    try:
        symbols = mt5.symbols_get()
        
        print("\n" + "="*60)
        print("AVAILABLE SYMBOLS")
        print("="*60)
        
        # Filter for gold/silver/forex
        gold = [s.name for s in symbols if 'XAU' in s.name or 'GOLD' in s.name]
        silver = [s.name for s in symbols if 'XAG' in s.name or 'SILVER' in s.name]
        forex = [s.name for s in symbols if any(x in s.name for x in ['USD', 'EUR', 'GBP', 'JPY', 'CHF'])]
        
        print("\nGold symbols:")
        for s in gold[:10]:
            print(f"  {s}")
        
        print("\nSilver symbols:")
        for s in silver[:10]:
            print(f"  {s}")
        
        print(f"\nForex symbols: {len(forex)} available")
        print("Examples:", forex[:5])
        
    finally:
        disconnect_mt5()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--symbols":
        print_available_symbols()
    else:
        main()
