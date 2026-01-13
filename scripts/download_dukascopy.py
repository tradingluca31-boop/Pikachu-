#!/usr/bin/env python3
"""
=============================================================================
PIKACHU - DUKASCOPY DATA DOWNLOADER
=============================================================================
Telecharge les donnees historiques GRATUITES depuis Dukascopy Bank.
Disponible depuis 2003 pour la plupart des paires.

Timeframes supportes: Tick, M1, M5, M15, M30, H1, H4, D1

Usage:
    python scripts/download_dukascopy.py

Requirements:
    pip install requests pandas numpy tqdm lzma
=============================================================================
"""

import os
import struct
import lzma
from datetime import datetime, timedelta
from pathlib import Path
from io import BytesIO
import time

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

# Dates
START_DATE = datetime(2015, 1, 1)
END_DATE = datetime(2025, 12, 31)

# Symbols mapping (Dukascopy format)
SYMBOLS = {
    "XAUUSD": "XAUUSD",   # Gold
    "XAGUSD": "XAGUSD",   # Silver
    "EURUSD": "EURUSD",
    "USDJPY": "USDJPY",
    "USDCHF": "USDCHF",
    "GBPUSD": "GBPUSD",
}

# Timeframes to generate
TIMEFRAMES = ["M15", "H1", "H4", "D1"]

# Output directory
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Dukascopy base URL
DUKASCOPY_URL = "https://datafeed.dukascopy.com/datafeed/{symbol}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"


# =============================================================================
# DUKASCOPY TICK DATA PARSER
# =============================================================================

def download_hour_ticks(symbol: str, dt: datetime) -> pd.DataFrame:
    """
    Download tick data for one hour from Dukascopy.
    
    Args:
        symbol: Instrument symbol (e.g., 'XAUUSD')
        dt: Datetime for the hour to download
    
    Returns:
        DataFrame with tick data
    """
    url = DUKASCOPY_URL.format(
        symbol=symbol,
        year=dt.year,
        month=dt.month - 1,  # Dukascopy uses 0-indexed months
        day=dt.day,
        hour=dt.hour
    )
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return pd.DataFrame()
        
        if len(response.content) == 0:
            return pd.DataFrame()
        
        # Decompress LZMA
        try:
            data = lzma.decompress(response.content)
        except:
            return pd.DataFrame()
        
        # Parse binary data
        # Format: 4 bytes time (ms), 4 bytes ask, 4 bytes bid, 4 bytes ask_vol, 4 bytes bid_vol
        ticks = []
        chunk_size = 20
        
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            if len(chunk) < chunk_size:
                break
            
            time_ms, ask, bid, ask_vol, bid_vol = struct.unpack('>IIIff', chunk)
            
            tick_time = dt + timedelta(milliseconds=time_ms)
            
            # Price adjustment (Dukascopy stores as integers)
            # For Gold, divide by 1000; for forex, divide by 100000
            if symbol in ["XAUUSD", "XAGUSD"]:
                divisor = 1000.0
            else:
                divisor = 100000.0
            
            ticks.append({
                'datetime': tick_time,
                'ask': ask / divisor,
                'bid': bid / divisor,
                'ask_volume': ask_vol,
                'bid_volume': bid_vol
            })
        
        if not ticks:
            return pd.DataFrame()
        
        df = pd.DataFrame(ticks)
        df['mid'] = (df['ask'] + df['bid']) / 2
        df['spread'] = df['ask'] - df['bid']
        
        return df
        
    except Exception as e:
        return pd.DataFrame()


def download_day_ticks(symbol: str, date: datetime) -> pd.DataFrame:
    """Download all ticks for one day."""
    all_ticks = []
    
    for hour in range(24):
        dt = datetime(date.year, date.month, date.day, hour)
        ticks = download_hour_ticks(symbol, dt)
        
        if not ticks.empty:
            all_ticks.append(ticks)
        
        time.sleep(0.1)  # Rate limiting
    
    if all_ticks:
        return pd.concat(all_ticks, ignore_index=True)
    return pd.DataFrame()


def ticks_to_ohlcv(ticks: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Convert tick data to OHLCV candles.
    
    Args:
        ticks: DataFrame with tick data
        timeframe: Target timeframe (M15, H1, H4, D1)
    
    Returns:
        DataFrame with OHLCV data
    """
    if ticks.empty:
        return pd.DataFrame()
    
    # Set datetime as index
    ticks = ticks.set_index('datetime')
    
    # Resample rule
    rules = {
        'M1': '1min',
        'M5': '5min',
        'M15': '15min',
        'M30': '30min',
        'H1': '1h',
        'H4': '4h',
        'D1': '1D'
    }
    
    rule = rules.get(timeframe, '1h')
    
    # Resample using mid price
    ohlcv = ticks['mid'].resample(rule).ohlc()
    ohlcv.columns = ['open', 'high', 'low', 'close']
    
    # Volume (sum of ask + bid volume)
    ohlcv['volume'] = ticks['ask_volume'].resample(rule).sum() + ticks['bid_volume'].resample(rule).sum()
    
    # Spread (average)
    ohlcv['spread'] = ticks['spread'].resample(rule).mean()
    
    # Drop NaN rows
    ohlcv = ohlcv.dropna()
    
    return ohlcv


# =============================================================================
# MAIN DOWNLOAD FUNCTIONS
# =============================================================================

def download_symbol(symbol: str, start: datetime, end: datetime) -> dict:
    """
    Download all data for a symbol and convert to multiple timeframes.
    
    Args:
        symbol: Instrument symbol
        start: Start date
        end: End date
    
    Returns:
        Dict of DataFrames by timeframe
    """
    print(f"\n{'='*60}")
    print(f"Downloading {symbol}")
    print(f"Period: {start.date()} to {end.date()}")
    print(f"{'='*60}")
    
    all_ticks = []
    
    # Generate list of days
    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    
    # Download day by day
    for day in tqdm(days, desc=f"{symbol} ticks"):
        try:
            day_ticks = download_day_ticks(symbol, day)
            if not day_ticks.empty:
                all_ticks.append(day_ticks)
        except Exception as e:
            pass
        
        # Save progress every 100 days
        if len(all_ticks) > 0 and len(all_ticks) % 100 == 0:
            temp_df = pd.concat(all_ticks, ignore_index=True)
            temp_df.to_parquet(DATA_DIR / f"{symbol}_ticks_temp.parquet")
    
    if not all_ticks:
        print(f"  [WARN] No data downloaded for {symbol}")
        return {}
    
    # Combine all ticks
    print(f"  Combining {len(all_ticks)} days of tick data...")
    ticks_df = pd.concat(all_ticks, ignore_index=True)
    ticks_df = ticks_df.sort_values('datetime')
    ticks_df = ticks_df.drop_duplicates(subset=['datetime'])
    
    print(f"  Total ticks: {len(ticks_df):,}")
    
    # Convert to different timeframes
    results = {}
    
    for tf in TIMEFRAMES:
        print(f"  Converting to {tf}...")
        ohlcv = ticks_to_ohlcv(ticks_df, tf)
        
        if not ohlcv.empty:
            results[tf] = ohlcv
            
            # Save to CSV
            filepath = DATA_DIR / f"{symbol}_{tf}.csv"
            ohlcv.to_csv(filepath)
            print(f"    Saved: {filepath.name} ({len(ohlcv):,} candles)")
    
    # Clean up temp file
    temp_file = DATA_DIR / f"{symbol}_ticks_temp.parquet"
    if temp_file.exists():
        temp_file.unlink()
    
    return results


def download_all():
    """Download all symbols."""
    print("="*60)
    print("PIKACHU - DUKASCOPY DATA DOWNLOADER")
    print("="*60)
    print(f"Start: {START_DATE.date()}")
    print(f"End: {END_DATE.date()}")
    print(f"Symbols: {list(SYMBOLS.keys())}")
    print(f"Timeframes: {TIMEFRAMES}")
    print("")
    print("NOTE: This will take several hours for full history!")
    print("      The script saves progress every 100 days.")
    print("")
    
    start_time = time.time()
    
    all_results = {}
    
    for symbol_name, duka_symbol in SYMBOLS.items():
        results = download_symbol(duka_symbol, START_DATE, END_DATE)
        all_results[symbol_name] = results
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"Time elapsed: {elapsed/3600:.1f} hours")
    print(f"Data directory: {DATA_DIR}")
    print("")
    
    # Show what was downloaded
    for symbol, tfs in all_results.items():
        print(f"\n{symbol}:")
        for tf, df in tfs.items():
            print(f"  {tf}: {len(df):,} candles")


# =============================================================================
# QUICK DOWNLOAD (Just Gold M15)
# =============================================================================

def download_gold_m15_only():
    """Quick download - just Gold M15 from 2015."""
    print("="*60)
    print("QUICK DOWNLOAD - GOLD M15 ONLY")
    print("="*60)
    
    results = download_symbol("XAUUSD", START_DATE, END_DATE)
    
    if "M15" in results:
        print(f"\n✓ Gold M15 downloaded: {len(results['M15']):,} candles")
        print(f"  File: {DATA_DIR / 'XAUUSD_M15.csv'}")


# =============================================================================
# ALTERNATIVE: MT5 DOWNLOAD
# =============================================================================

def download_from_mt5():
    """
    Alternative: Download from MetaTrader 5.
    Requires MT5 installed and a broker account.
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 not installed. Run: pip install MetaTrader5")
        print("Note: MT5 only works on Windows")
        return
    
    # Initialize MT5
    if not mt5.initialize():
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return
    
    print("MT5 connected successfully!")
    print(f"Terminal: {mt5.terminal_info()}")
    
    # Timeframes
    tf_map = {
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1
    }
    
    symbol = "XAUUSD"
    
    for tf_name, tf_mt5 in tf_map.items():
        print(f"\nDownloading {symbol} {tf_name}...")
        
        # Get rates
        rates = mt5.copy_rates_range(
            symbol,
            tf_mt5,
            START_DATE,
            END_DATE
        )
        
        if rates is None or len(rates) == 0:
            print(f"  [WARN] No data for {tf_name}")
            continue
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('time')
        df = df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume',
            'spread': 'spread',
            'real_volume': 'real_volume'
        })
        
        # Save
        filepath = DATA_DIR / f"{symbol}_{tf_name}_mt5.csv"
        df.to_csv(filepath)
        print(f"  Saved: {filepath.name} ({len(df):,} candles)")
    
    mt5.shutdown()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            download_gold_m15_only()
        elif sys.argv[1] == "--mt5":
            download_from_mt5()
        else:
            print("Usage:")
            print("  python download_dukascopy.py          # Download all symbols")
            print("  python download_dukascopy.py --quick  # Just Gold M15")
            print("  python download_dukascopy.py --mt5    # Use MetaTrader 5")
    else:
        download_all()
