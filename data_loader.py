"""
data_loader.py — HELPER: Data Ingestion, Pair Configuration & Preprocessing
=============================================================================

Role: Provides all data infrastructure for the project. Downloads price data from
      Yahoo Finance (or loads from local CSV cache), synchronizes timestamps,
      splits into formation/trading periods, and computes summary statistics.
      This module defines the canonical TARGET_PAIRS and date ranges used everywhere.

Data Sources (prioritized):
  1. Local CSV cache in DataSet/ (avoids network latency, rate limits)
  2. Yahoo Finance API via yfinance (fallback for missing pairs)

Target Pairs (7 canonical equity pairs from the papers):
  AAP-IBM, EWA-EWC, NKE-AES, MRK-VZ, GS-CVX, KO-PEP, XOM-CVX

Period Structure:
  Formation (In-Sample) : 2009-09-01 → 2017-08-30 (~2,014 trading days, ~8 years)
  Trading (Out-of-Sample): 2017-08-31 → 2021-08-31 (~1,007 trading days, ~4 years)

Key Exports:
  - PairData             : Structured container for all pair data (prices, arrays, dates, stats)
  - fetch_pair_data()    : Download and preprocess a single pair
  - fetch_all_target_pairs() : Batch-download all target pairs
  - print_data_summary() : Pretty-print data quality report
  - TARGET_PAIRS         : List of (ticker1, ticker2) tuples
  - DEFAULT_FORMATION_START/END, DEFAULT_TRADING_START/END : Date range constants
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


# Canonical Target Stock / Sector Pairs (7 Pairs)
CANONICAL_PAIRS: List[Tuple[str, str]] = [
    ("AAP", "IBM"),
    ("EWA", "EWC"),
    ("NKE", "AES"),
    ("MRK", "VZ"),
    ("GS", "CVX"),
    ("KO", "PEP"),
    ("XOM", "CVX"),
]

# Commodity & Energy Pairs (4 Pairs)
COMMODITY_ENERGY_PAIRS: List[Tuple[str, str]] = [
    ("GLD", "SLV"),   # Gold vs Silver (Metals ratio, dynamic hedging, mean reversion)
    ("XLE", "XOP"),   # Energy ETF vs Exploration & Production (Sensitivity to commodity shocks)
    ("XLE", "OIH"),   # Energy ETF vs Oil Services (Value chain stages, varying hedge ratio)
    ("USO", "XLE"),   # Oil Commodity vs Energy Equity (Cross-asset temporary vs permanent divergence)
]

# Validated Passing Pairs from Empirical Screening Research (3 Pairs)
VALIDATED_PASSING_PAIRS: List[Tuple[str, str]] = [
    ("MS", "GS"),     # Morgan Stanley vs Goldman Sachs (Investment banking, passes ALL 4 STRICT criteria)
    ("V", "MA"),       # Visa vs Mastercard (Payment network duopoly, passes PCI-Aware test)
    ("HD", "LOW"),     # Home Depot vs Lowe's (Retail duopoly, passes PCI-Aware test)
]

# Complete 14-Pair Cross-Asset Universe (11 Benchmark + 3 Validated Passing Pairs)
ALL_PAIRS: List[Tuple[str, str]] = CANONICAL_PAIRS + COMMODITY_ENERGY_PAIRS + VALIDATED_PASSING_PAIRS

# Default evaluated target pairs
TARGET_PAIRS: List[Tuple[str, str]] = ALL_PAIRS

# Default Date Ranges (12-Year Analysis: 8-Year In-Sample Formation Period, 4-Year Trading Period)
DEFAULT_FORMATION_START = "2009-09-01"
DEFAULT_FORMATION_END = "2017-08-30"  # 2,014 trading days
DEFAULT_TRADING_START = "2017-08-31"
DEFAULT_TRADING_END = "2021-08-31"    # 1,007 trading days


@dataclass
class PairData:
    """
    Structured container for processed pair data.
    
    Attributes
    ----------
    ticker1 : str
        Ticker symbol of Asset 1 (X1, driver / independent leg).
    ticker2 : str
        Ticker symbol of Asset 2 (X2, dependent / response leg).
    df_all : pd.DataFrame
        Full synchronous DataFrame containing both series with DatetimeIndex.
    X1 : pd.DataFrame
        DataFrame of Asset 1 prices across full date range.
    X2 : pd.DataFrame
        DataFrame of Asset 2 prices across full date range.
    df_formation : pd.DataFrame
        In-sample formation period DataFrame.
    df_trading : pd.DataFrame
        Out-of-sample trading period DataFrame.
    x1_formation, x2_formation : np.ndarray
        1D numpy arrays of in-sample formation prices.
    x1_trading, x2_trading : np.ndarray
        1D numpy arrays of out-of-sample trading prices.
    x1_all, x2_all : np.ndarray
        1D numpy arrays of full-period prices.
    dates_all, dates_formation, dates_trading : pd.DatetimeIndex
        Synchronous DatetimeIndexes for the respective periods.
    summary : dict
        Detailed metadata, date ranges, data health and correlation statistics.
    """
    ticker1: str
    ticker2: str
    df_all: pd.DataFrame
    X1: pd.DataFrame
    X2: pd.DataFrame
    df_formation: pd.DataFrame
    df_trading: pd.DataFrame
    x1_formation: np.ndarray
    x2_formation: np.ndarray
    x1_trading: np.ndarray
    x2_trading: np.ndarray
    x1_all: np.ndarray
    x2_all: np.ndarray
    dates_all: pd.DatetimeIndex
    dates_formation: pd.DatetimeIndex
    dates_trading: pd.DatetimeIndex
    summary: Dict[str, Union[str, int, float]]


def _download_adj_close_batch(
    tickers: List[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Download daily adjusted close prices for multiple tickers in a single batch request.
    """
    query_end = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    unique_tickers = sorted(list(set(tickers)))
    
    raw_df = yf.download(
        tickers=unique_tickers,
        start=start_date,
        end=query_end,
        auto_adjust=False,
        progress=False,
    )
    
    if raw_df.empty:
        raise ValueError(f"No data returned from yfinance for tickers {unique_tickers} between {start_date} and {end_date}.")
    
    # Extract adjusted close prices
    if "Adj Close" in raw_df.columns:
        adj_close = raw_df["Adj Close"]
    elif "Close" in raw_df.columns:
        adj_close = raw_df["Close"]
    else:
        adj_close = raw_df
        
    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame(name=unique_tickers[0])
    elif isinstance(adj_close, pd.DataFrame) and len(unique_tickers) == 1:
        if adj_close.shape[1] == 1 and adj_close.columns[0] != unique_tickers[0]:
            adj_close.columns = [unique_tickers[0]]
            
    adj_close = adj_close.loc[start_date:end_date]
    return adj_close


def _process_pair_from_raw(
    raw_df: pd.DataFrame,
    ticker1: str,
    ticker2: str,
    formation_start: str,
    formation_end: str,
    trading_start: str,
    trading_end: str,
) -> PairData:
    """Clean, forward-fill, dropna, and partition a pair from raw price DataFrame."""
    pair_df = raw_df[[ticker1, ticker2]].copy()
    pair_df.index = pd.to_datetime(pair_df.index)
    pair_df.sort_index(inplace=True)
    
    raw_total_days = len(pair_df)
    missing_before_x1 = int(pair_df[ticker1].isna().sum())
    missing_before_x2 = int(pair_df[ticker2].isna().sum())
    
    # Forward-fill then backward-fill/drop remaining NaNs
    clean_df = pair_df.ffill().bfill().dropna()
    clean_total_days = len(clean_df)
    
    # Structure into separate DataFrames X1 and X2
    df_x1 = clean_df[[ticker1]].copy()
    df_x2 = clean_df[[ticker2]].copy()
    
    # Split into Formation (in-sample) and Trading (out-of-sample)
    df_formation = clean_df.loc[formation_start:formation_end].copy()
    df_trading = clean_df.loc[trading_start:trading_end].copy()
    
    if len(df_formation) == 0:
        raise ValueError(f"Formation slice [{formation_start}, {formation_end}] contains 0 rows for {ticker1}-{ticker2}!")
    if len(df_trading) == 0:
        raise ValueError(f"Trading slice [{trading_start}, {trading_end}] contains 0 rows for {ticker1}-{ticker2}!")
        
    x1_form = df_formation[ticker1].to_numpy(dtype=float)
    x2_form = df_formation[ticker2].to_numpy(dtype=float)
    
    x1_trade = df_trading[ticker1].to_numpy(dtype=float)
    x2_trade = df_trading[ticker2].to_numpy(dtype=float)
    
    x1_all = clean_df[ticker1].to_numpy(dtype=float)
    x2_all = clean_df[ticker2].to_numpy(dtype=float)
    
    # Return correlations
    ret_form_x1 = df_formation[ticker1].pct_change().dropna()
    ret_form_x2 = df_formation[ticker2].pct_change().dropna()
    corr_form = float(ret_form_x1.corr(ret_form_x2))
    
    ret_trade_x1 = df_trading[ticker1].pct_change().dropna()
    ret_trade_x2 = df_trading[ticker2].pct_change().dropna()
    corr_trade = float(ret_trade_x1.corr(ret_trade_x2))
    
    summary = {
        "ticker1": ticker1,
        "ticker2": ticker2,
        "total_start_date": clean_df.index[0].strftime("%Y-%m-%d"),
        "total_end_date": clean_df.index[-1].strftime("%Y-%m-%d"),
        "total_trading_days": clean_total_days,
        "raw_calendar_rows": raw_total_days,
        f"missing_before_{ticker1}": missing_before_x1,
        f"missing_before_{ticker2}": missing_before_x2,
        "missing_after_cleaning": int(clean_df.isna().sum().sum()),
        "formation_start_date": df_formation.index[0].strftime("%Y-%m-%d"),
        "formation_end_date": df_formation.index[-1].strftime("%Y-%m-%d"),
        "formation_trading_days": len(df_formation),
        "trading_start_date": df_trading.index[0].strftime("%Y-%m-%d"),
        "trading_end_date": df_trading.index[-1].strftime("%Y-%m-%d"),
        "trading_period_days": len(df_trading),
        "formation_return_corr": corr_form,
        "trading_return_corr": corr_trade,
    }
    
    return PairData(
        ticker1=ticker1,
        ticker2=ticker2,
        df_all=clean_df,
        X1=df_x1,
        X2=df_x2,
        df_formation=df_formation,
        df_trading=df_trading,
        x1_formation=x1_form,
        x2_formation=x2_form,
        x1_trading=x1_trade,
        x2_trading=x2_trade,
        x1_all=x1_all,
        x2_all=x2_all,
        dates_all=clean_df.index,
        dates_formation=df_formation.index,
        dates_trading=df_trading.index,
        summary=summary,
    )


def _try_load_local_csv(ticker1: str, ticker2: str) -> Optional[pd.DataFrame]:
    """Check if pre-saved CSV exists in DataSet/ to avoid network latency and rate limits."""
    candidates = [
        os.path.join("DataSet", f"{ticker1}_{ticker2}.csv"),
        os.path.join("DataSet", f"{ticker2}_{ticker1}.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                col1 = f"{ticker1} Close"
                col2 = f"{ticker2} Close"
                if col1 in df.columns and col2 in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
                    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
                    res = pd.DataFrame({
                        ticker1: df[col1].astype(float),
                        ticker2: df[col2].astype(float),
                    })
                    return res
            except Exception:
                pass
    return None


def fetch_pair_data(
    ticker1: str,
    ticker2: str,
    formation_start: str = DEFAULT_FORMATION_START,
    formation_end: str = DEFAULT_FORMATION_END,
    trading_start: str = DEFAULT_TRADING_START,
    trading_end: str = DEFAULT_TRADING_END,
) -> PairData:
    """
    Fetch, clean, synchronize, and partition daily adjusted close prices for a single pair.
    """
    overall_start = min(formation_start, trading_start)
    overall_end = max(formation_end, trading_end)
    
    local_df = _try_load_local_csv(ticker1, ticker2)
    if local_df is not None and not local_df.empty:
        raw_df = local_df.loc[overall_start:overall_end]
    else:
        raw_df = _download_adj_close_batch([ticker1, ticker2], overall_start, overall_end)

    return _process_pair_from_raw(
        raw_df=raw_df,
        ticker1=ticker1,
        ticker2=ticker2,
        formation_start=formation_start,
        formation_end=formation_end,
        trading_start=trading_start,
        trading_end=trading_end,
    )


def fetch_all_target_pairs(
    pairs: Optional[List[Tuple[str, str]]] = None,
    formation_start: str = DEFAULT_FORMATION_START,
    formation_end: str = DEFAULT_FORMATION_END,
    trading_start: str = DEFAULT_TRADING_START,
    trading_end: str = DEFAULT_TRADING_END,
) -> Dict[Tuple[str, str], PairData]:
    """
    Download and preprocess all target pairs, using local DataSet/ cache where available.
    """
    if pairs is None:
        pairs = TARGET_PAIRS
        
    overall_start = min(formation_start, trading_start)
    overall_end = max(formation_end, trading_end)
    
    pair_data_dict = {}
    remaining_pairs = []

    for t1, t2 in pairs:
        local_df = _try_load_local_csv(t1, t2)
        if local_df is not None and not local_df.empty:
            raw_pair = local_df.loc[overall_start:overall_end]
            pair_data_dict[(t1, t2)] = _process_pair_from_raw(
                raw_df=raw_pair,
                ticker1=t1,
                ticker2=t2,
                formation_start=formation_start,
                formation_end=formation_end,
                trading_start=trading_start,
                trading_end=trading_end,
            )
        else:
            remaining_pairs.append((t1, t2))
            
    if remaining_pairs:
        rem_tickers = sorted(list({t for p in remaining_pairs for t in p}))
        raw_batch = _download_adj_close_batch(rem_tickers, overall_start, overall_end)
        for t1, t2 in remaining_pairs:
            pair_data_dict[(t1, t2)] = _process_pair_from_raw(
                raw_df=raw_batch,
                ticker1=t1,
                ticker2=t2,
                formation_start=formation_start,
                formation_end=formation_end,
                trading_start=trading_start,
                trading_end=trading_end,
            )
            
    return pair_data_dict


def print_data_summary(pair_data: PairData) -> None:
    """Print nicely formatted summary statistics for a PairData instance."""
    s = pair_data.summary
    t1, t2 = pair_data.ticker1, pair_data.ticker2
    
    print("=" * 72)
    print(f" PAIRS TRADING DATA SUMMARY: {t1} (X1) vs {t2} (X2)")
    print("=" * 72)
    print(f"Total Date Range          : {s['total_start_date']} to {s['total_end_date']}")
    print(f"Total Aligned Days        : {s['total_trading_days']} trading sessions")
    print(f"Raw Missing Values ({t1}) : {s[f'missing_before_{t1}']} entries")
    print(f"Raw Missing Values ({t2}) : {s[f'missing_before_{t2}']} entries")
    print(f"Missing Post-Clean (NaN)  : {s['missing_after_cleaning']} entries (100% synchronous)")
    print("-" * 72)
    print("In-Sample (Formation / Estimation Period):")
    print(f"  Range                   : {s['formation_start_date']} to {s['formation_end_date']}")
    print(f"  Trading Days            : {s['formation_trading_days']} days")
    print(f"  Daily Return Correlation: {s['formation_return_corr']:.4f}")
    print("-" * 72)
    print("Out-of-Sample (Trading / Backtest Period):")
    print(f"  Range                   : {s['trading_start_date']} to {s['trading_end_date']}")
    print(f"  Trading Days            : {s['trading_period_days']} days")
    print(f"  Daily Return Correlation: {s['trading_return_corr']:.4f}")
    print("=" * 72)


def plot_pair_sections(
    pair_data: PairData,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Plot normalized price series (base 100), log ratio, rolling correlation,
    and formation/trading split in a unified multi-section figure.
    """
    t1, t2 = pair_data.ticker1, pair_data.ticker2
    df = pair_data.df_all
    
    # Normalized prices
    norm1 = (df[t1] / df[t1].iloc[0]) * 100.0
    norm2 = (df[t2] / df[t2].iloc[0]) * 100.0
    log_ratio = np.log(df[t2]) - np.log(df[t1])
    
    # Rolling correlation (60-day)
    ret1 = df[t1].pct_change()
    ret2 = df[t2].pct_change()
    roll_corr = ret1.rolling(60).corr(ret2)
    
    split_date = pd.to_datetime(pair_data.summary["trading_start_date"])
    
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.2, 1.0]})
    
    # Section 1: Normalized prices
    axes[0].plot(df.index, norm1, label=f"{t1} (X1, Base=100)", color="#1f77b4", linewidth=1.6)
    axes[0].plot(df.index, norm2, label=f"{t2} (X2, Base=100)", color="#ff7f0e", linewidth=1.6)
    axes[0].axvline(split_date, color="#dc2626", linestyle="--", linewidth=1.5,
                    label=f"Trading Start ({split_date.strftime('%Y-%m-%d')})")
    axes[0].axvspan(df.index[0], split_date, color="#e0f2fe", alpha=0.5, label="Formation Period (In-Sample)")
    axes[0].axvspan(split_date, df.index[-1], color="#fef3c7", alpha=0.35, label="Trading Period (Out-of-Sample)")
    axes[0].set_title(f"[Section 1] Normalized Price Series: {t1} vs {t2}", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Normalized Price (100)", fontsize=10)
    axes[0].legend(loc="upper left", framealpha=0.9, fontsize=9)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    
    # Section 2: Log Spread / Ratio
    mean_form = np.log(pair_data.df_formation[t2]) - np.log(pair_data.df_formation[t1])
    axes[1].plot(df.index, log_ratio, label=f"Log Price Ratio: ln({t2}) - ln({t1})", color="#2ca02c", linewidth=1.3)
    axes[1].axhline(mean_form.mean(), color="black", linestyle=":", label=f"Formation Mean ({mean_form.mean():.3f})")
    axes[1].axvline(split_date, color="#dc2626", linestyle="--", linewidth=1.5)
    axes[1].axvspan(df.index[0], split_date, color="#e0f2fe", alpha=0.5)
    axes[1].axvspan(split_date, df.index[-1], color="#fef3c7", alpha=0.35)
    axes[1].set_title("[Section 2] Relative Log Price Ratio & Spread Dynamics", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("ln(X2) - ln(X1)", fontsize=10)
    axes[1].legend(loc="upper left", framealpha=0.9, fontsize=9)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    
    # Section 3: Rolling 60-Day Return Correlation
    axes[2].plot(df.index, roll_corr, label="60-Day Rolling Return Correlation", color="#7c3aed", linewidth=1.2)
    axes[2].axhline(0, color="gray", linestyle=":", linewidth=0.8)
    axes[2].axvline(split_date, color="#dc2626", linestyle="--", linewidth=1.5)
    axes[2].axvspan(df.index[0], split_date, color="#e0f2fe", alpha=0.5)
    axes[2].axvspan(split_date, df.index[-1], color="#fef3c7", alpha=0.35)
    axes[2].set_title("[Section 3] 60-Day Rolling Return Correlation", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Correlation", fontsize=10)
    axes[2].set_xlabel("Date", fontsize=10)
    axes[2].legend(loc="lower left", framealpha=0.9, fontsize=9)
    axes[2].grid(True, linestyle=":", alpha=0.6)
    
    axes[2].xaxis.set_major_locator(mdates.YearLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, dpi=140)
        print(f"Saved pair sections plot to: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Fetch and verify real market price data for pairs trading.")
    parser.add_argument("--ticker1", type=str, default="KO", help="First ticker (X1)")
    parser.add_argument("--ticker2", type=str, default="PEP", help="Second ticker (X2)")
    parser.add_argument("--all-pairs", action="store_true", help="Download and process all target ticker pairs")
    parser.add_argument("--formation-start", type=str, default=DEFAULT_FORMATION_START, help="Formation start (YYYY-MM-DD)")
    parser.add_argument("--formation-end", type=str, default=DEFAULT_FORMATION_END, help="Formation end (YYYY-MM-DD)")
    parser.add_argument("--trading-start", type=str, default=DEFAULT_TRADING_START, help="Trading start (YYYY-MM-DD)")
    parser.add_argument("--trading-end", type=str, default=DEFAULT_TRADING_END, help="Trading end (YYYY-MM-DD)")
    parser.add_argument("--output-plot", type=str, default=None, help="Path to save individual pair plot")
    args = parser.parse_args()

    if args.all_pairs:
        print(f"\nFetching all {len(TARGET_PAIRS)} target pairs in batch...")
        all_pairs = fetch_all_target_pairs(
            formation_start=args.formation_start,
            formation_end=args.formation_end,
            trading_start=args.trading_start,
            trading_end=args.trading_end,
        )
        for (t1, t2), pdata in all_pairs.items():
            print_data_summary(pdata)
            save_name = f"pair_{t1}_{t2}.png"
            plot_pair_sections(pdata, save_path=save_name)
    else:
        print(f"\nFetching pair: ({args.ticker1}, {args.ticker2})...")
        pair_data = fetch_pair_data(
            ticker1=args.ticker1,
            ticker2=args.ticker2,
            formation_start=args.formation_start,
            formation_end=args.formation_end,
            trading_start=args.trading_start,
            trading_end=args.trading_end,
        )
        print_data_summary(pair_data)
        out_plot = args.output_plot or f"pair_{args.ticker1}_{args.ticker2}.png"
        plot_pair_sections(pair_data, save_path=out_plot)


if __name__ == "__main__":
    main()
