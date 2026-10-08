"""Daily prices for every NSE stock, cached so the backtest can re-run offline."""

from __future__ import annotations

import io
import pickle
from pathlib import Path

import pandas as pd
import yfinance as yf
from curl_cffi import requests

UNIVERSE_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
DATA_DIR = Path(__file__).parent / "data"
CACHE = DATA_DIR / "bars.pkl"


def load_universe() -> list[str]:
    """Every symbol NSE lists today, all series."""
    r = requests.get(UNIVERSE_URL, impersonate="chrome", timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    return sorted(df["SYMBOL"].astype(str).str.strip().unique())


def download(symbols: list[str], period: str = "4y", chunk: int = 100,
             min_rows: int = 120) -> dict[str, pd.DataFrame]:
    """Split- and dividend-adjusted daily OHLCV, keyed by NSE symbol."""
    bars: dict[str, pd.DataFrame] = {}
    for n in range(0, len(symbols), chunk):
        batch = symbols[n:n + chunk]
        raw = yf.download(
            [f"{s}.NS" for s in batch], period=period, interval="1d",
            auto_adjust=True, group_by="ticker", threads=True, progress=False,
        )
        for s in batch:
            try:
                df = raw[f"{s}.NS"].dropna(subset=["Open", "High", "Low", "Close"])
            except KeyError:
                continue
            df = df[(df["Volume"] > 0) & (df["Low"] > 0)]
            if len(df) >= min_rows:
                bars[s] = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        print(f"  {min(n + chunk, len(symbols))}/{len(symbols)} symbols, {len(bars)} usable")
    return bars


def load_bars(refresh: bool = False) -> dict[str, pd.DataFrame]:
    if CACHE.exists() and not refresh:
        with CACHE.open("rb") as f:
            return pickle.load(f)
    print("Downloading the NSE universe...")
    bars = download(load_universe())
    DATA_DIR.mkdir(exist_ok=True)
    with CACHE.open("wb") as f:
        pickle.dump(bars, f)
    return bars
