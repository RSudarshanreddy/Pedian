"""Daily prices for every NSE stock: NSE's own end-of-day files for the live list,
Yahoo for long backtest history (cached so the backtest can re-run offline)."""

from __future__ import annotations

import datetime as dt
import io
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from curl_cffi import requests

UNIVERSE_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
BHAVCOPY_URL = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{:%Y%m%d}_F_0000.csv.zip"
SERIES = ("EQ", "BE", "BZ")          # the main board; SME and debt series are left out
COMMON_RATIOS = (1 / 100, 1 / 50, 1 / 25, 1 / 20, 1 / 10, 1 / 6, 1 / 5, 1 / 4, 1 / 3, 2 / 5, 1 / 2, 2 / 3,
                 2, 5, 10)           # splits, bonuses (1:1 -> 1/2, 2:1 -> 1/3, 3:2 -> 2/5) and consolidations
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


def bhavcopy(day: dt.date, session: requests.Session) -> pd.DataFrame | None:
    """NSE's end-of-day file for one session, or None when there is none (a holiday, or not
    published yet -- NSE puts it out in the evening)."""
    r = session.get(BHAVCOPY_URL.format(day), timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))
    df = df[df.SctySrs.isin(SERIES)].drop_duplicates("TckrSymb")
    return pd.DataFrame({"symbol": df.TckrSymb, "Open": df.OpnPric, "High": df.HghPric, "Low": df.LwPric,
                         "Close": df.ClsPric, "Volume": df.TtlTradgVol,
                         "date": pd.Timestamp(day)})


def split_ratio(gap: float) -> float:
    """A split or bonus from the opening gap: NSE's files carry raw prices (the previous close is
    NOT adjusted), but price bands stop any stock opening 30% away from its last close without a
    corporate action. The gap is snapped to the nearest common ratio (10:1 split -> 0.1, 1:1 bonus
    -> 0.5, 3:2 bonus -> 0.4, ...), or used as it is when none is close."""
    if 0.7 <= gap <= 1.45:
        return 1.0
    nearest = min(COMMON_RATIOS, key=lambda r: abs(np.log(gap / r)))
    return nearest if abs(np.log(gap / nearest)) < 0.12 else gap


def nse_bars(symbols: list[str], sessions: int, until: dt.date) -> dict[str, pd.DataFrame]:
    """The last `sessions` sessions up to `until`, straight from NSE's end-of-day files,
    adjusted for splits and bonuses (see split_ratio) the way Yahoo adjusts them."""
    session, days, day = requests.Session(impersonate="chrome"), [], until
    while len(days) < sessions and day > until - dt.timedelta(days=sessions * 2 + 20):
        if day.weekday() < 5 and (b := bhavcopy(day, session)) is not None:
            days.append(b)
        day -= dt.timedelta(days=1)
    if not days:
        return {}
    wanted = set(symbols)
    panel = pd.concat(days)
    bars = {}
    for sym, df in panel[panel.symbol.isin(wanted)].groupby("symbol"):
        df = df.set_index("date").sort_index()
        df = df[(df.Volume > 0) & (df.Low > 0)]
        ratio = (df.Open / df.Close.shift()).fillna(1.0).map(split_ratio)
        scale = ratio[::-1].cumprod()[::-1].shift(-1, fill_value=1.0)   # product of every later ratio
        out = df[["Open", "High", "Low", "Close"]].mul(scale, axis=0)
        out["Volume"] = df.Volume / scale
        out.index.name = "Date"
        bars[sym] = out
    print(f"  NSE files: {len(days)} sessions, {panel.date.min():%d %b} to {panel.date.max():%d %b}, {len(bars)} stocks")
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
