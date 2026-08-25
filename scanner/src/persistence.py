from __future__ import annotations

import argparse
import math
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from google.api_core.exceptions import NotFound
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


@dataclass
class ScannerConfig:
    min_days: int = 200
    persistence_lookback: int = 180
    persistence_window: int = 3
    persistence_move_needed: float = 3.0

    volatility_lookback: int = 180
    min_avg_volatility: float = 3.5
    min_median_volatility: float = 2.5
    min_volatile_days: int = 30
    min_volatility_ratio: float = 0.20

    persistence_min_rate: float = 45.0
    persistence_min_sample: int = 50
    persistence_stability_threshold: float = 10.0

    min_price: float = 250.0
    max_price: float = 3500.0
    min_avg_traded_value_cr: float = 5.0

    support_window: int = 20
    volatility_threshold: float = 3.0
    atr_period: int = 14

    breakout_volume_mult: float = 1.5
    pullback_min: float = 2.0
    pullback_max: float = 8.0
    max_today_range: float = 12.0

    support_distance_threshold: float = 5.0

    max_risk_pct: float = 8.0
    min_rr: float = 1.5
    stop_atr_mult: float = 1.5
    recent_low_buffer: float = 0.975
    target_risk_mult: float = 2.0

    top_n: int = 40
    history_period: str = "2y"
    interval: str = "1d"

    # --- download batching ---
    batch_size: int = 100
    max_download_retries: int = 2

    # --- scoring weights (previously hardcoded magic numbers) ---
    score_persistence_max: float = 30.0
    score_persistence_divisor: float = 100.0
    score_expected_move_max: float = 10.0
    score_expected_move_divisor: float = 6.0
    score_stability_bonus_high: float = 10.0
    score_stability_bonus_mid: float = 5.0
    score_stability_mid_mult: float = 1.5  # threshold * this = mid-tier cutoff
    score_avg_vol_max: float = 15.0
    score_avg_vol_divisor: float = 8.0
    score_median_max: float = 10.0
    score_median_divisor: float = 6.0
    score_frequency_max: float = 5.0
    score_frequency_mult: float = 5.0
    score_today_bonus_max: float = 8.0
    score_today_bonus_soft_cap: float = 8.0  # range% below which bonus scales up
    score_today_bonus_decay_mult: float = 1.5
    score_volume_bonus_max: float = 5.0
    score_volume_bonus_divisor: float = 3.0
    score_setup_bonus_breakout: float = 7.0
    score_setup_bonus_pullback: float = 6.0
    score_setup_bonus_reclaim: float = 5.0
    score_setup_bonus_default: float = 3.0
    score_support_penalty_mult: float = 0.3
    score_support_penalty_max: float = 5.0


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()

    df = data.copy()

    if isinstance(df.columns, pd.MultiIndex):
        if len(df.columns.get_level_values(-1).unique()) == 1:
            df.columns = df.columns.get_level_values(0)
        elif len(df.columns.get_level_values(0).unique()) == 1:
            df.columns = df.columns.get_level_values(1)
        else:
            raise ValueError("Expected single-ticker OHLCV data")

    df.columns = [str(c).strip().title() for c in df.columns]

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # FIX: previously only Open/High/Low/Close were checked for NaN; a NaN
    # Volume day would silently corrupt avg_volume20 / traded value / volume
    # spike calculations downstream.
    return df.dropna(subset=required).sort_index()


def has_enough_data(data: pd.DataFrame, config: ScannerConfig) -> bool:
    return data is not None and len(data) >= config.min_days


def add_indicators(data: pd.DataFrame, config: ScannerConfig) -> pd.DataFrame:
    df = data.copy()

    # Volatility
    df["range_pct"] = (df["High"] - df["Low"]) / df["Close"] * 100
    df["close_move_pct"] = df["Close"].pct_change().abs() * 100
    df["is_volatile_day"] = df["range_pct"] >= config.volatility_threshold
    df["volatility_measure"] = df["range_pct"]

    # Trend
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["MA20"] = df["Close"].rolling(config.support_window).mean()

    # Structure
    df["recent_low"] = df["Low"].rolling(config.support_window).min()
    df["high_20_prev"] = df["High"].shift(1).rolling(config.support_window).max()
    df["high_20"] = df["High"].rolling(config.support_window).max()
    df["pullback_pct"] = (df["high_20"] - df["Close"]) / df["high_20"] * 100

    # Volume
    df["avg_volume20"] = df["Volume"].rolling(20).mean()
    df["avg_traded_value20_cr"] = (
        df["Close"].rolling(20).mean() * df["Volume"].rolling(20).mean()
    ) / 10_000_000

    # ATR(14) -- simple rolling mean of true range (not Wilder's smoothing)
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR"] = tr.rolling(config.atr_period).mean()

    # --- Vectorized precomputation for verify_volatility_persistence ---
    # "past" volatility mean as of day i: mean of volatility_measure over the
    # volatility_lookback days STRICTLY BEFORE day i (hence the shift(1)).
    df["past_vol_mean"] = (
        df["volatility_measure"]
        .rolling(config.volatility_lookback, min_periods=20)
        .mean()
        .shift(1)
    )
    # forward max range over the next `persistence_window` days, INCLUDING day i.
    df["fwd_max_range"] = (
        df["volatility_measure"]
        .rolling(config.persistence_window)
        .max()
        .shift(-(config.persistence_window - 1))
    )

    return df


def verify_volatility_identity(
    data: pd.DataFrame, config: ScannerConfig
) -> tuple[float, float, int, float]:
    hist = data.tail(config.volatility_lookback)
    if len(hist) < config.volatility_lookback:
        return 0.0, 0.0, 0, 0.0

    avg = float(hist["volatility_measure"].mean())
    median = float(hist["volatility_measure"].median())
    volatile_days = int(hist["is_volatile_day"].sum())
    ratio = volatile_days / len(hist)
    return avg, median, volatile_days, ratio


def verify_volatility_persistence(
    data: pd.DataFrame, config: ScannerConfig
) -> tuple[float, int, float, float]:
    """
    Checks whether days that were historically volatile (mean daily range
    over the prior `volatility_lookback` days >= min_avg_volatility) were
    followed by a further move of at least `persistence_move_needed`%
    within the next `persistence_window` days.

    Computed over the most recent `persistence_lookback` days, split into
    4 roughly-equal sub-periods so we can also report how STABLE the hit
    rate is over time (`persistence_stability`, the std dev of the
    per-period hit rates) rather than just the aggregate rate.

    Requires add_indicators() to have already populated `past_vol_mean`
    and `fwd_max_range` on `data`.

    Returns: (hit_rate_pct, sample_size, expected_forward_move_pct, stability_std)
    """
    lookback = config.persistence_lookback
    window = config.persistence_window
    needed = config.persistence_move_needed

    if len(data) < lookback + window:
        return 0.0, 0, 0.0, 999.0
    if "past_vol_mean" not in data.columns or "fwd_max_range" not in data.columns:
        raise ValueError("data must be passed through add_indicators() first")

    end_base = len(data) - window
    start_base = end_base - lookback

    eligible = data["past_vol_mean"] >= config.min_avg_volatility
    hit = data["fwd_max_range"] >= needed

    period_size = max(1, lookback // 4)
    hits = 0
    total = 0
    forward_moves: list[float] = []
    persistence_by_period: list[float] = []

    for period_num in range(4):
        p_start = start_base + period_num * period_size
        p_end = (
            start_base + (period_num + 1) * period_size
            if period_num < 3 else end_base
        )
        p_end = min(p_end, end_base)
        if p_start >= p_end:
            continue

        mask = eligible.iloc[p_start:p_end]
        idx = mask[mask].index
        period_total = len(idx)
        if period_total == 0:
            continue

        period_hit = hit.loc[idx]
        period_hits = int(period_hit.sum())

        forward_moves.extend(data.loc[idx, "fwd_max_range"].tolist())
        hits += period_hits
        total += period_total
        persistence_by_period.append(period_hits / period_total * 100)

    if total == 0:
        return 0.0, 0, 0.0, 999.0

    rate = round(hits / total * 100, 1)
    expected_move = round(float(np.mean(forward_moves)), 2)
    stability = (
        round(float(np.std(persistence_by_period)), 1)
        if len(persistence_by_period) > 1 else 0.0
    )

    return rate, total, expected_move, stability


def get_entry_trigger(data: pd.DataFrame, config: ScannerConfig) -> tuple[str, str, str]:
    if len(data) < 2:
        return "WATCH", "insufficient", "Not enough data"

    latest = data.iloc[-1]
    prev = data.iloc[-2]

    close = float(latest["Close"])
    ema20 = float(latest["EMA20"])
    pullback = float(latest["pullback_pct"])
    high_20_prev = float(latest["high_20_prev"])
    today_range = float(latest["volatility_measure"])

    avg_volume = float(latest["avg_volume20"])
    volume_spike = float(latest["Volume"]) / avg_volume if avg_volume > 0 else 0

    bullish = close > float(latest["Open"])
    closed_above_prev_high = close > float(prev["High"])

    if pd.isna(ema20) or pd.isna(high_20_prev) or pd.isna(pullback):
        return "WATCH", "no_data", "Indicators not ready"

    if today_range > config.max_today_range:
        return "WATCH", "exhaustion", f"Today's range {today_range:.1f}% > {config.max_today_range:.1f}%"

    if close > high_20_prev and volume_spike >= config.breakout_volume_mult and bullish:
        return "BUY", "breakout", f"Broke 20D high on {volume_spike:.1f}x volume"

    if (
        config.pullback_min <= pullback <= config.pullback_max
        and close > ema20
        and bullish
        and closed_above_prev_high
    ):
        return "BUY", "pullback_bounce", f"Bounce from {pullback:.1f}% pullback, confirmed"

    prev_close = float(prev["Close"])
    prev_ema20 = float(prev["EMA20"])
    if close > ema20 and prev_close <= prev_ema20 and bullish:
        return "BUY", "reclaim", "Reclaimed EMA20 with bullish close"

    if pullback > 12:
        return "WATCH", "deep_pullback", f"{pullback:.1f}% pullback - wait for base"
    if pullback < config.pullback_min:
        return "WATCH", "extended", f"{pullback:.1f}% pullback - wait for pullback"
    if close < ema20:
        return "WATCH", "below_trend", "Below EMA20 - wait for reclaim"

    return "WATCH", "cooling", "Volatile but no clear trigger"


def calculate_risk_reward(data: pd.DataFrame, config: ScannerConfig) -> Optional[dict[str, float]]:
    latest = data.iloc[-1]
    entry = float(latest["Close"])
    atr = float(latest["ATR"])
    recent_low = float(latest["recent_low"])

    if not all(math.isfinite(x) for x in [entry, atr, recent_low]) or atr <= 0:
        return None

    stop_loss = max(
        entry - config.stop_atr_mult * atr,
        recent_low * config.recent_low_buffer,
    )
    risk = entry - stop_loss

    if risk <= 0:
        return None

    target = entry + config.target_risk_mult * risk
    rr = (target - entry) / risk
    risk_pct = risk / entry * 100

    return {
        "entry": round(entry, 2),
        "stop_loss": round(stop_loss, 2),
        "target": round(target, 2),
        "risk_per_share": round(risk, 2),
        "risk_pct": round(risk_pct, 2),
        "rr_ratio": round(rr, 2),
        "atr": round(atr, 2),
    }


def calculate_score(
    avg_volatility: float,
    median_volatility: float,
    volatility_ratio: float,
    persistence_rate: float,
    persistence_stability: float,
    expected_move: float,
    today_range: float,
    volume_spike: float,
    action: str,
    setup_type: str,
    distance_from_support: float,
    config: ScannerConfig,
) -> float:
    """
    Composite 0-100 score. All weights/divisors/thresholds live on
    ScannerConfig (score_*) so the scoring model can be tuned without
    touching code.
    """
    persistence_score = min(
        persistence_rate / config.score_persistence_divisor * config.score_persistence_max,
        config.score_persistence_max,
    )
    expected_move_score = min(
        expected_move / config.score_expected_move_divisor * config.score_expected_move_max,
        config.score_expected_move_max,
    )

    if persistence_stability < config.persistence_stability_threshold:
        stability_bonus = config.score_stability_bonus_high
    elif persistence_stability < config.persistence_stability_threshold * config.score_stability_mid_mult:
        stability_bonus = config.score_stability_bonus_mid
    else:
        stability_bonus = 0.0

    avg_vol_score = min(
        avg_volatility / config.score_avg_vol_divisor * config.score_avg_vol_max,
        config.score_avg_vol_max,
    )
    median_score = min(
        median_volatility / config.score_median_divisor * config.score_median_max,
        config.score_median_max,
    )
    frequency_score = min(volatility_ratio * config.score_frequency_mult, config.score_frequency_max)

    if today_range <= config.score_today_bonus_soft_cap:
        today_bonus = min(
            today_range / (config.score_today_bonus_soft_cap * 0.75) * config.score_today_bonus_max,
            config.score_today_bonus_max,
        )
    else:
        today_bonus = max(
            config.score_today_bonus_max
            - (today_range - config.score_today_bonus_soft_cap) * config.score_today_bonus_decay_mult,
            0.0,
        )

    volume_bonus = min(
        volume_spike / config.score_volume_bonus_divisor * config.score_volume_bonus_max,
        config.score_volume_bonus_max,
    )

    if action == "BUY":
        setup_bonus = {
            "breakout": config.score_setup_bonus_breakout,
            "pullback_bounce": config.score_setup_bonus_pullback,
            "reclaim": config.score_setup_bonus_reclaim,
        }.get(setup_type, config.score_setup_bonus_default)
    else:
        setup_bonus = config.score_setup_bonus_default

    support_penalty = min(
        max(distance_from_support - config.support_distance_threshold, 0) * config.score_support_penalty_mult,
        config.score_support_penalty_max,
    )

    score = (
        persistence_score
        + expected_move_score
        + stability_bonus
        + avg_vol_score
        + median_score
        + frequency_score
        + today_bonus
        + volume_bonus
        + setup_bonus
        - support_penalty
    )

    return round(max(min(score, 100), 0), 2)


def build_candidate(
    ticker: str,
    data: pd.DataFrame,
    config: ScannerConfig,
    run_date: str,
) -> Optional[dict[str, Any]]:
    if len(data) < config.min_days:
        return None

    latest = data.iloc[-1]

    required = [
        latest.get("MA20"),
        latest.get("recent_low"),
        latest.get("high_20"),
        latest.get("avg_volume20"),
        latest.get("avg_traded_value20_cr"),
        latest.get("ATR"),
    ]
    if any(pd.isna(x) for x in required):
        return None

    price = float(latest["Close"])
    traded_value = float(latest["avg_traded_value20_cr"])

    if not (config.min_price < price < config.max_price):
        return None
    if traded_value < config.min_avg_traded_value_cr:
        return None

    avg_v, median_v, volatile_days, vol_ratio = verify_volatility_identity(data, config)

    if avg_v < config.min_avg_volatility:
        return None
    if median_v < config.min_median_volatility:
        return None
    if volatile_days < config.min_volatile_days:
        return None
    if vol_ratio < config.min_volatility_ratio:
        return None

    persistence_rate, sample_size, expected_move, stability = verify_volatility_persistence(data, config)

    if persistence_rate < config.persistence_min_rate:
        return None
    if sample_size < config.persistence_min_sample:
        return None

    action, setup_type, reason = get_entry_trigger(data, config)

    risk = calculate_risk_reward(data, config)
    if risk is None:
        return None
    if risk["risk_pct"] > config.max_risk_pct:
        return None
    if risk["rr_ratio"] < config.min_rr:
        return None

    avg_volume = float(latest["avg_volume20"])
    volume_spike = float(latest["Volume"]) / avg_volume if avg_volume > 0 else 0
    today_range = float(latest["volatility_measure"])
    distance_from_support = (price - float(latest["recent_low"])) / price * 100

    score = calculate_score(
        avg_volatility=avg_v,
        median_volatility=median_v,
        volatility_ratio=vol_ratio,
        persistence_rate=persistence_rate,
        persistence_stability=stability,
        expected_move=expected_move,
        today_range=today_range,
        volume_spike=volume_spike,
        action=action,
        setup_type=setup_type,
        distance_from_support=distance_from_support,
        config=config,
    )

    return {
        "Ticker": ticker,
        "Bar_Date": pd.Timestamp(data.index[-1]).date().isoformat(),
        "Run_Date": run_date,
        "Action": action,
        "Setup_Type": setup_type,
        "Score": score,
        "Reason": reason,
        "Price": round(price, 2),
        "Avg_Volatility": round(avg_v, 2),
        "Median_Volatility": round(median_v, 2),
        "Volatile_Days": volatile_days,
        "Volatility_Ratio": round(vol_ratio, 4),
        "Persistence_Rate": round(persistence_rate, 2),
        "Persistence_Sample": sample_size,
        "Expected_Move": round(expected_move, 2),
        "Persistence_Stability": round(stability, 2),
        "Today_Range": round(today_range, 2),
        "Volume_Spike": round(volume_spike, 2),
        "Entry": risk["entry"],
        "Stop_Loss": risk["stop_loss"],
        "Target": risk["target"],
        "Risk_Per_Share": risk["risk_per_share"],
        "Risk_Pct": risk["risk_pct"],
        "RR_Ratio": risk["rr_ratio"],
        "Max_Hold_Days": config.persistence_window,
        "ATR": risk["atr"],
    }


DEFAULT_NSE_CSV = "EQUITY_L.csv"
NSE_EQUITY_CSV_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
NSE_HOME_URL = "https://www.nseindia.com"
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/vnd.ms-excel,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def _fetch_nse_csv_with_session() -> pd.DataFrame:
    """
    archives.nseindia.com frequently rejects bare requests (401/403) unless
    a cookie/session has been established against the main site first.
    """
    session = requests.Session()
    session.headers.update(NSE_HEADERS)

    # Warm-up request to establish cookies before hitting the archive endpoint.
    session.get(NSE_HOME_URL, timeout=20)

    response = session.get(NSE_EQUITY_CSV_URL, timeout=20)
    response.raise_for_status()
    return pd.read_csv(BytesIO(response.content))


def load_nse_tickers(path: Optional[str] = None) -> list[str]:
    requested_path = path or DEFAULT_NSE_CSV
    csv_path = Path(requested_path)

    if path is None:
        local_paths = [csv_path, Path(__file__).resolve().parent / DEFAULT_NSE_CSV]
        existing_path = next((candidate for candidate in local_paths if candidate.exists()), None)
    else:
        existing_path = csv_path if csv_path.exists() else None

    if existing_path is not None:
        df = pd.read_csv(existing_path)
    elif path is not None:
        raise FileNotFoundError(f"{requested_path} not found")
    else:
        try:
            df = _fetch_nse_csv_with_session()
        except Exception as exc:
            module_path = Path(__file__).resolve().parent / DEFAULT_NSE_CSV
            raise FileNotFoundError(
                f"{DEFAULT_NSE_CSV} not found and NSE download failed: {exc}. "
                f"Place the file at {module_path} or provide tickers explicitly."
            ) from exc

    symbol_col = next((c for c in ["SYMBOL", "Symbol", "symbol"] if c in df.columns), None)
    if symbol_col is None:
        raise ValueError("EQUITY_L.csv does not contain SYMBOL")

    return sorted(set(
        df[symbol_col].astype(str).str.strip()
        .replace("", np.nan).dropna()
        .map(lambda x: x if x.endswith(".NS") else f"{x}.NS")
    ))


def download_batch(tickers: list[str], config: ScannerConfig) -> dict[str, pd.DataFrame]:
    """Download a batch of tickers in a single yfinance call, split per ticker."""
    raw = yf.download(
        tickers,
        period=config.history_period,
        interval=config.interval,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    result: dict[str, pd.DataFrame] = {}

    if len(tickers) == 1:
        # yfinance returns a flat (non-MultiIndex) frame for a single ticker.
        try:
            result[tickers[0]] = normalize_columns(raw)
        except Exception:
            LOGGER.exception("Failed normalizing %s", tickers[0])
        return result

    top_level = set(raw.columns.get_level_values(0)) if isinstance(raw.columns, pd.MultiIndex) else set()
    for ticker in tickers:
        try:
            if ticker not in top_level:
                continue
            sub = raw[ticker]
            if sub is None or sub.empty:
                continue
            result[ticker] = normalize_columns(sub)
        except Exception:
            LOGGER.exception("Failed normalizing %s", ticker)

    return result


def download_ticker(ticker: str, config: ScannerConfig) -> pd.DataFrame:
    """Single-ticker download path, kept for diagnose_ticker() / ad-hoc use."""
    data = yf.download(
        ticker,
        period=config.history_period,
        interval=config.interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return normalize_columns(data)


def scan_tickers(
    tickers: list[str],
    config: ScannerConfig,
    run_date: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Downloads and scores `tickers` in batches. Returns (results, failed_tickers)
    -- failed_tickers is every symbol that never returned usable data after
    retries, so a systemic failure (e.g. an API format change) is visible
    instead of silently looking like "0 candidates found".
    """
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    batch_size = config.batch_size
    n_batches = math.ceil(len(tickers) / batch_size) if tickers else 0

    for b in range(n_batches):
        batch = tickers[b * batch_size: (b + 1) * batch_size]
        # LOGGER.info("Batch %d/%d (%d tickers)", b + 1, n_batches, len(batch))

        frames: dict[str, pd.DataFrame] = {}
        remaining = batch
        for attempt in range(config.max_download_retries + 1):
            if not remaining:
                break
            try:
                fetched = download_batch(remaining, config)
            except Exception:
                LOGGER.exception("Batch download failed (attempt %d/%d)", attempt + 1, config.max_download_retries + 1)
                fetched = {}
            frames.update(fetched)
            remaining = [t for t in remaining if t not in frames]

        failures.extend(remaining)

        for ticker, data in frames.items():
            try:
                if not has_enough_data(data, config):
                    continue
                candidate = build_candidate(ticker, add_indicators(data, config), config, run_date)
                if candidate:
                    rows.append(candidate)
            except Exception:
                LOGGER.exception("Failed processing %s", ticker)
                failures.append(ticker)

    if failures:
        LOGGER.warning("%d/%d tickers failed or returned no data", len(failures), len(tickers))

    if not rows:
        return pd.DataFrame(), failures

    df = pd.DataFrame(rows).drop_duplicates(["Ticker", "Bar_Date"])
    df["_action_rank"] = df["Action"].map({"BUY": 0, "WATCH": 1}).fillna(2)
    results = (
        df.sort_values(
            ["_action_rank", "Persistence_Rate", "Persistence_Stability", "Score"],
            ascending=[True, False, True, False],
        )
        .drop(columns="_action_rank")
        .head(config.top_n)
        .reset_index(drop=True)
    )
    return results, failures


BQ_SCHEMA = [
    bigquery.SchemaField("Ticker", "STRING"),
    bigquery.SchemaField("Bar_Date", "DATE"),
    bigquery.SchemaField("Run_Date", "DATE"),
    bigquery.SchemaField("Action", "STRING"),
    bigquery.SchemaField("Setup_Type", "STRING"),
    bigquery.SchemaField("Score", "FLOAT64"),
    bigquery.SchemaField("Reason", "STRING"),
    bigquery.SchemaField("Price", "FLOAT64"),
    bigquery.SchemaField("Avg_Volatility", "FLOAT64"),
    bigquery.SchemaField("Median_Volatility", "FLOAT64"),
    bigquery.SchemaField("Volatile_Days", "INT64"),
    bigquery.SchemaField("Volatility_Ratio", "FLOAT64"),
    bigquery.SchemaField("Persistence_Rate", "FLOAT64"),
    bigquery.SchemaField("Persistence_Sample", "INT64"),
    bigquery.SchemaField("Expected_Move", "FLOAT64"),
    bigquery.SchemaField("Persistence_Stability", "FLOAT64"),
    bigquery.SchemaField("Today_Range", "FLOAT64"),
    bigquery.SchemaField("Volume_Spike", "FLOAT64"),
    bigquery.SchemaField("Entry", "FLOAT64"),
    bigquery.SchemaField("Stop_Loss", "FLOAT64"),
    bigquery.SchemaField("Target", "FLOAT64"),
    bigquery.SchemaField("Risk_Per_Share", "FLOAT64"),
    bigquery.SchemaField("Risk_Pct", "FLOAT64"),
    bigquery.SchemaField("RR_Ratio", "FLOAT64"),
    bigquery.SchemaField("Max_Hold_Days", "INT64"),
    bigquery.SchemaField("ATR", "FLOAT64"),
]


def write_to_bigquery(df: pd.DataFrame, project_id: str, dataset_id: str, table_id: str) -> int:
    """
    Appends `df` to BigQuery, first deleting any existing rows that share
    (Ticker, Bar_Date) with the incoming data. This makes the write
    idempotent -- re-running the scanner for the same trading day (cron
    retry, manual re-run) no longer produces duplicate rows, since the old
    WRITE_APPEND-only path did.
    """
    if df.empty:
        return 0

    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    try:
        client.get_table(table_ref)
        table_exists = True
    except NotFound:
        client.create_table(bigquery.Table(table_ref, schema=BQ_SCHEMA))
        LOGGER.info("Created %s", table_ref)
        table_exists = False

    output = df.copy()
    output["Bar_Date"] = pd.to_datetime(output["Bar_Date"]).dt.date
    output["Run_Date"] = pd.to_datetime(output["Run_Date"]).dt.date

    if table_exists:
        pairs = output[["Ticker", "Bar_Date"]].drop_duplicates()
        conditions = " OR ".join(
            f"(Ticker = '{row.Ticker}' AND Bar_Date = DATE('{row.Bar_Date.isoformat()}'))"
            for row in pairs.itertuples(index=False)
        )
        delete_query = f"DELETE FROM `{table_ref}` WHERE {conditions}"
        client.query(delete_query).result()

    job = client.load_table_from_dataframe(
        output,
        table_ref,
        job_config=bigquery.LoadJobConfig(
            schema=BQ_SCHEMA,
            write_disposition="WRITE_APPEND",
        ),
    )
    job.result()
    return len(output)


def _resolve_project_id(project_id: Optional[str]) -> str:
    resolved = project_id or os.getenv("GCP_PROJECT")
    if not resolved:
        raise ValueError(
            "No GCP project_id provided and GCP_PROJECT env var is not set."
        )
    return resolved


def run_scanner(
    tickers: list[str],
    config: Optional[ScannerConfig] = None,
    project_id: Optional[str] = None,
    dataset_id: str = "data_options",
    table_id: str = "persistence",
) -> tuple[pd.DataFrame, list[str]]:
    """Returns (results, failed_tickers)."""
    config = config or ScannerConfig()
    resolved_project_id = _resolve_project_id(project_id)

    results, failures = scan_tickers(
        tickers,
        config,
        datetime.now(timezone.utc).date().isoformat(),
    )

    if not results.empty:
        write_to_bigquery(results, resolved_project_id, dataset_id, table_id)

    return results, failures


def diagnose_ticker(ticker: str, config: ScannerConfig) -> dict[str, Any]:
    r: dict[str, Any] = {"Ticker": ticker}

    try:
        data = download_ticker(ticker, config)
    except Exception as exc:
        r.update(status="DOWNLOAD_ERROR", reason=str(exc))
        return r

    r["rows"] = len(data)
    if not has_enough_data(data, config):
        r.update(status="INSUFFICIENT_DATA", reason=f"{len(data)} < {config.min_days}")
        return r

    data = add_indicators(data, config)
    latest = data.iloc[-1]
    price = float(latest["Close"])
    traded = latest["avg_traded_value20_cr"]

    if pd.isna(traded):
        r.update(status="INDICATORS_NOT_READY", reason="traded value is NaN")
        return r
    if not config.min_price < price < config.max_price:
        r.update(status="PRICE_FILTER", reason=f"price={price:.2f}")
        return r
    if traded < config.min_avg_traded_value_cr:
        r.update(status="LIQUIDITY_FILTER", reason=f"traded={traded:.2f}Cr")
        return r

    avg, med, days, ratio = verify_volatility_identity(data, config)
    r.update(avg_volatility=avg, median_volatility=med, volatile_days=days, volatility_ratio=ratio)

    if avg < config.min_avg_volatility:
        r.update(status="AVG_VOLATILITY_FILTER", reason=f"{avg:.2f}%")
        return r
    if med < config.min_median_volatility:
        r.update(status="MEDIAN_VOLATILITY_FILTER", reason=f"{med:.2f}%")
        return r
    if days < config.min_volatile_days:
        r.update(status="VOLATILE_DAYS_FILTER", reason=f"{days}")
        return r
    if ratio < config.min_volatility_ratio:
        r.update(status="VOLATILITY_RATIO_FILTER", reason=f"{ratio:.3f}")
        return r

    pr, sample, expected, stability = verify_volatility_persistence(data, config)
    r.update(persistence_rate=pr, persistence_sample=sample, expected_move=expected, persistence_stability=stability)

    if pr < config.persistence_min_rate:
        r.update(status="PERSISTENCE_RATE_FILTER", reason=f"{pr:.1f}%")
        return r
    if sample < config.persistence_min_sample:
        r.update(status="PERSISTENCE_SAMPLE_FILTER", reason=f"{sample}")
        return r

    action, setup, reason = get_entry_trigger(data, config)
    r.update(action=action, setup_type=setup, trigger_reason=reason)

    risk = calculate_risk_reward(data, config)
    if risk is None:
        r.update(status="RISK_CALCULATION", reason="invalid ATR/risk")
        return r

    r.update(risk)
    if risk["risk_pct"] > config.max_risk_pct:
        r.update(status="RISK_FILTER", reason=f"{risk['risk_pct']:.2f}%")
        return r
    if risk["rr_ratio"] < config.min_rr:
        r.update(status="RR_FILTER", reason=f"{risk['rr_ratio']:.2f}")
        return r

    r["status"] = "PASS"
    return r


def diagnose_tickers(tickers: list[str], config: Optional[ScannerConfig] = None) -> pd.DataFrame:
    config = config or ScannerConfig()
    return pd.DataFrame([diagnose_ticker(t, config) for t in tickers])


def main(request=None):
    if request is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--tickers", nargs="+")
        parser.add_argument("--nse-csv")
        parser.add_argument("--diagnose", action="store_true")
        parser.add_argument("--project-id")
        args = parser.parse_args()

        tickers = args.tickers or load_nse_tickers(args.nse_csv)
        if args.diagnose:
            results = diagnose_tickers(tickers)
        else:
            results, failures = run_scanner(tickers, project_id=args.project_id)
            if failures:
                LOGGER.warning("Tickers with no usable data: %s", ", ".join(failures[:20]))
        print(results.to_string(index=False))
        return

    try:
        body = request.get_json(silent=True) or {}

        # FIX: explicit None-check instead of `or`, so an intentionally empty
        # "tickers": [] doesn't silently fall back to scanning all of NSE.
        raw_tickers = body.get("tickers")
        tickers = raw_tickers if raw_tickers is not None else load_nse_tickers(body.get("nse_csv"))

        config = ScannerConfig(top_n=int(body.get("top_n", 40)))

        if body.get("diagnose", False):
            diagnostics = diagnose_tickers(tickers, config)
            return {"status": "diagnostic", "rows": diagnostics.to_dict(orient="records")}, 200

        results, failures = run_scanner(
            tickers=tickers,
            config=config,
            project_id=body.get("project_id") or os.getenv("GCP_PROJECT"),
            dataset_id=os.getenv("BQ_DATASET", "data_options"),
            table_id=os.getenv("BQ_TABLE", "persistence"),
        )

        return {
            "status": "success",
            "rows_inserted": len(results),
            "results": results.to_dict(orient="records"),
            "failed_tickers": failures,
        }, 200

    except Exception as exc:
        LOGGER.exception("Scanner failed")
        return {"status": "error", "error": str(exc)}, 500


if __name__ == "__main__":
    main()