"""
VOLATILITY PERSISTENCE SWING SCANNER
=====================================
Core Philosophy:
- A stock is "volatile" based on its full 6-month history (not a rolling 20-day window)
- The exit rule is a FIXED profit target (config.target_pct, default 3%) -- not a
  target sized as a multiple of risk. Stop-loss is sized so risk never exceeds what
  that fixed target can justify at config.min_rr (see compute_trade_levels).
- We BACKTEST that exact rule (buy at close, exit at +target_pct% or the stop,
  whichever the next `max_hold_days` days hit first) against every historically
  eligible day in the MOST RECENT `persistence_lookback` days (not stale,
  index-0-anchored history) -- see run_barrier_backtest. The resulting win rate
  and expected P&L are what actually matches "buy, take 3%, exit, repeat."
- Persistence score is independent of today's entry trigger
- Today's setup is a bonus for entry timing, not persistence ranking
- Results are written to BigQuery idempotently (safe to re-run same-day)
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

import numpy as np
import pandas as pd
from google.api_core.exceptions import NotFound
from google.cloud import bigquery

logging.basicConfig(level=logging.WARNING)
LOGGER = logging.getLogger(__name__)

NSE_EQUITY_LIST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}
DEFAULT_BQ_TABLE_ID = "daily_stocks_v2"

# =========================================================
# CONFIG
# =========================================================
@dataclass(frozen=True)
class ScannerConfig:
    # Data
    period: str = "2y"            # needs enough runway for two stacked 180d lookbacks
    interval: str = "1d"
    min_days: int = 220

    # Universe (hard filters)
    min_price: float = 450.0
    max_price: float = 1500.0
    min_avg_traded_value_cr: float = 10.0

    # Volatility identity (what makes a stock "volatile")
    volatility_lookback: int = 180
    range_event_threshold: float = 3.5
    close_move_threshold: float = 2.5
    min_avg_volatility: float = 3.5
    min_median_volatility: float = 2.5
    min_volatile_days: int = 30
    min_volatility_ratio: float = 0.20

    # Barrier backtest (tested against the MOST RECENT persistence_lookback days).
    # min_persistence_rate is a WIN RATE now (see run_barrier_backtest), not a
    # "did volatility touch X%" proxy. Breakeven win rate = risk/(risk+reward);
    # with target_pct=3.0 and min_rr=1.0 (risk capped ~3%, see min_rr below)
    # that's 50%. min_persistence_rate=50 is that breakeven line with zero
    # margin -- deliberately left there rather than padded, so the scanner's
    # actual historical hit rate is what does the filtering. Empirically (5
    # tickers from a real trade book, 2y data): AEGISLOG/SKYGOLD/ANTELOPUS
    # clear 50-56% and come out expected-P&L-positive; SKIPPER/BALUFORGE sit
    # at 39-41% and are expected-P&L-negative at every risk level tested --
    # i.e. this bar is doing real work, not rubber-stamping.
    persistence_lookback: int = 180
    min_persistence_sample: int = 60
    min_persistence_rate: float = 50.0
    # Used only as a scoring bonus (see calculate_score), not a hard reject --
    # a stock can have a strong win rate that's unevenly distributed across
    # sub-periods and still be worth surfacing (e.g. AEGISLOG.NS: 95%+ raw
    # persistence, 4.3% avg volatility, rejected outright by the old
    # stability > threshold*2 hard gate).
    persistence_stability_threshold: float = 4.0

    # Entry timing (independent of persistence)
    breakout_lookback: int = 20
    support_window: int = 20
    support_distance_threshold: float = 8.0
    breakout_volume_mult: float = 1.5
    pullback_min: float = 2.0
    pullback_max: float = 8.0

    # Trade levels / risk / exit
    atr_period: int = 14
    stop_atr_mult: float = 1.5
    recent_low_buffer: float = 0.975   # stop can't be looser than this * recent_low
    target_pct: float = 3.0            # FIXED exit target: target = entry * (1 + target_pct/100)
    max_risk_pct: float = 8.0          # safety-net hard filter; rarely binds since
                                        # add_indicators' trade_stop already caps risk
                                        # near target_pct/min_rr
    min_rr: float = 1.0                # risk is capped at target_pct/min_rr (~3% here)
                                        # in add_indicators, so this is a guaranteed floor,
                                        # not just a post-hoc filter. 1.0 was chosen
                                        # empirically: it's roughly where a fixed 3%
                                        # target's stop distance stops being tighter than
                                        # these stocks' own daily noise (see
                                        # min_persistence_rate comment above) -- pushing
                                        # min_rr higher tightens the stop below that noise
                                        # floor and the win rate collapses for everything.
    max_hold_days: int = 5

    # Quality gate -- rejects candidates outright rather than just ranking them lower.
    # This is the main lever for "fewer but better": raise it to cut junk,
    # lower it if legitimate setups are getting excluded.
    # NOTE: persistence_score (max 30) and expected_move_score (max 10) now
    # come from the barrier backtest's honest win rate / P&L-per-trade (see
    # calculate_score), which structurally score lower than the old inflated
    # proxy metrics did (~95% "persistence" / 5-7% "expected move" vs. real
    # win rates of 40-60% and per-trade P&L of 0-1.5%). 65 rejected every
    # single candidate tested, including known-profitable ones -- 40 is where
    # AEGISLOG/ANTELOPUS actually score. min_persistence_rate is what does
    # the expectancy filtering now; this gate mainly separates BUY-timed,
    # low-noise setups from the rest. Retune after a real backtest run.
    min_score: float = 40.0

    # Processing
    chunk_size: int = 100
    max_batch_retries: int = 2
    top_n: int = 200                   # high ceiling -- min_score does the real filtering
    verbose: bool = False


# =========================================================
# HELPERS
# =========================================================
def load_yfinance() -> Any:
    # yfinance's cookie/cache layer uses SQLite. Some minimal Cloud Run
    # Functions images omit the SQLite shared library, which otherwise makes
    # every ticker in every batch look like a failed Yahoo download.
    try:
        import sqlite3
        sqlite3.connect(":memory:").close()
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "This Cloud Function runtime has no usable SQLite driver, required by yfinance. "
            "Redeploy with the full Python base image (for example, python313 on google-22-full)."
        ) from exc

    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: yfinance. Install: pip install yfinance") from exc
    return yf


def to_yahoo_nse_ticker(symbol: str) -> str:
    symbol = symbol.strip().upper()
    return symbol if symbol.endswith(".NS") else f"{symbol}.NS"


def load_nse_tickers(source: str = NSE_EQUITY_LIST_URL) -> list[str]:
    symbols = pd.read_csv(source)["SYMBOL"].dropna().astype(str)
    return [to_yahoo_nse_ticker(s) for s in symbols]


def parse_tickers(raw: list[str]) -> list[str]:
    out = []
    for value in raw:
        out.extend(to_yahoo_nse_ticker(p) for p in value.split(",") if p.strip())
    return out


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[i:i + size] for i in range(0, len(values), size)]


def normalize_single_ticker_columns(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        data = data.copy()
        data.columns = [c[0] for c in data.columns]
    # FIX: NaN volume rows previously survived (only `dropna(how="all")` was
    # applied), which silently corrupted avg_volume20 / traded value / volume spike.
    return data.dropna(subset=[c for c in REQUIRED_COLUMNS if c in data.columns])


def download_batch(yf: Any, tickers: list[str], config: ScannerConfig) -> pd.DataFrame:
    return yf.download(
        tickers=tickers,
        period=config.period,
        interval=config.interval,
        progress=False,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
    )


def get_ticker_frame(batch_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if not isinstance(batch_data.columns, pd.MultiIndex):
        return normalize_single_ticker_columns(batch_data)
    if ticker not in batch_data.columns.get_level_values(0):
        return pd.DataFrame()
    return normalize_single_ticker_columns(batch_data[ticker].dropna(how="all"))


def has_enough_data(data: pd.DataFrame, config: ScannerConfig) -> bool:
    return not data.empty and len(data) >= config.min_days and REQUIRED_COLUMNS.issubset(data.columns)


# =========================================================
# INDICATORS
# =========================================================
def add_indicators(data: pd.DataFrame, config: ScannerConfig) -> pd.DataFrame:
    data = data.copy()

    data["range_pct"] = (data["High"] - data["Low"]) / data["Close"] * 100
    data["is_volatile_day_range"] = data["range_pct"] >= config.range_event_threshold

    data["close_move"] = data["Close"].pct_change(fill_method=None) * 100
    data["abs_close_move"] = data["close_move"].abs()
    data["is_volatile_day_close"] = data["abs_close_move"] >= config.close_move_threshold

    data["is_volatile_day"] = data["is_volatile_day_range"] | data["is_volatile_day_close"]
    data["volatility_measure"] = np.maximum(data["range_pct"], data["abs_close_move"])

    data["EMA20"] = data["Close"].ewm(span=20).mean()
    data["MA20"] = data["Close"].rolling(config.support_window).mean()

    data["recent_low"] = data["Low"].rolling(config.support_window).min()
    data["high_20_prev"] = data["High"].shift(1).rolling(config.breakout_lookback).max()
    data["high_20"] = data["High"].rolling(config.support_window).max()
    data["pullback_pct"] = ((data["high_20"] - data["Close"]) / data["high_20"]) * 100

    data["avg_volume20"] = data["Volume"].rolling(20).mean()
    data["avg_traded_value20_cr"] = (
        data["Close"].rolling(20).mean() * data["Volume"].rolling(20).mean()
    ) / 10_000_000

    # ATR(14) -- simple rolling mean of true range
    prev_close = data["Close"].shift(1)
    tr = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - prev_close).abs(),
            (data["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["ATR"] = tr.rolling(config.atr_period).mean()

    # Precomputed for run_barrier_backtest: was this stock already "volatile"
    # (trailing average, no lookahead) as of each day.
    data["past_vol_mean"] = (
        data["volatility_measure"].rolling(config.volatility_lookback, min_periods=20).mean().shift(1)
    )

    # Trade levels for EVERY row (vectorized, no lookahead -- ATR/recent_low/Close
    # are all trailing). The stop is the TIGHTEST (highest) of three candidates:
    #   - an ATR-based stop (stop_atr_mult * ATR below entry)
    #   - a structural floor (recent_low * recent_low_buffer)
    #   - a floor implied by min_rr against the FIXED target_pct target: the
    #     largest risk we can take and still clear min_rr for that target.
    # That third candidate is what keeps risk from ballooning past what a fixed
    # 3% target can justify on a wide-ATR (highly volatile) stock -- unlike a
    # target sized as a multiple of risk (which trivially always hits the
    # ratio), the target here is fixed at what the user actually wants.
    atr_stop = data["Close"] - config.stop_atr_mult * data["ATR"]
    structural_stop = data["recent_low"] * config.recent_low_buffer
    rr_floor_stop = data["Close"] * (1 - (config.target_pct / config.min_rr) / 100)
    data["trade_stop"] = pd.concat([atr_stop, structural_stop, rr_floor_stop], axis=1).max(axis=1)
    data["trade_target"] = data["Close"] * (1 + config.target_pct / 100)

    return data


# =========================================================
# BARRIER BACKTEST -- simulates the exact "buy, take target_pct%, exit" rule
# =========================================================
def run_barrier_backtest(data: pd.DataFrame, config: ScannerConfig) -> tuple[float, int, float, float]:
    """
    For each historically-eligible day (trailing avg volatility already >=
    min_avg_volatility as of that day -- no lookahead), simulates the EXACT
    rule this scanner surfaces today: buy at Close, exit at trade_target
    (entry + target_pct%) or trade_stop, whichever the next `max_hold_days`
    days hits first. A day that hits neither within the window is a TIMEOUT,
    exited at that final day's close (counted as neither a win nor free --
    its actual P&L is included in expected_pnl, and it does not count toward
    hits/win_rate).

    On a day that touches both stop and target, the stop is assumed hit
    first -- daily OHLC can't tell us the intraday order, so this is the
    conservative assumption.

    The 4 sub-periods are anchored to the END of the array (most recent
    `persistence_lookback` days), not absolute index 0, so a stock whose
    volatility regime changed recently isn't scored on stale behavior.

    Returns: (win_rate_pct, sample_size, expected_pnl_pct, stability_std)
      - win_rate_pct: % of trials that reached target_pct% before the stop/timeout.
      - expected_pnl_pct: mean REALIZED return across all trials (wins at
        +target_pct%, losses at their actual stop distance, timeouts at the
        close on the final held day) -- the expected P&L per trade this rule
        would have produced, not just a hit/miss rate.
    """
    lookback = config.persistence_lookback
    hold = config.max_hold_days
    if len(data) < lookback + hold:
        return 0.0, 0, 0.0, 999.0

    end_base = len(data) - hold
    start_base = max(0, end_base - lookback)
    period_size = max(1, lookback // 4)

    eligible = data["past_vol_mean"].to_numpy() >= config.min_avg_volatility
    close = data["Close"].to_numpy()
    high = data["High"].to_numpy()
    low = data["Low"].to_numpy()
    stop_arr = data["trade_stop"].to_numpy()
    target_arr = data["trade_target"].to_numpy()

    hits = 0
    total = 0
    pnl_pcts: list[float] = []
    win_rate_by_period: list[float] = []

    for period_num in range(4):
        p_start = start_base + period_num * period_size
        p_end = start_base + (period_num + 1) * period_size if period_num < 3 else end_base
        p_end = min(p_end, end_base)
        if p_start >= p_end:
            continue

        period_hits = 0
        period_total = 0
        for i in range(p_start, p_end):
            if not eligible[i]:
                continue
            entry = close[i]
            stop = stop_arr[i]
            target = target_arr[i]
            if not (np.isfinite(entry) and np.isfinite(stop) and np.isfinite(target)) or stop >= entry:
                continue

            exit_pct = None
            for j in range(i + 1, min(i + 1 + hold, len(data))):
                if low[j] <= stop:
                    exit_pct = (stop - entry) / entry * 100
                    break
                if high[j] >= target:
                    exit_pct = config.target_pct
                    hits += 1
                    period_hits += 1
                    break
            if exit_pct is None:
                last_idx = min(i + hold, len(data) - 1)
                exit_pct = (close[last_idx] - entry) / entry * 100

            pnl_pcts.append(exit_pct)
            total += 1
            period_total += 1

        if period_total > 0:
            win_rate_by_period.append(period_hits / period_total * 100)

    if total == 0:
        return 0.0, 0, 0.0, 999.0

    win_rate = round(hits / total * 100, 1)
    expected_pnl = round(float(np.mean(pnl_pcts)), 2)
    stability = round(float(np.std(win_rate_by_period)), 1) if len(win_rate_by_period) > 1 else 0.0

    return win_rate, total, expected_pnl, stability


def verify_volatility_identity(data: pd.DataFrame, config: ScannerConfig) -> tuple[float, float, int, float]:
    """Is this stock fundamentally volatile, judged over the most recent volatility_lookback days?"""
    lookback = config.volatility_lookback
    if len(data) < lookback:
        return 0.0, 0.0, 0, 0.0

    hist = data.tail(lookback)
    avg_volatility = float(hist["volatility_measure"].mean())
    median_volatility = float(hist["volatility_measure"].median())
    volatile_days = int(hist["is_volatile_day"].sum())
    volatility_ratio = volatile_days / len(hist)

    return avg_volatility, median_volatility, volatile_days, volatility_ratio


# =========================================================
# ENTRY TRIGGER ENGINE
# =========================================================
def get_entry_trigger(data: pd.DataFrame, config: ScannerConfig) -> tuple[str, str, str]:
    if len(data) < 2:
        return "WATCH", "insufficient", "Not enough data"

    latest = data.iloc[-1]
    prev = data.iloc[-2]

    close = float(latest["Close"])
    ema20 = float(latest["EMA20"])
    pullback = float(latest["pullback_pct"])
    high_20_prev = float(latest["high_20_prev"])
    avg_volume = float(latest["avg_volume20"])
    volume_spike = float(latest["Volume"]) / avg_volume if avg_volume > 0 else 0

    bullish = close > float(latest["Open"])
    closed_above_prev_high = close > float(prev["High"])

    if pd.isna(ema20) or pd.isna(high_20_prev) or pd.isna(pullback):
        return "WATCH", "no_data", "Indicators not ready"

    if close > high_20_prev and volume_spike >= config.breakout_volume_mult and bullish:
        return "BUY", "breakout", f"Broke 20D high on {volume_spike:.1f}x volume"

    if (config.pullback_min <= pullback <= config.pullback_max
            and close > ema20 and bullish and closed_above_prev_high):
        return "BUY", "pullback_bounce", f"Bounce from {pullback:.1f}% pullback, confirmed"

    prev_close = float(prev["Close"])
    prev_ema20 = float(prev["EMA20"])
    if close > ema20 and prev_close <= prev_ema20 and bullish:
        return "BUY", "reclaim", "Reclaimed EMA20 with bullish close"

    if pullback > 12:
        return "WATCH", "deep_pullback", f"{pullback:.1f}% pullback - wait for base"
    if pullback < config.pullback_min:
        return "WATCH", "extended", "Near highs - wait for pullback"
    if close < ema20:
        return "WATCH", "below_trend", "Below EMA20 - wait for reclaim"

    return "WATCH", "coiling", "Volatile & coiling - watch for trigger"


def compute_setup_age(data: pd.DataFrame, config: ScannerConfig, current_setup_type: str,
                       max_lookback: int = 15) -> int:
    """
    How many most-recent consecutive trading days (including today) has this
    exact setup_type held, walking backward one day at a time and re-running
    get_entry_trigger as of each prior day (no lookahead -- indicators are
    already trailing/causal, so truncating the frame is enough).

    Why this exists: a BUY trigger is frequently a SINGLE-DAY event (e.g.
    "reclaim" requires yesterday to have been below EMA20, by construction).
    DYCL.NS surfaced BUY/pullback_bounce for exactly one day (2026-08-12)
    then flipped to WATCH/coiling the next day and stayed there -- a report
    read or acted on a few days late looks identical to a fresh one unless
    something says otherwise. Setup_Age_Days == 1 means "triggered today";
    a WATCH row with a large value has been sitting a while.
    """
    age = 0
    for back in range(max_lookback):
        end = len(data) - back
        if end < 3:
            break
        _, setup_type, _ = get_entry_trigger(data.iloc[:end], config)
        if setup_type != current_setup_type:
            break
        age += 1
    return age

# =========================================================
# RISK / TRADE LEVELS
# =========================================================
def calculate_risk_reward(data: pd.DataFrame, config: ScannerConfig) -> Optional[dict[str, float]]:
    """
    Reads the trade_stop / trade_target columns add_indicators already
    computed for the latest row (see compute_trade_levels comment there for
    how the stop is sized). Candidates whose risk % or R:R don't clear
    config thresholds are rejected by the caller.
    """
    latest = data.iloc[-1]
    entry = float(latest["Close"])
    atr = float(latest["ATR"])
    stop_loss = float(latest["trade_stop"])
    target = float(latest["trade_target"])

    if not all(math.isfinite(x) for x in [entry, atr, stop_loss, target]) or atr <= 0:
        return None

    risk = entry - stop_loss
    if risk <= 0:
        return None

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


# =========================================================
# SCORING
# =========================================================
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
    persistence_score = min(persistence_rate / 100 * 30, 30)
    # expected_move is now the barrier backtest's mean realized P&L per trade
    # (see run_barrier_backtest), bounded above by target_pct (a perfect
    # win rate) -- score relative to that ceiling rather than a fixed divisor
    # so it stays meaningful if target_pct is retuned.
    expected_move_score = min(max(expected_move, 0) / config.target_pct * 10, 10)

    if persistence_stability < config.persistence_stability_threshold:
        stability_bonus = 10
    elif persistence_stability < config.persistence_stability_threshold * 1.5:
        stability_bonus = 5
    else:
        stability_bonus = 0

    avg_vol_score = min(avg_volatility / 8 * 15, 15)
    median_score = min(median_volatility / 6 * 10, 10)
    frequency_score = min(volatility_ratio * 5, 5)

    today_bonus = min(today_range / 6 * 8, 8)
    volume_bonus = min(volume_spike / 3 * 5, 5)

    if action == "BUY":
        setup_bonus = {"breakout": 7, "pullback_bounce": 6, "reclaim": 5}.get(setup_type, 3)
    else:
        setup_bonus = 0

    support_penalty = min(max(distance_from_support - config.support_distance_threshold, 0) * 0.3, 5)

    score = (
        persistence_score + expected_move_score + stability_bonus
        + avg_vol_score + median_score + frequency_score
        + today_bonus + volume_bonus + setup_bonus
        - support_penalty
    )
    return round(max(min(score, 100), 0), 2)

# =========================================================
# CANDIDATE BUILDER
# =========================================================
def build_candidate(ticker: str, data: pd.DataFrame, config: ScannerConfig, run_date: str) -> Optional[dict[str, Any]]:
    if len(data) < config.min_days:
        return None

    avg_volatility, median_volatility, volatile_days, volatility_ratio = verify_volatility_identity(data, config)

    latest = data.iloc[-1]
    today_range = float(latest["volatility_measure"])
    last_close = float(latest["Close"])
    ma20 = float(latest["MA20"])
    recent_low = float(latest["recent_low"])
    high_20 = float(latest["high_20"])
    avg_volume20 = float(latest["avg_volume20"])
    avg_traded_value20_cr = float(latest["avg_traded_value20_cr"])

    if (pd.isna(ma20) or pd.isna(recent_low) or pd.isna(high_20)
            or pd.isna(avg_volume20) or pd.isna(avg_traded_value20_cr)
            or last_close <= 0 or avg_volume20 <= 0):
        return None

    if not (config.min_price < last_close < config.max_price):
        return None
    if avg_traded_value20_cr < config.min_avg_traded_value_cr:
        return None

    if avg_volatility < config.min_avg_volatility:
        return None
    if median_volatility < config.min_median_volatility:
        return None
    if volatile_days < config.min_volatile_days:
        return None
    if volatility_ratio < config.min_volatility_ratio:
        return None

    persistence_rate, sample_size, expected_move, persistence_stability = run_barrier_backtest(data, config)
    if sample_size < config.min_persistence_sample:
        return None
    if persistence_rate < config.min_persistence_rate:
        return None
    # persistence_stability is scoring-only (see calculate_score) -- not a
    # hard reject. See the config field comment for why.

    action, setup_type, reason = get_entry_trigger(data, config)
    setup_age_days = compute_setup_age(data, config, setup_type)

    risk = calculate_risk_reward(data, config)
    if risk is None:
        return None
    if risk["risk_pct"] > config.max_risk_pct:
        return None
    if risk["rr_ratio"] < config.min_rr:
        return None

    distance_from_support = ((last_close - recent_low) / last_close) * 100
    distance_from_ma20 = ((last_close - ma20) / last_close) * 100
    volume_spike = float(latest["Volume"]) / avg_volume20

    score = calculate_score(
        avg_volatility, median_volatility, volatility_ratio,
        persistence_rate, persistence_stability, expected_move, today_range, volume_spike,
        action, setup_type, distance_from_support, config,
    )

    if score < config.min_score:
        return None

    return {
        "Ticker": ticker,
        "Bar_Date": pd.Timestamp(data.index[-1]).date().isoformat(),
        "Run_Date": run_date,
        "Action": action,
        "Setup_Type": setup_type,
        "Setup_Age_Days": setup_age_days,
        "Score": score,
        "Reason": reason,
        "Price": round(last_close, 2),
        "Avg_Volatility": round(avg_volatility, 2),
        "Median_Volatility": round(median_volatility, 2),
        "Volatile_Days": volatile_days,
        "Volatility_Ratio": round(volatility_ratio, 4),
        "Today_Range": round(today_range, 2),
        "Persistence_Rate": persistence_rate,
        "Persistence_Sample": sample_size,
        "Expected_Move": expected_move,
        "Persistence_Stability": persistence_stability,
        "Traded_Value_Cr": round(avg_traded_value20_cr, 2),
        "Volume_Spike": round(volume_spike, 2),
        "Pullback_Pct": round(float(latest["pullback_pct"]), 2),
        "Dist_Support_Pct": round(distance_from_support, 2),
        "Dist_MA20_Pct": round(distance_from_ma20, 2),
        "Entry": risk["entry"],
        "Stop_Loss": risk["stop_loss"],
        "Target": risk["target"],
        "Risk_Per_Share": risk["risk_per_share"],
        "Risk_Pct": risk["risk_pct"],
        "RR_Ratio": risk["rr_ratio"],
        "ATR": risk["atr"],
        "Max_Hold_Days": config.max_hold_days,
    }
# =========================================================
# SCANNER
# =========================================================
def scan_ticker_data(ticker: str, data: pd.DataFrame, config: ScannerConfig, run_date: str) -> Optional[dict[str, Any]]:
    if not has_enough_data(data, config):
        return None
    data = add_indicators(data, config)
    return build_candidate(ticker, data, config, run_date)


def scan_tickers(tickers: list[str], config: ScannerConfig, run_date: str) -> tuple[pd.DataFrame, list[str], int]:
    """Returns (results, failed_tickers, total_quality_candidates_before_top_n_cap).
    failed_tickers lets a systemic failure (bad batch, API change) surface
    instead of just looking like '0 candidates found'. total_quality_candidates
    lets a top_n truncation be visible instead of silently hiding real results."""
    yf = load_yfinance()
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    batches = chunks(tickers, config.chunk_size)
    total = len(batches)

    for bn, batch in enumerate(batches, start=1):
        batch_data = None
        remaining = batch
        for attempt in range(config.max_batch_retries + 1):
            try:
                batch_data = download_batch(yf, remaining, config)
                break
            except Exception as exc:
                if config.verbose:
                    print(f"\nBatch {bn} attempt {attempt + 1} failed: {exc}")
                batch_data = None

        if batch_data is None:
            failures.extend(batch)
        else:
            for ticker in batch:
                try:
                    candidate = scan_ticker_data(ticker, get_ticker_frame(batch_data, ticker), config, run_date)
                    if candidate:
                        results.append(candidate)
                except Exception as exc:
                    failures.append(ticker)
                    if config.verbose:
                        print(f"\nSkipped {ticker}: {exc}")

        print(f"\rProgress: {int(bn/total*100)}% ({bn}/{total}) | {len(results)} found", end="", flush=True)
    print()

    if failures:
        LOGGER.warning("%d/%d tickers failed or returned no data", len(failures), len(tickers))

    if not results:
        return pd.DataFrame(), failures, 0

    df = pd.DataFrame(results)
    df["_action_rank"] = df["Action"].map({"BUY": 0, "WATCH": 1}).fillna(2)
    df = df.sort_values(
        ["_action_rank", "Persistence_Rate", "Persistence_Stability", "Score"],
        ascending=[True, False, True, False],
    ).drop(columns=["_action_rank"])

    total_quality_candidates = len(df)
    return df.head(config.top_n).reset_index(drop=True), failures, total_quality_candidates


# =========================================================
# BIGQUERY
# =========================================================
BQ_SCHEMA = [
    bigquery.SchemaField("Ticker", "STRING"),
    bigquery.SchemaField("Bar_Date", "DATE"),
    bigquery.SchemaField("Run_Date", "DATE"),
    bigquery.SchemaField("Action", "STRING"),
    bigquery.SchemaField("Setup_Type", "STRING"),
    bigquery.SchemaField("Setup_Age_Days", "INT64"),
    bigquery.SchemaField("Score", "FLOAT64"),
    bigquery.SchemaField("Reason", "STRING"),
    bigquery.SchemaField("Price", "FLOAT64"),
    bigquery.SchemaField("Avg_Volatility", "FLOAT64"),
    bigquery.SchemaField("Median_Volatility", "FLOAT64"),
    bigquery.SchemaField("Volatile_Days", "INT64"),
    bigquery.SchemaField("Volatility_Ratio", "FLOAT64"),
    bigquery.SchemaField("Today_Range", "FLOAT64"),
    bigquery.SchemaField("Persistence_Rate", "FLOAT64"),
    bigquery.SchemaField("Persistence_Sample", "INT64"),
    bigquery.SchemaField("Expected_Move", "FLOAT64"),
    bigquery.SchemaField("Persistence_Stability", "FLOAT64"),
    bigquery.SchemaField("Traded_Value_Cr", "FLOAT64"),
    bigquery.SchemaField("Volume_Spike", "FLOAT64"),
    bigquery.SchemaField("Pullback_Pct", "FLOAT64"),
    bigquery.SchemaField("Dist_Support_Pct", "FLOAT64"),
    bigquery.SchemaField("Dist_MA20_Pct", "FLOAT64"),
    bigquery.SchemaField("Entry", "FLOAT64"),
    bigquery.SchemaField("Stop_Loss", "FLOAT64"),
    bigquery.SchemaField("Target", "FLOAT64"),
    bigquery.SchemaField("Risk_Per_Share", "FLOAT64"),
    bigquery.SchemaField("Risk_Pct", "FLOAT64"),
    bigquery.SchemaField("RR_Ratio", "FLOAT64"),
    bigquery.SchemaField("ATR", "FLOAT64"),
    bigquery.SchemaField("Max_Hold_Days", "INT64"),
]


def _schema_for_existing_table(table: bigquery.Table) -> tuple[list[bigquery.SchemaField], dict[str, str]]:
    """Preserve legacy field names/types while adding fields from BQ_SCHEMA.

    The original ``daily_stocks`` table has fields such as ``tikker`` and
    ``run_date`` and lacks ``Bar_Date``.  Field names cannot be guessed by
    BigQuery, and an existing INTEGER ``score`` cannot be loaded as FLOAT64.
    """
    existing = {field.name.lower(): field for field in table.schema}
    field_map: dict[str, str] = {}
    schema: list[bigquery.SchemaField] = []
    for expected in BQ_SCHEMA:
        # `tikker` is the one known spelling error in the legacy table.
        actual = existing.get(expected.name.lower())
        if expected.name == "Ticker" and actual is None:
            actual = existing.get("tikker")
        if actual is not None:
            field_map[expected.name] = actual.name
            schema.append(actual)
        else:
            field_map[expected.name] = expected.name
            schema.append(expected)
    return schema, field_map


def write_to_bigquery(df: pd.DataFrame, project_id: str, dataset_id: str, table_id: str) -> int:
    """Idempotent write: deletes any existing rows sharing (Ticker, Bar_Date)
    with the incoming data before appending, so re-running the scanner for
    the same trading day doesn't produce duplicate rows."""
    if df.empty:
        return 0

    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    try:
        table = client.get_table(table_ref)
        table_exists = True
    except NotFound:
        table = client.create_table(bigquery.Table(table_ref, schema=BQ_SCHEMA))
        LOGGER.info("Created %s", table_ref)
        table_exists = False

    load_schema, field_map = _schema_for_existing_table(table)
    ticker_field = field_map["Ticker"]
    # Legacy tables did not record a market bar date.  On the first upgraded
    # run use run_date for de-duplication; Bar_Date is then added by the load.
    dedupe_date_field = field_map["Bar_Date"]
    if dedupe_date_field == "Bar_Date" and not any(f.name == "Bar_Date" for f in table.schema):
        dedupe_date_field = field_map["Run_Date"]
        LOGGER.warning("%s lacks Bar_Date; using %s for this run's de-duplication", table_ref, dedupe_date_field)

    output = df.copy()
    output["Bar_Date"] = pd.to_datetime(output["Bar_Date"]).dt.date
    output["Run_Date"] = pd.to_datetime(output["Run_Date"]).dt.date
    output = output.rename(columns={source: target for source, target in field_map.items() if source != target})

    # The legacy table stores score as INTEGER.  Retain that established type
    # instead of failing the load; new tables keep the FLOAT64 schema above.
    score_field = next(field for field in load_schema if field.name == field_map["Score"])
    if score_field.field_type == "INTEGER":
        output[field_map["Score"]] = output[field_map["Score"]].round().astype("Int64")
    for field in load_schema:
        if field.field_type == "NUMERIC" and field.name in output:
            output[field.name] = output[field.name].map(
                lambda value: Decimal(str(value)) if pd.notna(value) else None
            )

    if table_exists:
        pairs = output[[ticker_field, dedupe_date_field]].drop_duplicates()
        conditions = " OR ".join(
            f"(`{ticker_field}` = '{getattr(row, ticker_field).replace(chr(39), chr(39) * 2)}' "
            f"AND `{dedupe_date_field}` = DATE('{getattr(row, dedupe_date_field).isoformat()}'))"
            for row in pairs.itertuples(index=False)
        )
        client.query(f"DELETE FROM `{table_ref}` WHERE {conditions}").result()

    job = client.load_table_from_dataframe(
        output,
        table_ref,
        job_config=bigquery.LoadJobConfig(
            schema=load_schema,
            write_disposition="WRITE_APPEND",
            schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
        ),
    )
    job.result()
    return len(output)


def _resolve_project_id(project_id: Optional[str]) -> str:
    # `GCP_PROJECT` is convenient locally, while Cloud Functions/Run exposes
    # the deployed project as `GOOGLE_CLOUD_PROJECT`.
    resolved = project_id or os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if not resolved:
        try:
            resolved = bigquery.Client().project
        except Exception:
            resolved = None
    if not resolved:
        raise ValueError(
            "No GCP project_id provided and neither GCP_PROJECT nor GOOGLE_CLOUD_PROJECT is set. "
            "Pass --project-id or configure a Cloud Function project."
        )
    return resolved


# =========================================================
# ARGS
# =========================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Volatility persistence swing scanner - buys volatile setups and "
                    "backtests exiting each one at a fixed profit target."
    )
    p.add_argument("--tickers", nargs="+", default=None)
    p.add_argument("--symbols-source", default=NSE_EQUITY_LIST_URL)
    p.add_argument("--limit", default=0, type=int)
    p.add_argument("--only-buy", action="store_true")

    p.add_argument("--min-avg-volatility", default=3.5, type=float)
    p.add_argument("--range-event-threshold", default=3.5, type=float)
    p.add_argument("--min-volatile-days", default=30, type=int)

    p.add_argument("--min-persistence", default=ScannerConfig.min_persistence_rate, type=float,
                    help="Minimum win rate (%%) from the barrier backtest -- see run_barrier_backtest")
    p.add_argument("--min-persistence-sample", default=ScannerConfig.min_persistence_sample, type=int)

    p.add_argument("--min-price", default=ScannerConfig.min_price, type=float)
    p.add_argument("--max-price", default=ScannerConfig.max_price, type=float)
    p.add_argument("--min-traded-value-cr", default=ScannerConfig.min_avg_traded_value_cr, type=float)

    p.add_argument("--target-pct", default=ScannerConfig.target_pct, type=float,
                    help="Fixed profit target %% above entry -- this IS the exit rule")
    p.add_argument("--min-rr", default=ScannerConfig.min_rr, type=float)
    p.add_argument("--max-risk-pct", default=ScannerConfig.max_risk_pct, type=float)
    p.add_argument("--max-hold-days", default=ScannerConfig.max_hold_days, type=int)
    p.add_argument("--min-score", default=ScannerConfig.min_score, type=float,
                    help="Hard quality gate (0-100); candidates scoring below this are rejected outright")

    p.add_argument("--top-n", default=ScannerConfig.top_n, type=int)
    p.add_argument("--output", default="", help="CSV output path")
    p.add_argument("--no-bq", action="store_true", help="Skip writing results to BigQuery")
    p.add_argument("--project-id", default=None, help="GCP project (else uses GCP_PROJECT env var)")
    p.add_argument("--dataset-id", default="data_options")
    p.add_argument(
        "--table-id",
        default=DEFAULT_BQ_TABLE_ID,
        help=f"BigQuery table (default: {DEFAULT_BQ_TABLE_ID}; created with the full scanner schema if absent)",
    )
    p.add_argument("--verbose", action="store_true")

    args, _ = p.parse_known_args()
    return args


# =========================================================
# MAIN
# =========================================================
def main(request: Any = None) -> Optional[tuple[str, int]]:
    """Run as either a CLI program or an HTTP Cloud Function.

    Functions Framework calls the configured entry point with the Flask
    request object.  The optional argument keeps ``python swings.py``
    working while making ``main`` a valid Cloud Function HTTP handler.
    """
    args = parse_args()
    config = ScannerConfig(
        min_avg_volatility=args.min_avg_volatility,
        range_event_threshold=args.range_event_threshold,
        min_volatile_days=args.min_volatile_days,
        min_persistence_rate=args.min_persistence,
        min_persistence_sample=args.min_persistence_sample,
        min_price=args.min_price,
        max_price=args.max_price,
        min_avg_traded_value_cr=args.min_traded_value_cr,
        target_pct=args.target_pct,
        min_rr=args.min_rr,
        max_risk_pct=args.max_risk_pct,
        max_hold_days=args.max_hold_days,
        min_score=args.min_score,
        top_n=args.top_n,
        verbose=args.verbose,
    )

    tickers = parse_tickers(args.tickers) if args.tickers else load_nse_tickers(args.symbols_source)
    if args.limit > 0:
        tickers = tickers[:args.limit]

    risk_cap_pct = config.target_pct / config.min_rr
    print(f"Universe: Rs.{config.min_price}-{config.max_price} | Liquidity >= {config.min_avg_traded_value_cr} Cr")
    print(f"Volatility: 6-month avg >= {config.min_avg_volatility}% | median >= {config.min_median_volatility}% | "
          f"{config.min_volatile_days}+ volatile days")
    print(f"Exit rule: +{config.target_pct}% target, ATR/structure stop capped at "
          f"~{risk_cap_pct:.2f}% risk (guarantees R:R >= {config.min_rr}), {config.max_hold_days}-day max hold")
    print(f"Barrier backtest: win rate >= {config.min_persistence_rate}% "
          f"({config.min_persistence_sample}+ eligible days in last {config.persistence_lookback})")
    print(f"Scanning {len(tickers)} tickers...")
    print("-" * 80)

    run_date = dt.datetime.now(dt.timezone.utc).date().isoformat()
    candidates, failures, total_quality = scan_tickers(tickers, config, run_date)

    if failures:
        print(f"\n{len(failures)} tickers failed or had no usable data.")
        if config.verbose:
            print(", ".join(failures[:30]))

    if candidates.empty:
        message = "No candidates matched. Try:"
        print(f"\n{message}")
        print("  --min-persistence 55        (lower persistence bar)")
        print("  --min-avg-volatility 3.0    (lower volatility bar)")
        print("  --min-persistence-sample 30 (allow smaller sample size)")
        print("  --min-rr 1.2                (relax risk/reward bar)")
        print("  --min-score 55              (relax the quality gate)")
        return (message, 200) if request is not None else None

    display = candidates
    if args.only_buy:
        display = candidates[candidates["Action"] == "BUY"]
        if display.empty:
            print("\nNo BUY signals today. Showing top WATCH candidates instead:")
            display = candidates.head(15)

    print("\n" + "-" * 100)
    print("RESULTS (sorted: BUY first, then by persistence, stability, score)")
    print("Age = consecutive days this Setup has held. BUY only trust Age==1 as a fresh")
    print("trigger -- most BUY setups are single-day; re-run before acting on an old report.")
    print("-" * 100)
    header = (f"{'Ticker':<13} {'Action':<7} {'Setup':<16} {'Age':>4} {'Score':>6} {'Vol%':>6} "
              f"{'Persist%':>9} {'Stabil':>7} {'ExpMove':>8} {'Today%':>7} {'RR':>5} "
              f"{'Entry':>9} {'SL':>9} {'Target':>9}")
    print(header)
    print("-" * 100)
    for _, r in display.iterrows():
        print(f"{r['Ticker']:<13} {r['Action']:<7} {r['Setup_Type']:<16} {r['Setup_Age_Days']:>4} "
              f"{r['Score']:>6.1f} {r['Avg_Volatility']:>5.1f}% "
              f"{r['Persistence_Rate']:>8.0f}% {r['Persistence_Stability']:>6.1f} "
              f"{r['Expected_Move']:>7.1f}% {r['Today_Range']:>6.1f}% {r['RR_Ratio']:>5.1f} "
              f"{r['Entry']:>9.2f} {r['Stop_Loss']:>9.2f} {r['Target']:>9.2f}")

    if args.output:
        candidates.to_csv(args.output, index=False)
        print(f"\nSaved {len(candidates)} rows to {args.output}")

    if total_quality > len(candidates):
        print(f"\nNote: {total_quality} total quality candidates found, only top {config.top_n} shown/saved. "
              f"Raise --top-n to see the rest.")
    else:
        print(f"\nTotal quality candidates: {total_quality}")

    if not args.no_bq:
        try:
            project_id = _resolve_project_id(args.project_id)
            n = write_to_bigquery(candidates, project_id, args.dataset_id, args.table_id)
            print(f"Wrote {n} rows to {project_id}.{args.dataset_id}.{args.table_id}")
        except Exception as exc:
            print(f"\nBigQuery write skipped/failed: {exc}")
            print("Use --no-bq to suppress this, or --project-id / set GCP_PROJECT to fix it.")

    if request is not None:
        return (f"Scanner completed: {len(candidates)} candidates processed.", 200)
    return None


if __name__ == "__main__":
    main()
