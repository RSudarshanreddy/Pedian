"""
VOLATILITY SWING SCANNER
========================
Core Philosophy:
- A stock is "volatile" based on its full 6-month history (not a rolling 20-day window)
- The scanner measures HOW FAR a stock typically travels over
  config.move_horizon_days (default 30 sessions), in BOTH directions -- a median
  best gain and a median worst loss, both net of transaction costs. See
  run_move_profile. It does not threshold that into a win rate, because a window
  that ran +19% and one that fell -5% are not the same thing and a pass/fail bar
  scores them identically.
- It does NOT model an exit. There is no profit floor, no holding clock and no
  simulated sell. Exits are a human decision; this answers only "which stocks
  habitually make moves worth holding for, and what would you sit through".
- Candidates are gated on the SIZE of that typical move and on its asymmetry
  (gain must at least match pain), not on clearing an arbitrary profit bar.
  This is deliberate: the previous 3%-floor win rate correlated -0.080 with the
  30-session forward peak and scored the four best real trades at 5.6, 10.0 and
  17.4 out of 100, rejecting a fourth outright.
- Stops are ATR/structural only. The old rr_floor_stop that capped risk at
  exactly 3% is gone -- a 3% stop on a stock with 4-6% average daily range sits
  inside its own noise.
- Today's setup is a bonus for entry timing, not for the move profile
- Results are written to BigQuery per run (keyed on Run_Timestamp)
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
import os
import time
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
DEFAULT_BQ_TABLE_ID = "swings"

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
    # The band is an affordability/position-sizing preference, not a predictor:
    # realized outcome correlated -0.030 with entry price across the tested
    # sample, i.e. nothing. Widening it changes how many names you see, not
    # their quality. Liquidity is policed separately by
    # min_avg_traded_value_cr, so lowering the floor does not admit thin
    # stocks -- a 250-rupee name still has to trade 10 Cr/day.
    min_price: float = 250.0
    max_price: float = 1400.0
    min_avg_traded_value_cr: float = 10.0

    # Volatility identity (what makes a stock "volatile")
    volatility_lookback: int = 180
    range_event_threshold: float = 3.5
    close_move_threshold: float = 2.5
    min_avg_volatility: float = 3.5
    min_median_volatility: float = 2.5
    min_volatile_days: int = 30
    min_volatility_ratio: float = 0.20

    # Move profile (measured over the MOST RECENT persistence_lookback days).
    #
    # What changed and why: this used to be a 3%-floor win rate, because the
    # original goal was "sell at minimum 3% profit and repeat". That goal is
    # gone. The four best trades on record were AEGISLOG +9%, DYCL +21%,
    # ANTELOPUS +46% and SHILPAMED +59%, all held for weeks. Measured against
    # the old metric those entries scored 5.6, 10.0 and 17.4 out of 100 and
    # DYCL was REJECTED outright by min_persistence_rate at 23.9% -- the gate
    # was actively deselecting the trades that made money.
    #
    # Confirmed on 2,073 forward windows across 536 tickers: the old
    # Persistence_Rate correlated -0.080 with the 30-session forward peak,
    # while a stock's own history of large moves correlated +0.158 and produced
    # a monotonic gradient (bottom quintile 7.6% hit rate, top 23.8%). It also
    # survived a control for volatility -- within a FIXED volatility band the
    # low-move-rate tercile stayed worst in all three bands -- so it is not
    # just "volatile stocks are volatile" restated.
    persistence_lookback: int = 180
    min_persistence_sample: int = 60
    # Forward window the move profile measures over. 30 sessions ~ 6 weeks,
    # which is the timescale the winning trades actually played out on
    # (SHILPAMED reached +20.7% in 8 sessions and kept running to +58.8%).
    move_horizon_days: int = 30
    # Gate on typical move SIZE, not on a pass/fail rate. This is the
    # "capture great ones" filter: it selects stocks that habitually travel,
    # rather than rejecting those that fail to clear an arbitrary bar.
    # Calibrated below against the live universe.
    min_typical_move_pct: float = 10.0
    # Hard floor on the asymmetry. A stock whose typical drawdown is worse than
    # its typical gain is not a candidate no matter how far it travels -- that
    # is the failure mode the top move-rate quintile exhibits (23.8% chance of
    # +20% but 9.4% chance of -20%). Set to 1.0 = "typical gain must at least
    # match typical pain".
    min_move_ratio: float = 1.0
    persistence_stability_threshold: float = 4.0

    # Entry timing (independent of persistence)
    breakout_lookback: int = 20
    support_window: int = 20
    support_distance: float = 8.0
    breakout_volume_mult: float = 2.3
    pullback_min: float = 2.0
    pullback_max: float = 8.0

    # Trade levels / risk / exit
    atr_period: int = 14
    stop_atr_mult: float = 1.5
    recent_low_buffer: float = 0.975   # stop can't be looser than this * recent_low
    target_pct: float = 3.0            # MINIMUM profit floor, not a fixed exit point:
                                        # floor = entry * (1 + target_pct/100). Once hit,
                                        # the trade extends (see max_extension_days) rather
                                        # than exiting immediately.
    max_extension_days: int = 3        # after the floor is reached, keep holding up to this
                                        # many more sessions as long as each day closes higher
                                        # than the previous close; exit on the first non-higher
                                        # close (or sooner if the stop is hit). This is what
                                        # lets a trade capture more than target_pct% when a
                                        # stock is genuinely still moving -- see
                                        # run_barrier_backtest for the exact simulation.
    max_risk_pct: float = 8.0          # safety-net hard filter; rarely binds since
                                        # add_indicators' trade_stop already caps risk
                                        # near target_pct/min_rr
    min_rr: float = 1.0                # risk is capped at target_pct/min_rr (~3% here)
                                        # in add_indicators, so this is a guaranteed floor
                                        # AT THE MINIMUM TARGET, not a post-hoc filter --
                                        # actual realized R:R is often higher once extension
                                        # captures more upside. 1.0 was chosen empirically:
                                        # it's roughly where a 3% floor's stop distance stops
                                        # being tighter than these stocks' own daily noise (see
                                        # min_persistence_rate comment above) -- pushing min_rr
                                        # higher tightens the stop below that noise floor and
                                        # the win rate collapses for everything.
    max_hold_days: int = 5             # window to reach the floor in the first place; the
                                        # extension (max_extension_days) is additional time
                                        # on top of this, only once the floor is hit

    # Round-trip transaction cost, deducted from every simulated trial so the
    # backtest reports NET expectancy instead of gross. Without this the whole
    # model overstates every result, and at a 3% target the drag is not a
    # rounding error -- it is ~8% of the entire objective.
    #
    # Built up from Zerodha NSE delivery equity, not guessed:
    #   STT           0.1% buy + 0.1% sell        = 0.2000%
    #   exchange txn  0.00297% each side          = 0.0059%
    #   SEBI turnover 0.0001% each side           = 0.0002%
    #   stamp duty    0.015% on buy only          = 0.0150%
    #   GST 18% on (txn + SEBI)                   = 0.0011%
    #   brokerage     nil on delivery             = 0
    #                                        total  0.2222%
    # Plus a FLAT DP charge of ~Rs.15.93 per scrip per sell day, which is why
    # the percentage depends on position size: it is 0.06% on a Rs.25,000
    # position but 0.16% on Rs.10,000. 0.25% corresponds to roughly Rs.25,000
    # per trade. Size smaller than that and this figure is too kind.
    #
    # Corroborated by the live account: charges over Jul-Sep roughly equalled
    # the entire realized P&L for that period.
    round_trip_cost_pct: float = 0.25

    # Quality gate -- rejects candidates outright rather than just ranking them lower.
    #
    # NOT COMPARABLE TO ANY EARLIER VALUE. calculate_score has been rescaled
    # twice: it dropped the 38 points that duplicated hard gates, then anchored
    # every component between its minimum admissible value and an excellent one
    # (see the ANCHORS block). The scale now genuinely runs 0-100 rather than
    # bunching: across 529 post-rework candidates it spans 2.4-76.0 with sd
    # 13.4, against 21.2-63.6 and sd 5.7 before.
    #
    # 5 keeps this near non-binding (99% of that population clears it), on
    # purpose: min_persistence_rate does the expectancy filtering, and each
    # rescale should change the ORDER and SPREAD of the list, not silently
    # change which stocks appear. Leaving it at 20 would now cut 25%.
    #
    # This is the lever for "fewer but better", and it is finally meaningful
    # because the scale is spread out. Measured survivor counts on that
    # population: 10 keeps 95%, 15 keeps 87%, 20 keeps 75%, 25 keeps 50%,
    # 30 keeps 40%. Nothing scored above 76 in three weeks, so a threshold
    # above ~55 will return almost nothing.
    min_score: float = 5.0

    # Processing
    chunk_size: int = 100
    max_batch_retries: int = 2
    # Fraction of downloaded tickers that may raise INSIDE scan_ticker_data
    # before scan_tickers treats it as a code fault and raises instead of
    # returning an empty result. Legitimate thin-history tickers return None
    # rather than raising, so a healthy run sits near zero here. 0.25 is well
    # clear of that while still catching the total-failure case.
    max_ticker_error_rate: float = 0.25
    top_n: int = 200                   # high ceiling -- min_score does the real filtering
    verbose: bool = False

    def __post_init__(self) -> None:
        # min_rr divides target_pct in add_indicators' rr_floor_stop and in
        # main's risk_cap_pct print. It's exposed via --min-rr, so 0 is
        # reachable from the CLI and would blow up mid-scan with a bare
        # ZeroDivisionError. Fail loudly here instead.
        if self.min_rr <= 0:
            raise ValueError(f"min_rr must be > 0 (got {self.min_rr}) -- it divides target_pct")
        if self.target_pct <= 0:
            raise ValueError(f"target_pct must be > 0 (got {self.target_pct})")
        if self.min_price >= self.max_price:
            raise ValueError(f"min_price ({self.min_price}) must be < max_price ({self.max_price})")


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


def _cache_nse_tickers(tickers: list[str], project_id: str) -> None:
    """Best-effort -- a caching failure should never break a live fetch that
    just succeeded. WRITE_TRUNCATE: this is a snapshot of the current
    universe, not history worth accumulating."""
    try:
        client = bigquery.Client(project=project_id)
        df = pd.DataFrame({"Ticker": tickers, "Cached_At": dt.datetime.now(dt.timezone.utc)})
        client.load_table_from_dataframe(
            df, f"{project_id}.data_options.nse_ticker_cache",
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
        ).result()
    except Exception as exc:
        LOGGER.warning("Failed to cache NSE ticker list (non-fatal): %s", exc)


def _load_cached_nse_tickers(project_id: str) -> list[str]:
    try:
        client = bigquery.Client(project=project_id)
        rows = client.query(f"SELECT Ticker FROM `{project_id}.data_options.nse_ticker_cache`").result()
        return [row["Ticker"] for row in rows]
    except Exception:
        return []


def load_nse_tickers(source: str = NSE_EQUITY_LIST_URL, project_id: Optional[str] = None) -> list[str]:
    """
    Retries the live NSE fetch a few times, then falls back to the last
    successfully-fetched list cached in BigQuery.

    Confirmed live: Cloud Run's egress IP gets a persistent 403 Forbidden
    from archives.nseindia.com (the identical fetch works fine from other
    IPs; not transient -- 5 straight scheduled runs failed the same way
    over 6.5 hours). Retrying from the same blocked IP won't fix that, but
    a stock universe list changes rarely (a handful of tickers a month), so
    falling back to a slightly stale cached list is a far better failure
    mode than the scanner producing nothing at all.
    """
    try:
        resolved_project = _resolve_project_id(project_id)
    except Exception:
        resolved_project = None

    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            symbols = pd.read_csv(source)["SYMBOL"].dropna().astype(str)
            tickers = [to_yahoo_nse_ticker(s) for s in symbols]
            if resolved_project:
                _cache_nse_tickers(tickers, resolved_project)
            return tickers
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(5)

    if resolved_project:
        cached = _load_cached_nse_tickers(resolved_project)
        if cached:
            LOGGER.warning(
                "Live NSE fetch failed after retries (%s) -- using %d cached tickers instead",
                last_exc, len(cached),
            )
            return cached
    raise last_exc


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
    # Baseline for the volume SPIKE: the 20 sessions BEFORE today, excluding
    # today. avg_volume20 above includes today, so using it as the denominator
    # lets a spike dilute its own baseline and understate itself -- worst
    # exactly on the days that matter most. For a true V = k * A spike the
    # included-today ratio computes as 20k/(19+k), so 2.3x reads as 2.16x and
    # 5x reads as 4.17x. That silently made the breakout gate
    # (breakout_volume_mult = 2.3) behave as 2.47x, ~7% stricter than written,
    # and cost real points off VOLUME_PTS in calculate_score.
    # Same .shift(1).rolling(...) idiom as high_20_prev above.
    data["avg_volume20_prior"] = data["Volume"].shift(1).rolling(20).mean()
    # Liquidity deliberately still uses the inclusive average -- it is a
    # "is this tradeable" measure, not a deviation-from-normal one, and
    # including today is the more current answer.
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
    # are all trailing). The stop is the TIGHTEST (higher) of two candidates:
    #   - an ATR-based stop (stop_atr_mult * ATR below entry)
    #   - a structural floor (recent_low * recent_low_buffer)
    #
    # A THIRD candidate used to sit here: rr_floor_stop, which capped risk at
    # target_pct/min_rr = exactly 3%. It has been removed. Its only purpose was
    # to keep risk proportionate to a fixed 3% profit target, and that target is
    # gone -- the scanner now measures how far a stock travels over 30 sessions
    # instead. Keeping it would have been actively harmful: a 3% stop on a stock
    # with 4-6% average daily range is inside the noise, so a position held for
    # a multi-week move would be stopped out almost immediately by ordinary
    # fluctuation. It also made max_risk_pct and min_rr unreachable, which is
    # why 384 of 384 stored rows had Risk_Pct == exactly 3.00.
    atr_stop = data["Close"] - config.stop_atr_mult * data["ATR"]
    structural_stop = data["recent_low"] * config.recent_low_buffer
    data["trade_stop"] = pd.concat([atr_stop, structural_stop], axis=1).max(axis=1)
    # Target is filled in per-stock by build_candidate from that stock's own
    # measured typical move, so it is no longer a flat percentage applied to
    # everything. Placeholder here keeps the column shape for any caller
    # reading the frame directly.
    data["trade_target"] = np.nan

    return data


# =========================================================
# MOVE PROFILE -- measures how far a stock actually travels over
# move_horizon_days, in both directions (see run_move_profile)
#
# What was here before: _stop_fill and _simulate_floor_extension_trade, which
# simulated "buy, never sell below +3%, extend while still closing higher".
# Both are deleted. They implemented a fixed-profit-floor exit rule that is no
# longer the goal, and nothing calls them. position_check.check_position_momentum
# still carries its own inline copy of that rule -- that is a separate change.
# =========================================================
def run_move_profile(data: pd.DataFrame, config: ScannerConfig) -> tuple[float, int, float, float]:
    """
    Measures HOW FAR this stock typically moves, rather than whether it cleared
    an arbitrary bar.

    This replaces run_barrier_backtest, which asked "what fraction of trials
    cleared a 3% floor within 5 days". That question came from the original
    "sell at minimum 3% and repeat" goal and stopped matching how the account
    actually made money: the four best trades on record were +9%, +21%, +46%
    and +59% held over weeks, not 3% scalps. Worse, a pass/fail bar throws away
    the only thing that matters here -- a window that ran +19% and one that fell
    -5% both scored zero.

    So nothing is thresholded. For every historically-eligible day (trailing avg
    volatility already >= min_avg_volatility as of that day -- no lookahead),
    this looks forward move_horizon_days sessions and records two magnitudes:
    how far the close got ABOVE entry at its best, and how far BELOW at its
    worst. Both are net of round_trip_cost_pct.

    Reporting both is deliberate. Measured across 536 tickers, the stocks that
    move hardest upward also fall hardest: the top quintile by big-move
    frequency hit +20% in 23.8% of windows but ALSO hit -20% in 9.4% of them,
    against 1.2% for the bottom quintile. A single "how big does it move" figure
    hides that entirely, which is how a volatile stock gets mistaken for a good
    one.

    The 4 sub-periods are anchored to the END of the array (most recent
    persistence_lookback days), not index 0, so a stock whose behaviour changed
    recently isn't judged on stale history.

    Returns: (typical_move_pct, sample_size, typical_drawdown_pct, move_stability)
      - typical_move_pct: MEDIAN best gain reached within the horizon. Median,
        not mean, because one 300% window would otherwise define a stock.
      - typical_drawdown_pct: MEDIAN worst loss within the same windows
        (negative). What you would have had to sit through.
      - move_stability: std of typical_move_pct across the 4 sub-periods. Low
        means the stock behaves consistently; high means it had one good spell.
    """
    lookback = config.persistence_lookback
    horizon = config.move_horizon_days
    cost = config.round_trip_cost_pct
    if len(data) < lookback + horizon:
        return 0.0, 0, 0.0, 999.0

    end_base = len(data) - horizon
    start_base = max(0, end_base - lookback)
    period_size = max(1, lookback // 4)

    eligible = data["past_vol_mean"].to_numpy() >= config.min_avg_volatility
    close = data["Close"].to_numpy()
    n = len(data)

    peaks: list[float] = []
    troughs: list[float] = []
    move_by_period: list[float] = []

    for period_num in range(4):
        p_start = start_base + period_num * period_size
        p_end = start_base + (period_num + 1) * period_size if period_num < 3 else end_base
        p_end = min(p_end, end_base)
        if p_start >= p_end:
            continue

        period_peaks: list[float] = []
        for i in range(p_start, p_end):
            if not eligible[i]:
                continue
            entry = close[i]
            if not np.isfinite(entry) or entry <= 0:
                continue
            window = close[i + 1: i + 1 + horizon]
            window = window[np.isfinite(window)]
            if len(window) < horizon // 2:
                continue
            peak = (window.max() - entry) / entry * 100 - cost
            trough = (window.min() - entry) / entry * 100 - cost
            peaks.append(peak)
            troughs.append(trough)
            period_peaks.append(peak)

        if period_peaks:
            move_by_period.append(float(np.median(period_peaks)))

    if not peaks:
        return 0.0, 0, 0.0, 999.0

    typical_move = round(float(np.median(peaks)), 2)
    typical_drawdown = round(float(np.median(troughs)), 2)
    stability = round(float(np.std(move_by_period)), 1) if len(move_by_period) > 1 else 0.0

    return typical_move, len(peaks), typical_drawdown, stability


def upside_rate(data: pd.DataFrame, config: ScannerConfig) -> float:
    """
    Share of forward windows in which the best gain beat the worst loss in
    magnitude -- "when this stock moves, how often does it move UP first/further".

    This is the honest replacement for the old win rate. It is not a probability
    of profit and must not be read as one: it says the upside dominated the
    downside in that window, not that a trade would have been exited there.
    Exits are a human decision (see position_check), so nothing here simulates
    one.
    """
    lookback = config.persistence_lookback
    horizon = config.move_horizon_days
    if len(data) < lookback + horizon:
        return 0.0

    end_base = len(data) - horizon
    start_base = max(0, end_base - lookback)
    eligible = data["past_vol_mean"].to_numpy() >= config.min_avg_volatility
    close = data["Close"].to_numpy()

    wins = total = 0
    for i in range(start_base, end_base):
        if not eligible[i]:
            continue
        entry = close[i]
        if not np.isfinite(entry) or entry <= 0:
            continue
        window = close[i + 1: i + 1 + horizon]
        window = window[np.isfinite(window)]
        if len(window) < horizon // 2:
            continue
        peak = (window.max() - entry) / entry * 100
        trough = (window.min() - entry) / entry * 100
        total += 1
        wins += peak > abs(trough)
    return round(wins / total * 100, 1) if total else 0.0


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
    # Prior-window baseline, not the inclusive one -- see avg_volume20_prior
    # in add_indicators for why the inclusive average understates a spike.
    avg_volume = float(latest["avg_volume20_prior"])
    volume_spike = (
        float(latest["Volume"]) / avg_volume
        if np.isfinite(avg_volume) and avg_volume > 0 else 0
    )

    bullish = close > float(latest["Open"])
    closed_above_prev_high = close > float(prev["High"])

    if pd.isna(ema20) or pd.isna(high_20_prev) or pd.isna(pullback):
        return "WATCH", "no_data", "Indicators not ready"

    # breakout and pullback_bounce both BUY. reclaim stays WATCH-only.
    #
    # No win-rate claim is being made by that split -- across three
    # independent 400-500 stock samples the three setups were statistically
    # indistinguishable (breakout 22.2%/27.3%, pullback_bounce 25.8%/21.7%,
    # reclaim 13.3%/26.7%). The ranking FLIPPED between samples, so at
    # n=15-46 per setup those gaps are sampling noise, and no ordering of
    # the three survives a second sample.
    #
    # What IS measured, and worth keeping in view when acting on a breakout:
    # it fires a median +6.35% AFTER the move (979 real trigger days; 87%
    # of them closed >3% up, 62% >5%). The 2.3x volume requirement
    # guarantees that lateness -- a stock cannot trade 2.3x normal volume
    # without having already moved. That is a worse entry PRICE for a 3%
    # target, though note the backtest already enters at the next day's open
    # and so already prices it in; it is not an extra unmeasured penalty.
    #
    # Cost note: two BUY triggers instead of one roughly restores the
    # candidate count (~60% more than pullback-only), and per-trade STT/DP
    # drag is real at this frequency -- charges roughly matched the entire
    # realized P&L for Jul-Sep. More signals is not free.
    if close > high_20_prev and volume_spike >= config.breakout_volume_mult and bullish:
        return "BUY", "breakout", f"Broke 20D high on {volume_spike:.1f}x volume"

    if (config.pullback_min <= pullback <= config.pullback_max
            and close > ema20 and bullish and closed_above_prev_high):
        return "BUY", "pullback_bounce", f"Bounce from {pullback:.1f}% pullback, confirmed"

    prev_close = float(prev["Close"])
    prev_ema20 = float(prev["EMA20"])
    if close > ema20 and prev_close <= prev_ema20 and bullish:
        return "WATCH", "reclaim", "Reclaimed EMA20 with bullish close - watch only"

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
def calculate_risk_reward(data: pd.DataFrame, config: ScannerConfig,
                          typical_move_pct: float) -> Optional[dict[str, float]]:
    """
    Reads the trade_stop column add_indicators computed for the latest row (see
    the comment there for how the stop is sized) and derives the target from
    THIS STOCK's own measured typical move.

    The target used to be a flat config.target_pct above entry, identical for
    every stock. It is now entry * (1 + typical_move_pct/100), so a stock that
    habitually travels 15% gets a 15% target and one that travels 8% gets 8%.
    That also makes rr_ratio meaningful for the first time: it compares this
    stock's usual move against this stock's own stop distance, instead of
    comparing a constant 3% against a stop that had been sized to produce
    exactly that ratio.
    """
    latest = data.iloc[-1]
    entry = float(latest["Close"])
    atr = float(latest["ATR"])
    stop_loss = float(latest["trade_stop"])
    target = entry * (1 + typical_move_pct / 100)

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
# Score point budget. The positive weights sum to 100 so the number reads as
# a percentage-like figure; SUPPORT_PENALTY_PTS is subtracted on top.
#
# WHY THESE AND NOT THE OLD ONES: the previous budget spent 38 of its 100
# points on avg_volatility (15), median_volatility (10), volatility_ratio (5)
# and today_range (8) -- every one of which is ALREADY a hard gate in
# build_candidate (min_avg_volatility, min_median_volatility,
# min_volatility_ratio). Scoring a filter you have already applied cannot
# rank the survivors, because they all passed it. Measured on a live
# 84-candidate run: those four averaged 27.2 points, 60% of a typical score
# of 45, while spanning only 6.6-11.7 of 15, 5.2-9.1 of 10, 2.3-4.6 of 5 and
# 3.3-8.0 of 8. They set the score's LEVEL and barely touched its ORDER, and
# that is the whole reason every score landed between 40 and 60 (mean 44.97,
# sd 4.01, full range 40.1-55.4 on a 100-point scale).
#
# Dropping them and re-budgeting widened the spread to 22.2 points with sd
# 5.49 on the same candidates -- 37% more discrimination -- and reordered the
# list materially (rank correlation +0.47 against the old score, 3/10 overlap
# in the top ten).
#
# Persistence takes half the budget because it IS the barrier backtest's
# measured win rate for that stock, the most direct expectancy estimate
# available here. Stability is deliberately small: it returns only 0, half or
# full (a 3-step flag, not a measurement), yet under the old budget it drove
# 37.8% of all ranking variance -- more than persistence, expected_move and
# volume combined.
#
# NOT A PREDICTIVE CLAIM. Score correlated -0.153 with realized outcome
# before this change, i.e. mildly ANTI-predictive, and rebudgeting the
# components does not create signal that was never measured. What it fixes is
# a score that could not discriminate and did not match the sort order. Treat
# the ranking as "which of these best fits the rule we backtested", not as a
# forecast.
PERSISTENCE_PTS = 50
EXPECTED_MOVE_PTS = 20
STABILITY_PTS = 10
VOLUME_PTS = 8
SETUP_PTS = 12
SUPPORT_PENALTY_PTS = 10

# ANCHORS -- what counts as 0 and what counts as full marks for each component.
#
# The rebudget above fixed WHICH components are scored. This fixes the scale
# they are scored on, which was the same mistake one level down: each component
# was a fraction of a theoretical maximum that cannot occur, so the score could
# never approach 100 and never approached 0 either.
#
# Persistence was the worst case. It is worth PERSISTENCE_PTS at a 100% win
# rate, but min_persistence_rate already guarantees >=25% and the real ceiling
# across 529 post-rework candidates is 46.1%. So it could only ever return
# 12.5-23 of its 50 points -- and the first 12.5 were handed free to every
# candidate that cleared the gate, exactly the "scoring a filter you already
# applied" error that removed the volatility block.
#
# Volume had the same flaw in miniature: scoring from 0 meant a stock trading
# its NORMAL volume collected a third of the budget for being unremarkable.
# 1.0x is average by definition, so that is where the scale starts.
#
# The persistence floor is read from config.min_persistence_rate rather than
# hardcoded, so raising the gate re-anchors the scale automatically instead of
# silently re-introducing the dead zone.
#
# Measured effect on those 529 candidates: spread 21.2-63.6 -> 2.4-76.0, sd
# 5.65 -> 13.38, and every one of the six components reaches full budget at
# least once instead of none of them doing so.
#
# 100 is now attainable but demanding: it needs >=45% win rate AND >=target_pct
# expected move AND stability under the threshold AND a 3x volume spike AND a
# breakout, together. Nothing scored above 76 in three weeks, which is the
# intended behaviour of an absolute scale -- a high score should be rare rather
# than rescaled into existence each day. The median sits near 25 because most
# candidates genuinely are marginal: 67% are passive setups at below-average
# volume.
EXCELLENT_MOVE_PCT = 20.0           # a stock that typically travels 20% in 30 sessions
EXCELLENT_UPSIDE_RATE = 65.0        # upside beat downside in ~2 of 3 windows
NORMAL_VOLUME_RATIO = 1.0           # 1x IS the average -- no credit for it
EXCELLENT_VOLUME_RATIO = 3.0


def _span(value: float, low: float, high: float) -> float:
    """Fraction of the way from `low` to `high`, clamped to 0-1."""
    if high <= low:
        return 0.0
    return min(max((value - low) / (high - low), 0.0), 1.0)

# Relative weight of each trigger within SETUP_PTS. The ordering carries no
# win-rate claim -- across three independent 400-500 stock samples the three
# setups were statistically indistinguishable and their ranking FLIPPED
# between samples (see get_entry_trigger). It exists so that a row which
# actually triggered outranks a passive state, nothing more.
SETUP_WEIGHTS = {"breakout": 1.0, "pullback_bounce": 6 / 7, "reclaim": 5 / 7}


def calculate_score(
    typical_move: float,
    move_stability: float,
    upside_pct: float,
    volume_spike: float,
    setup_type: str,
    distance_from_support: float,
    config: ScannerConfig,
) -> float:
    # Scored from the GATE, not from zero -- clearing min_typical_move_pct is
    # the price of admission, not an achievement. See the ANCHORS block.
    # typical_move is this stock's MEDIAN best gain over move_horizon_days.
    move_score = _span(
        typical_move, config.min_typical_move_pct, EXCELLENT_MOVE_PCT
    ) * PERSISTENCE_PTS
    # How often the upside actually dominated the downside over the horizon.
    # Anchored from 50% -- below that the stock fell further than it rose in
    # most windows, which deserves nothing, not partial credit.
    upside_score = _span(upside_pct, 50.0, EXCELLENT_UPSIDE_RATE) * EXPECTED_MOVE_PTS

    if move_stability < config.persistence_stability_threshold:
        stability_bonus = STABILITY_PTS
    elif move_stability < config.persistence_stability_threshold * 1.5:
        stability_bonus = STABILITY_PTS / 2
    else:
        stability_bonus = 0.0

    # From 1.0x, not from 0 -- trading your own average volume is the null
    # result, and used to collect a third of this budget for it.
    volume_bonus = _span(volume_spike, NORMAL_VOLUME_RATIO, EXCELLENT_VOLUME_RATIO) * VOLUME_PTS

    # Keyed on setup_type, NOT on action. reclaim is WATCH (see
    # get_entry_trigger) but it's still a real trigger that fired, and scoring
    # it 0 here would drop rows below min_score and delete them from the
    # report entirely -- losing visibility rather than just not buying.
    # Passive states (coiling, extended, below_trend, ...) still get 0.
    setup_bonus = SETUP_WEIGHTS.get(setup_type, 0.0) * SETUP_PTS

    support_penalty = min(
        max(distance_from_support - config.support_distance, 0) * 0.6, SUPPORT_PENALTY_PTS
    )

    score = (
        move_score + upside_score + stability_bonus
        + volume_bonus + setup_bonus
        - support_penalty
    )
    return round(max(min(score, 100), 0), 2)

# =========================================================
# CANDIDATE BUILDER
# =========================================================
def build_candidate(ticker: str, data: pd.DataFrame, config: ScannerConfig, run_date: str, run_timestamp: str) -> Optional[dict[str, Any]]:
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
    avg_volume20_prior = float(latest["avg_volume20_prior"])
    avg_traded_value20_cr = float(latest["avg_traded_value20_cr"])

    if (pd.isna(ma20) or pd.isna(recent_low) or pd.isna(high_20)
            or pd.isna(avg_volume20) or pd.isna(avg_traded_value20_cr)
            or pd.isna(avg_volume20_prior)
            or last_close <= 0 or avg_volume20 <= 0 or avg_volume20_prior <= 0):
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

    typical_move, sample_size, typical_drawdown, move_stability = run_move_profile(data, config)
    if sample_size < config.min_persistence_sample:
        return None
    # Gate on how far the stock TRAVELS, not on whether it cleared a bar.
    if typical_move < config.min_typical_move_pct:
        return None
    # ...and on the asymmetry. A stock that habitually gives back more than it
    # gains is not a candidate however far it moves; this is what stops the
    # "high move rate" filter from simply selecting the most violent stocks.
    if abs(typical_drawdown) > 0 and typical_move / abs(typical_drawdown) < config.min_move_ratio:
        return None
    persistence_rate = upside_rate(data, config)
    # move_stability is scoring-only (see calculate_score) -- not a hard reject.

    action, setup_type, reason = get_entry_trigger(data, config)
    setup_age_days = compute_setup_age(data, config, setup_type)

    risk = calculate_risk_reward(data, config, typical_move)
    if risk is None:
        return None
    # These are LIVE again. They were inert while rr_floor_stop capped every
    # stop at exactly target_pct/min_rr = 3.0% (384 of 384 rows had Risk_Pct
    # == 3.00). That cap is gone -- it existed to make risk fit a 3% target,
    # and a 3% target is no longer the goal -- so the stop is now purely
    # ATR/structural and risk_pct genuinely varies. max_risk_pct is what keeps
    # a wide-ATR stock from arriving with an unbounded stop.
    if risk["risk_pct"] > config.max_risk_pct:
        return None

    distance_from_support = ((last_close - recent_low) / last_close) * 100
    distance_from_ma20 = ((last_close - ma20) / last_close) * 100
    # Must match get_entry_trigger's denominator, or the Volume_Spike column
    # and the breakout gate that consumed it would disagree.
    volume_spike = float(latest["Volume"]) / avg_volume20_prior

    score = calculate_score(
        typical_move, move_stability, persistence_rate, volume_spike,
        setup_type, distance_from_support, config,
    )

    if score < config.min_score:
        return None

    return {
        "Ticker": ticker,
        "Bar_Date": pd.Timestamp(data.index[-1]).date().isoformat(),
        "Run_Date": run_date,
        "Run_Timestamp": run_timestamp,
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
        # Column names kept so the Dataform views keep compiling without a
        # manual workspace pull; the MEANINGS changed with the move profile.
        "Persistence_Rate": persistence_rate,      # % of windows upside beat downside
        "Persistence_Sample": sample_size,
        "Expected_Move": typical_move,             # median 30-session peak gain %
        "Persistence_Stability": move_stability,   # variability of that gain
        "Typical_Drawdown_Pct": typical_drawdown,  # median worst loss in the same windows
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
        "Max_Hold_Days": config.move_horizon_days,
    }
# =========================================================
# SCANNER
# =========================================================
def scan_ticker_data(ticker: str, data: pd.DataFrame, config: ScannerConfig, run_date: str, run_timestamp: str) -> Optional[dict[str, Any]]:
    if not has_enough_data(data, config):
        return None
    data = add_indicators(data, config)
    return build_candidate(ticker, data, config, run_date, run_timestamp)


def scan_tickers(tickers: list[str], config: ScannerConfig, run_date: str, run_timestamp: str) -> tuple[pd.DataFrame, list[str], int]:
    """Returns (results, failed_tickers, total_quality_candidates_before_top_n_cap).
    failed_tickers lets a systemic failure (bad batch, API change) surface
    instead of just looking like '0 candidates found'. total_quality_candidates
    lets a top_n truncation be visible instead of silently hiding real results."""
    yf = load_yfinance()
    results: list[dict[str, Any]] = []
    batch_failures: list[str] = []
    # ticker -> exception repr. Kept separate from batch_failures because the
    # two mean completely different things: a failed batch download is a
    # network/API problem, while an exception raised INSIDE scan_ticker_data
    # on data that downloaded fine is almost always a bug in this file.
    # has_enough_data already returns None (not an exception) for thin or
    # missing history, so these are not "no data" tickers.
    ticker_errors: dict[str, str] = {}
    batches = chunks(tickers, config.chunk_size)
    total = len(batches)
    attempted = 0

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
            batch_failures.extend(batch)
        else:
            for ticker in batch:
                attempted += 1
                try:
                    candidate = scan_ticker_data(ticker, get_ticker_frame(batch_data, ticker), config, run_date, run_timestamp)
                    if candidate:
                        results.append(candidate)
                except Exception as exc:
                    ticker_errors[ticker] = f"{type(exc).__name__}: {exc}"
                    if config.verbose:
                        print(f"\nSkipped {ticker}: {exc}")

        print(f"\rProgress: {int(bn/total*100)}% ({bn}/{total}) | {len(results)} found", end="", flush=True)
    print()

    # A near-total per-ticker failure rate is not a data condition, it is a
    # broken build, and it must not be allowed to read as "0 candidates found".
    # This is exactly how a field rename (support_distance_threshold ->
    # support_distance) took the scanner down: every one of 2,574 tickers
    # raised AttributeError inside calculate_score, the broad except below
    # swallowed all of them, and five consecutive scheduled runs reported zero
    # results with nothing worse than a warning in the log.
    if attempted and len(ticker_errors) / attempted > config.max_ticker_error_rate:
        counts: dict[str, int] = {}
        for msg in ticker_errors.values():
            counts[msg.split(":")[0]] = counts.get(msg.split(":")[0], 0) + 1
        top_kind = max(counts, key=counts.get)
        sample = next(m for m in ticker_errors.values() if m.startswith(top_kind))
        raise RuntimeError(
            f"{len(ticker_errors)}/{attempted} tickers raised inside scan_ticker_data "
            f"({len(ticker_errors) / attempted:.0%} > max_ticker_error_rate "
            f"{config.max_ticker_error_rate:.0%}) -- this is a code fault, not missing "
            f"data. Most common: {top_kind} x{counts[top_kind]}. Example: {sample}"
        )

    failures = batch_failures + list(ticker_errors)
    if failures:
        LOGGER.warning(
            "%d/%d tickers unusable (%d failed batch download, %d raised while scanning)",
            len(failures), len(tickers), len(batch_failures), len(ticker_errors),
        )

    if not results:
        return pd.DataFrame(), failures, 0

    df = pd.DataFrame(results)
    # Score is the ranker, so it must also be the sort key. It previously sat
    # FOURTH, behind Persistence_Rate and Persistence_Stability, which meant it
    # almost never affected the order at all: on a live 84-candidate run only
    # 6 rows shared an (Action, Persistence_Rate, Persistence_Stability) tuple
    # for Score to break, the displayed position correlated -0.366 with Score,
    # and the top 8 shown shared just 3 names with the top 8 by Score. The
    # report therefore ranked by one number while printing another beside it.
    # Persistence_Rate is still in the output as a column, and is now 50 of
    # Score's 100 points, so it keeps most of its influence -- explicitly
    # rather than accidentally.
    df["_action_rank"] = df["Action"].map({"BUY": 0, "WATCH": 1}).fillna(2)
    df = df.sort_values(
        ["_action_rank", "Score", "Persistence_Rate"],
        ascending=[True, False, False],
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
    bigquery.SchemaField("Run_Timestamp", "TIMESTAMP"),
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
    bigquery.SchemaField("Typical_Drawdown_Pct", "FLOAT64"),
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
    """Appends this run's rows, first deleting anything already stored under
    the same Run_Timestamp so a retried load can't double-write. Each scan is
    kept as its own observation rather than overwriting earlier runs of the
    same day -- see the de-duplication comment below for why that changed."""
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
    if "Run_Timestamp" in output.columns:
        output["Run_Timestamp"] = pd.to_datetime(output["Run_Timestamp"], utc=True)
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

    # De-duplication key.
    #
    # It used to be (Ticker, Bar_Date), which silently destroyed the intraday
    # history that Run_Timestamp was added to capture. With four-plus scheduled
    # runs a day, each run DELETEd the earlier runs' rows for any ticker it also
    # found, so for a given ticker on a given bar only the LAST run that
    # surfaced it survived. Confirmed against the live table: 0 (Ticker,
    # Bar_Date) pairs ever appeared twice, even on days with 6 retained
    # Run_Timestamps -- those 6 came from two different Bar_Dates (the 09:07
    # pre-open run stamps the previous session's bar), not from intraday
    # sequence. The result was that "appeared at 11:00, gone by 15:00" was
    # unanswerable from BigQuery, which is exactly the question a 1:30pm run
    # was added to answer.
    #
    # Now keyed on Run_Timestamp, and since a whole scan shares one timestamp
    # the predicate is a single equality rather than a chain of ORs per ticker.
    # Re-running produces a new timestamp and therefore a new row, which is
    # correct: that IS a second observation. The DELETE still protects the one
    # case that matters -- a partially-failed load retried with the SAME
    # timestamp cleans up after itself instead of double-writing.
    timestamp_field = field_map.get("Run_Timestamp", "Run_Timestamp")
    has_timestamp = (
        "Run_Timestamp" in output.columns
        and any(f.name == timestamp_field for f in table.schema)
    )
    if table_exists and has_timestamp:
        run_ts = pd.to_datetime(output["Run_Timestamp"]).max().to_pydatetime()
        client.query(
            f"DELETE FROM `{table_ref}` WHERE `{timestamp_field}` = @ts",
            job_config=bigquery.QueryJobConfig(query_parameters=[
                bigquery.ScalarQueryParameter("ts", "TIMESTAMP", run_ts)
            ]),
        ).result()
    elif table_exists:
        # Legacy table with no Run_Timestamp column: fall back to the old
        # (Ticker, date) key. This collapses intraday runs, but it is the only
        # key such a table can express, and the load below adds the column so
        # subsequent runs take the branch above.
        LOGGER.warning(
            "%s has no %s column; falling back to (%s, %s) de-duplication, which "
            "keeps only the last run per ticker per bar",
            table_ref, timestamp_field, ticker_field, dedupe_date_field,
        )
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


DATAFORM_LOCATION = "us-central1"
DATAFORM_REPOSITORY_ID = "sudarshan_repo"
DATAFORM_WORKSPACE_ID = "worker1"
DATAFORM_SERVICE_ACCOUNT = "347050126858-compute@developer.gserviceaccount.com"


DATAFORM_COMPILE_RETRIES = 3
DATAFORM_COMPILE_RETRY_DELAY_SEC = 8


def trigger_dataform_run(
    project_id: str,
    location: str = DATAFORM_LOCATION,
    repository_id: str = DATAFORM_REPOSITORY_ID,
    workspace_id: str = DATAFORM_WORKSPACE_ID,
    service_account: str = DATAFORM_SERVICE_ACCOUNT,
) -> str:
    """
    Triggers a Dataform workflow invocation (compile the repository's
    WORKSPACE state, then run it) via the REST API, so fact_stock_scan /
    the lifecycle views / vw_daily_digest pick up this run's fresh rows
    without a human manually starting an execution. No Dataform CLI is
    available in every environment this runs in, and a separate
    orchestration service would be overkill for two sequential API calls --
    this runs inside the same Cloud Run invocation that just wrote to
    BigQuery, using whatever credentials that service already has (needs
    roles/dataform.editor on the calling service account).

    Deliberately compiles from the "worker1" workspace (Dataform's own
    stored file state) rather than a live gitCommitish fetch. A live git
    fetch through Developer Connect's proxy showed intermittent "Remote
    repository ... could not be reached" failures in practice; compiling
    from the workspace means this trigger never touches GitHub at all, so
    that failure mode is gone entirely, not just retried around.

    KNOWN TRADE-OFF, confirmed directly (not assumed): workspace sync with
    GitHub is a MANUAL action (Pull in the Dataform UI / workspaces.pull),
    not automatic on push. That means every scheduled run here compiles
    whatever was last manually pulled into "worker1" -- NOT necessarily
    what's currently on GitHub. After pushing a change to any .sqlx file,
    it will not take effect in scheduled runs until someone manually pulls
    the workspace. This is a deliberate choice (reliability over
    always-fresh) -- do not "fix" it by adding an automatic pull here, since
    a pull is itself a live GitHub fetch and would reintroduce the exact
    flakiness this was built to avoid.

    Returns the workflow invocation resource name. Raises on failure --
    callers should catch this so a Dataform hiccup doesn't take down the
    scan/BQ-write response that already succeeded.
    """
    import google.auth
    import google.auth.transport.requests

    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    session = google.auth.transport.requests.AuthorizedSession(credentials)

    resource_base = f"projects/{project_id}/locations/{location}/repositories/{repository_id}"
    base = f"https://dataform.googleapis.com/v1/{resource_base}"
    workspace_name = f"{resource_base}/workspaces/{workspace_id}"

    compile_resp = None
    last_exc: Optional[Exception] = None
    for attempt in range(1, DATAFORM_COMPILE_RETRIES + 1):
        try:
            compile_resp = session.post(f"{base}/compilationResults", json={"workspace": workspace_name})
            compile_resp.raise_for_status()
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            compile_resp = None
            if attempt < DATAFORM_COMPILE_RETRIES:
                LOGGER.warning(
                    "Dataform compile attempt %d/%d failed (%s), retrying in %ds",
                    attempt, DATAFORM_COMPILE_RETRIES, exc, DATAFORM_COMPILE_RETRY_DELAY_SEC,
                )
                time.sleep(DATAFORM_COMPILE_RETRY_DELAY_SEC)
    if last_exc is not None:
        raise last_exc
    compilation_result_name = compile_resp.json()["name"]

    invoke_resp = session.post(
        f"{base}/workflowInvocations",
        json={
            "compilationResult": compilation_result_name,
            "invocationConfig": {"serviceAccount": service_account},
        },
    )
    invoke_resp.raise_for_status()
    return invoke_resp.json()["name"]


IST_OFFSET = dt.timedelta(hours=5, minutes=30)


def format_telegram_digest(candidates: pd.DataFrame, run_date: str) -> Optional[str]:
    """
    Builds the notification text: only same-day-fresh BUY signals
    (Setup_Age_Days == 1) -- the whole point of pushing this immediately is
    to close the "already high by the time I woke up" gap, so anything
    already a day or more stale doesn't belong in an urgent alert. Returns
    None if there's nothing fresh to send (caller should skip sending).

    Includes the actual run time (IST, not just the date) in the header --
    with 4 scheduled runs a day, the same ticker can legitimately appear in
    more than one, and "9:07 AM run" vs "3 PM run" for the same name is
    very different information: was this signal just caught, or has it
    already had hours to move since it first showed up.
    """
    if candidates.empty:
        return None

    fresh_buys = candidates[
        (candidates["Action"] == "BUY") & (candidates["Setup_Age_Days"] == 1)
    ].sort_values("Score", ascending=False)

    if fresh_buys.empty:
        return None

    run_time_ist = (dt.datetime.now(dt.timezone.utc) + IST_OFFSET).strftime("%Y-%m-%d %H:%M IST")

    # Plain text, deliberately -- see send_telegram_notification for why.
    # Kept to one line per ticker on purpose -- score/entry/SL/target are all
    # in BigQuery (vw_daily_digest) for whoever wants to dig in; this is the
    # 5-second phone read, not the full record.
    lines = [f"Swing scan -- {run_time_ist}", f"{len(fresh_buys)} fresh BUY signal(s):", ""]
    for _, r in fresh_buys.iterrows():
        # breakout is a BUY again (see get_entry_trigger), so the setup name
        # is doing real work here -- a breakout line means the move has
        # already happened (median +6.35% on the signal day), a
        # pullback_bounce means you're buying a dip. Same win rate as far as
        # anything measured, different entry price.
        lines.append(f"{r['Ticker']} ({r['Setup_Type']}) -- win rate {r['Persistence_Rate']:.0f}%")
    lines.append("")
    lines.append(f"Captured at {run_time_ist} -- Age==1 only, check current price before acting.")
    return "\n".join(lines)


def send_telegram_notification(text: str, bot_token: str, chat_id: str) -> None:
    """
    Sends `text` to a single Telegram chat via the Bot API. Raises on
    failure -- callers should catch this so a notification hiccup doesn't
    take down the scan/BQ-write/Dataform-trigger response that already
    succeeded (same pattern as trigger_dataform_run).

    Deliberately plain text, no parse_mode. Telegram's legacy Markdown
    parser treats a single "_" as an unclosed italic marker -- setup types
    like "pullback_bounce" and "deep_pullback" have exactly one underscore
    each, which broke every notification with a 400 "can't parse entities"
    error (confirmed: this is exactly what silently killed the first two
    real notifications this was tested against). Ticker/setup content here
    is data-driven and not fully predictable, so the fix is to stop relying
    on a fragile parser for it rather than trying to escape every value.
    """
    import requests

    resp = requests.post(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=15,
    )
    resp.raise_for_status()


def send_telegram_to_all(text: str, bot_token: str, chat_ids: list[str]) -> list[str]:
    """
    Sends to every chat_id, continuing past individual failures -- one bad
    ID (someone blocked the bot, a typo'd ID) shouldn't silently drop the
    notification to everyone else. Returns the chat_ids that failed
    (empty if all succeeded), so the caller can log which ones need
    attention rather than a single opaque exception.
    """
    failures = []
    for cid in chat_ids:
        try:
            send_telegram_notification(text, bot_token, cid)
        except Exception as exc:
            failures.append(cid)
            LOGGER.warning("Telegram send failed for chat_id %s: %s", cid, exc)
    return failures


# =========================================================
# ARGS
# =========================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Volatility persistence swing scanner - buys volatile setups and "
                    "backtests never exiting below a minimum profit floor, extending "
                    "while the move keeps closing higher."
    )
    p.add_argument("--tickers", nargs="+", default=None)
    p.add_argument("--symbols-source", default=NSE_EQUITY_LIST_URL)
    p.add_argument("--limit", default=0, type=int)
    p.add_argument("--only-buy", action="store_true")

    p.add_argument("--min-avg-volatility", default=3.5, type=float)
    p.add_argument("--range-event-threshold", default=3.5, type=float)
    p.add_argument("--min-volatile-days", default=30, type=int)

    p.add_argument("--min-typical-move", default=ScannerConfig.min_typical_move_pct, type=float,
                    help="Minimum typical move %% over move_horizon_days -- see run_move_profile")
    p.add_argument("--move-horizon-days", default=ScannerConfig.move_horizon_days, type=int,
                    help="Forward window the move profile measures over")
    p.add_argument("--min-persistence-sample", default=ScannerConfig.min_persistence_sample, type=int)

    p.add_argument("--min-price", default=ScannerConfig.min_price, type=float)
    p.add_argument("--max-price", default=ScannerConfig.max_price, type=float)
    p.add_argument("--min-traded-value-cr", default=ScannerConfig.min_avg_traded_value_cr, type=float)

    p.add_argument("--target-pct", default=ScannerConfig.target_pct, type=float,
                    help="Minimum profit floor %% above entry -- never exit below this")
    p.add_argument("--max-extension-days", default=ScannerConfig.max_extension_days, type=int,
                    help="After the floor is hit, keep holding this many more sessions "
                         "while still closing higher; exit on the first non-higher close")
    p.add_argument("--round-trip-cost-pct", default=ScannerConfig.round_trip_cost_pct, type=float,
                    help="Round-trip transaction cost %% charged to every backtest trial. "
                         "Raise it if you trade smaller than ~Rs.25,000 a position, since the "
                         "flat DP charge is a bigger share of a smaller trade")
    p.add_argument("--min-rr", default=ScannerConfig.min_rr, type=float)
    p.add_argument("--max-risk-pct", default=ScannerConfig.max_risk_pct, type=float)
    p.add_argument("--max-hold-days", default=ScannerConfig.max_hold_days, type=int)
    p.add_argument("--min-score", default=ScannerConfig.min_score, type=float,
                    help="Hard quality gate (0-100); candidates scoring below this are rejected outright")

    p.add_argument("--top-n", default=ScannerConfig.top_n, type=int)
    p.add_argument("--output", default="", help="CSV output path")
    p.add_argument("--no-bq", action="store_true", help="Skip writing results to BigQuery")
    p.add_argument("--no-dataform-trigger", action="store_true",
                    help="Skip triggering a Dataform run after a successful BigQuery write")
    p.add_argument("--no-telegram", action="store_true",
                    help="Skip sending a Telegram notification for fresh BUY signals")
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

    IMPORTANT: when invoked over HTTP, sys.argv still holds whatever flags
    the CONTAINER was launched with (e.g. functions-framework's own
    "--target=main --source=swings.py --port=8080"), not anything about
    this request -- parse_args() must never run in that path. It used to,
    and argparse's prefix-matching silently mapped "--target=main" onto this
    file's own --target-pct flag, then crashed trying to parse "main" as a
    float -- which failed every single HTTP request. Config here comes from
    the request's JSON body (if any) or plain defaults instead.
    """
    if request is not None:
        try:
            body = request.get_json(silent=True) or {}
        except Exception:
            body = {}

        config = ScannerConfig(
            min_avg_volatility=float(body.get("min_avg_volatility", ScannerConfig.min_avg_volatility)),
            range_event_threshold=float(body.get("range_event_threshold", ScannerConfig.range_event_threshold)),
            min_volatile_days=int(body.get("min_volatile_days", ScannerConfig.min_volatile_days)),
            min_typical_move_pct=float(body.get("min_typical_move", ScannerConfig.min_typical_move_pct)),
            move_horizon_days=int(body.get("move_horizon_days", ScannerConfig.move_horizon_days)),
            min_persistence_sample=int(body.get("min_persistence_sample", ScannerConfig.min_persistence_sample)),
            min_price=float(body.get("min_price", ScannerConfig.min_price)),
            max_price=float(body.get("max_price", ScannerConfig.max_price)),
            min_avg_traded_value_cr=float(body.get("min_traded_value_cr", ScannerConfig.min_avg_traded_value_cr)),
            target_pct=float(body.get("target_pct", ScannerConfig.target_pct)),
            max_extension_days=int(body.get("max_extension_days", ScannerConfig.max_extension_days)),
            round_trip_cost_pct=float(body.get("round_trip_cost_pct", ScannerConfig.round_trip_cost_pct)),
            min_rr=float(body.get("min_rr", ScannerConfig.min_rr)),
            max_risk_pct=float(body.get("max_risk_pct", ScannerConfig.max_risk_pct)),
            max_hold_days=int(body.get("max_hold_days", ScannerConfig.max_hold_days)),
            min_score=float(body.get("min_score", ScannerConfig.min_score)),
            top_n=int(body.get("top_n", ScannerConfig.top_n)),
            verbose=bool(body.get("verbose", False)),
        )
        raw_tickers = body.get("tickers")
        tickers = parse_tickers(raw_tickers) if raw_tickers else load_nse_tickers(
            body.get("symbols_source", NSE_EQUITY_LIST_URL), body.get("project_id"))
        limit = int(body.get("limit", 0))
        if limit > 0:
            tickers = tickers[:limit]

        only_buy = bool(body.get("only_buy", False))
        output_path = ""
        no_bq = bool(body.get("no_bq", False))
        no_dataform = bool(body.get("no_dataform_trigger", False))
        no_telegram = bool(body.get("no_telegram", False))
        project_id = body.get("project_id")
        dataset_id = body.get("dataset_id", "data_options")
        table_id = body.get("table_id", DEFAULT_BQ_TABLE_ID)
    else:
        args = parse_args()
        config = ScannerConfig(
            min_avg_volatility=args.min_avg_volatility,
            range_event_threshold=args.range_event_threshold,
            min_volatile_days=args.min_volatile_days,
            min_typical_move_pct=args.min_typical_move,
            move_horizon_days=args.move_horizon_days,
            min_persistence_sample=args.min_persistence_sample,
            min_price=args.min_price,
            max_price=args.max_price,
            min_avg_traded_value_cr=args.min_traded_value_cr,
            target_pct=args.target_pct,
            max_extension_days=args.max_extension_days,
            round_trip_cost_pct=args.round_trip_cost_pct,
            min_rr=args.min_rr,
            max_risk_pct=args.max_risk_pct,
            max_hold_days=args.max_hold_days,
            min_score=args.min_score,
            top_n=args.top_n,
            verbose=args.verbose,
        )

        tickers = parse_tickers(args.tickers) if args.tickers else load_nse_tickers(args.symbols_source, args.project_id)
        if args.limit > 0:
            tickers = tickers[:args.limit]

        only_buy = args.only_buy
        output_path = args.output
        no_bq = args.no_bq
        no_dataform = args.no_dataform_trigger
        no_telegram = args.no_telegram
        project_id = args.project_id
        dataset_id = args.dataset_id
        table_id = args.table_id

    print(f"Universe: Rs.{config.min_price}-{config.max_price} | Liquidity >= {config.min_avg_traded_value_cr} Cr")
    print(f"Volatility: 6-month avg >= {config.min_avg_volatility}% | median >= {config.min_median_volatility}% | "
          f"{config.min_volatile_days}+ volatile days")
    print(f"Stop: ATR/structure only, risk capped at {config.max_risk_pct}% "
          f"(no fixed profit floor -- exits are a human decision)")
    print(f"Move profile: typical gain >= {config.min_typical_move_pct}% over "
          f"{config.move_horizon_days} sessions, gain/pain >= {config.min_move_ratio} "
          f"({config.min_persistence_sample}+ eligible days in last {config.persistence_lookback})")
    print(f"Scanning {len(tickers)} tickers...")
    print("-" * 80)

    now_utc = dt.datetime.now(dt.timezone.utc)
    run_date = now_utc.date().isoformat()
    run_timestamp = now_utc.isoformat()
    candidates, failures, total_quality = scan_tickers(tickers, config, run_date, run_timestamp)

    if failures:
        print(f"\n{len(failures)} tickers failed or had no usable data.")
        if config.verbose:
            print(", ".join(failures[:30]))

    if candidates.empty:
        message = "No candidates matched. Try:"
        print(f"\n{message}")
        # Hints must be LOWER than the current defaults to actually relax
        # anything -- the old block suggested --min-persistence 55 and
        # --min-score 55, both ABOVE the defaults of that time, which would
        # have tightened the scan while claiming to loosen it. --min-rr is
        # not listed: raising it tightens the stop and rr_floor_stop already
        # guarantees the ratio, so it can't surface more candidates.
        #
        # --min-score is also not listed: at 5.0 it is already near-zero on a
        # 0-100 scale (see the field comment), so there is no lower value left
        # that means anything as a relaxation hint. min_persistence_rate is
        # what actually gates candidates now.
        print(f"  --min-typical-move 5         (lower move bar, now {config.min_typical_move_pct:.0f})")
        print(f"  --min-avg-volatility 3.0     (lower volatility bar, now {config.min_avg_volatility})")
        print(f"  --min-persistence-sample 30  (allow smaller sample, now {config.min_persistence_sample})")
        print(f"  --max-price 2000             (widen the universe, now {config.max_price:.0f})")
        return (message, 200) if request is not None else None

    display = candidates
    if only_buy:
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

    if output_path:
        candidates.to_csv(output_path, index=False)
        print(f"\nSaved {len(candidates)} rows to {output_path}")

    if total_quality > len(candidates):
        print(f"\nNote: {total_quality} total quality candidates found, only top {config.top_n} shown/saved. "
              f"Raise --top-n to see the rest.")
    else:
        print(f"\nTotal quality candidates: {total_quality}")

    if not no_telegram:
        try:
            bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
            chat_id_raw = os.getenv("TELEGRAM_CHAT_ID")
            if not bot_token or not chat_id_raw:
                print("\nTelegram notification skipped: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set.")
            else:
                digest_text = format_telegram_digest(candidates, run_date)
                if digest_text is None:
                    print("\nNo fresh (Age==1) BUY signals -- Telegram notification skipped.")
                else:
                    chat_ids = [c.strip() for c in chat_id_raw.split(",") if c.strip()]
                    # NOT `failures` -- that name already holds the failed
                    # TICKER list from scan_tickers above, and reusing it here
                    # silently destroyed it (it's still printed further up, but
                    # anything added later would have read Telegram chat ids).
                    telegram_failures = send_telegram_to_all(digest_text, bot_token, chat_ids)
                    if telegram_failures:
                        print(f"\nSent Telegram notification to {len(chat_ids) - len(telegram_failures)}/{len(chat_ids)} "
                              f"recipient(s); failed: {telegram_failures}")
                    else:
                        print(f"\nSent Telegram notification to {len(chat_ids)} recipient(s).")
        except Exception as exc:
            print(f"\nTelegram notification skipped/failed: {exc}")
            print("Use --no-telegram to suppress this.")
    if not no_bq:
        try:
            resolved_project_id = _resolve_project_id(project_id)
            n = write_to_bigquery(candidates, resolved_project_id, dataset_id, table_id)
            print(f"Wrote {n} rows to {resolved_project_id}.{dataset_id}.{table_id}")

            if not no_dataform:
                try:
                    invocation_name = trigger_dataform_run(resolved_project_id)
                    print(f"Triggered Dataform run: {invocation_name}")
                except Exception as exc:
                    print(f"\nDataform trigger skipped/failed: {exc}")
                    print("Use --no-dataform-trigger to suppress this, or check the calling "
                          "service account has roles/dataform.editor.")
        except Exception as exc:
            print(f"\nBigQuery write skipped/failed: {exc}")
            print("Use --no-bq to suppress this, or --project-id / set GCP_PROJECT to fix it.")

    if request is not None:
        return (f"Scanner completed: {len(candidates)} candidates processed.", 200)
    return None

if __name__ == "__main__":
    main()
