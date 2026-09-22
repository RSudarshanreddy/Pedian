"""
SWING SCANNER -- buy the dip, in a stock that keeps coming back

Rebuilt 2026-09-23. The previous contents of this file ranked stocks by how
hard they were CURRENTLY moving, which measurement showed to be a momentum
screener wearing a swing name (Expected_Move correlates +0.701 with trailing
3-month return). That logic now lives in momentum.py, unchanged and still
deployed. This file is the strategy the name always claimed.

THE RULE. Among stocks that qualify on volatility and liquidity, wait until one
has fallen at least min_dip_pct from its 20-session high while still holding
above its 50-day average, then rank what is left by the stock's OWN history of
falling and recovering.

MEASURED, 160,308 observations, 2 years, entry at next open, net of costs, no
lookahead, top 3 per day held 30 sessions:
    rule                                    dates trades  ret30   win  halves
    momentum (the old contents, same pool)    136    363  +7.70  57.3  6.07/8.98
    dip>=12 + above MA50, rank bounce         75    157 +10.06  65.0  9.61/10.62
    dip>=12 + above MA50, rank Expected_Move  75    157  +9.06  62.4 10.87/6.87
    dip>=12 + above MA50, rank deepest dip    75    157  +8.76  60.5 11.40/5.95
    dip>=8  + above MA50, rank bounce        115    280  +6.99  60.4  8.96/4.40
    dip>=12, NO trend filter                 126    308  +6.09  58.8  7.23/4.53
All three components earn their place: the dip, the trend filter (worth 4
points) and the bounce-history ranker (worth 1 point over ranking by
Expected_Move). Time halves are 9.10 / 10.76, the steadiest result measured.

WHY BUYING THE DIP WORKS HERE. Forward return by how far the stock has fallen
from its 20-session high, across the qualified pool:
    -20..-12%  n=1667  +7.61%  win 63.3%      -8..-4%   n=2784  +3.59%
    -12..-8%   n=1809  +5.14%  win 57.9%      at high   n=2825  +3.93%
Roughly double the return for buying a 12-20% pullback over buying the high.

PROVISIONAL -- read this before trusting the numbers above. The winning cell is
also the smallest: 157 trades on overlapping 30-session windows is perhaps 5-6
independent observations, and the shallower dip>=8 variant is WORSE, so the
effect lives entirely in the thinnest slice. That is the shape of an overfit
even though both splits are clean. Treat this as a hypothesis under test until
signal_outcomes carries real forward data (~December), and prefer momentum.py
for anything that has to work.

It fires on roughly every other session (75 of 164 dates), by design -- a dip
that deep is not an everyday event.

- Does NOT model an exit. Exits are a human decision.
- Ranked by Bounce_Median. Action/Setup are recorded but do not order the list.
- Written to BigQuery table `swings`, keyed on Run_Timestamp.
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

IST_OFFSET = dt.timedelta(hours=5, minutes=30)
# NSE continuous trading ends 15:30 IST. Yahoo keeps revising the day's bar for
# a while after that (closing-auction prints, late corrections), so a bar dated
# today is only trusted once this much of the day has passed.
SESSION_SETTLED_IST_HOUR = 16

# How many tickers market_session_calendar samples to measure Yahoo's coverage.
# Taken as an even stride through the universe, not at random, so a rerun of the
# same universe derives the same calendar and an odd result is reproducible.
SESSION_SAMPLE_SIZE = 150

# =========================================================
# CONFIG
# =========================================================
@dataclass(frozen=True)
class ScannerConfig:
    period: str = "2y"
    interval: str = "1d"
    min_days: int = 220

    # --- Data integrity ---
    # Yahoo's NSE prices are correct (0.0000% diff vs Kite over 33 bars) but its
    # SESSIONS are not: it invents flat zero-volume bars on holidays (2.96% of
    # all bars) and omits real sessions for most of the market (73% of tickers
    # missing 2026-09-18). A missing session is a hard reject, not a penalty --
    # every day-over-day term silently spans the gap, turning AEROFLEX's true
    # +2.67% close move into +6.19% and fabricating a breakout.
    # See market_session_calendar / align_to_session.
    require_current_bar: bool = True
    # Share of the probe sample needing a bar before a date counts as a session.
    # Not delicate: coverage is bimodal (100% or ~30%), so anything 60-99% picks
    # the same session.
    min_session_coverage: float = 0.80
    # Recent sessions that must be hole-free. 20 covers every rolling window
    # add_indicators computes. Nearly free: of 94 tickers with a current bar,
    # requiring 2 clean sessions rejected 36 and requiring 20 rejected 37.
    contiguous_sessions_required: int = 20
    # False => a partial bar's volume_spike is NaN (UNKNOWN), not a small number.
    # Volume is the one measure compared against a FULL-day baseline, so on a
    # live bar it reads as a clock, not a signal: median spike 0.21 at 11:00
    # rising to 0.95 after the close.
    trust_partial_volume: bool = False

    # --- Universe ---
    # Affordability preference, NOT a predictor: price correlates -0.016 to
    # -0.031 with forward return. Swept on the full stack, 15-session hold:
    #   0 -> +4.03%  250 -> +4.44%  350 -> +4.63%  500 -> +3.77%
    # 250-350 is a flat plateau; the curve falls away above 400. What actually
    # removes collapse-prone names is min_drawdown_pct / max_spike_ratio, which
    # cut the worst 1-year drawdown among survivors from -74% to -48%.
    min_price: float = 250.0
    # Ceiling 1400 -> 1600 on 2026-09-23, to bring AEGISLOG (Rs.1,486) into the
    # universe. Measured cost: none either way. Swept on the swing rule, top
    # 3/day, 30-session hold:
    #   1400 +10.14% (halves 10.85/9.29)   1800 +9.92% (6.08/14.01)
    #   2200 +10.26% (12.20/8.45)          none  +9.42% (7.85/10.70)
    # All inside noise, and 1400 had the best balance -- so this is a
    # preference call, not a performance one, and it is recorded as such.
    max_price: float = 1600.0
    # NOT raised. Measured the wrong way round: tightening to 20/30/50/100 Cr
    # moved returns +2.14 -> 1.62 -> 1.22 -> 1.34 -> 0.89, and corr(traded
    # value, forward return) is -0.042. This exists to guarantee you can get in
    # and out, nothing more.
    min_avg_traded_value_cr: float = 10.0

    # --- Volatility identity ---
    # 180 -> 60 on 2026-09-22, to stop a live move being averaged against six
    # quiet months. ANTELOPUS read typical move 5.6 at 180 days and 31.8 at 60
    # at the start of its +56.2% leg. Swept, 15-session hold, net, no lookahead:
    #   180d +1.92% (halves 3.26/0.69)   90d +1.59%
    #    60d +2.25% (halves 2.83/1.69)   50d +2.32%
    # Shorter wins on every column AND the ticker halves converge.
    # COST, measured after the fact: this made Expected_Move substantially a
    # MOMENTUM measure -- corr with trailing 3-month return went +0.152 -> +0.701.
    # Momentum paid here (+6.4% for the hottest quintile vs +2.3% for the
    # coldest, and it held through market declines), but that is 8 months of one
    # market. Does NOT rescue AEGISLOG, which was genuinely dormant beforehand.
    volatility_lookback: int = 60
    range_event_threshold: float = 3.5
    close_move_threshold: float = 2.5
    min_avg_volatility: float = 3.5
    min_median_volatility: float = 2.5
    # Scaled 30 -> 12 WITH the lookback to keep behaviour fixed rather than the
    # number: 12 is 0.20*60, so the ratio below still binds, as it did at 180.
    min_volatile_days: int = 12
    min_volatility_ratio: float = 0.20
    # Deliberately NOT tied to volatility_lookback. measure_spike_ratio is a
    # p95/median, and a p95 of 60 points is essentially the 3rd largest value.
    # The 4.5 threshold was calibrated on 180.
    spike_lookback: int = 180

    # --- Move profile ---
    # See volatility_lookback for the sweep behind 60.
    persistence_lookback: int = 60
    # Arithmetic, not preference: a 60-day lookback yields at most 60 windows,
    # so the old floor of 60 would reject nearly everything. 20 is the same
    # one-third proportion as the old 60-of-180.
    min_persistence_sample: int = 20
    # ~6 weeks, the timescale the winning trades actually played out on.
    move_horizon_days: int = 30
    # A QUALITY FLOOR under the bounce ranker, not the membership gate it is in
    # momentum.py (14.0). Bounce_Median alone will happily rank a stock that
    # recovers 28% from dips but travels only 7% over 30 sessions -- BALUFORGE
    # on the first live run, Score 31.9, RR 0.8, ranked 3rd.
    #
    # Measured on the dip cohort, top 3/day, 30-session hold:
    #   no floor   +10.14%  win 65.4%  halves 10.85 / 9.29
    #   >= 10      +10.41%  win 67.7%  halves 10.18 / 10.76   <- set here
    #   >= 14       +8.90%  win 66.1%  halves  8.21 / 9.49
    # 10 is better on return, win rate AND balance; 14 (momentum.py's value)
    # overshoots and costs 1.5 points. The gate that decides membership here is
    # the dip, not this.
    min_typical_move_pct: float = 10.0

    # --- The swing rule ---
    # How far below its 20-session high the stock must have fallen. Measured
    # across the qualified pool, forward 30-session return by dip depth:
    #   -20..-12% +7.61% (win 63.3%)   -8..-4%  +3.59%
    #   -12..-8%  +5.14% (win 57.9%)   at high  +3.93%
    # 12 is where the step happens. At 8 the top-3 rule returns +6.99% against
    # +10.06% at 12 -- but note the deeper cell is also the thinner one, which
    # is the overfitting risk named in the module docstring.
    min_dip_pct: float = 12.0
    # The dip must be a pullback in an uptrend, not a falling knife. Dropping
    # this costs 4 points: +10.06% with it, +6.09% without.
    trend_ma_period: int = 50
    # Reversal threshold for the zigzag that counts completed trough->peak legs.
    # 5% is large enough that ordinary daily noise in a stock with 4-6% average
    # range does not register as a swing.
    zigzag_pct: float = 5.0
    # Minimum completed bounce legs before a stock's bounce history means
    # anything. A single leg is an anecdote.
    min_bounce_legs: int = 1
    # Scoring only. Gain/pain and tail drawdown were gates briefly and it was a
    # mistake: a 3.0 ratio floor excluded ANTELOPUS and a -12% tail floor
    # excluded SHILPAMED -- two of the four trades this exists to find.
    move_stability_threshold: float = 4.0

    # --- Character: what KIND of volatility ---
    # p95/median daily move. High means the stock sits still then gaps, and you
    # cannot enter a gap. Swept: no filter +3.86%, <=4.5 +4.19%, <=4.0 +4.63%,
    # <=3.5 +4.86%. Settled at 4.5 after 3.5 rejected ANTELOPUS (3.55) and 4.0
    # rejected DYCL (4.18) -- a cutoff that must move every time it meets a
    # known-good stock is fitted to noise. 4.5 sits above the pool median 3.72,
    # so it trims the gap-driven tail rather than the middle.
    max_spike_ratio: float = 4.5
    # Backward-looking, not a forecast: a stock that halved has demonstrated it
    # can halve. Worth 0.32pp, and it improves median and win rate more than
    # mean -- the right shape for a safety filter.
    min_drawdown_pct: float = -50.0
    drawdown_lookback: int = 250

    # --- Entry timing ---
    breakout_lookback: int = 20
    support_window: int = 20
    support_distance: float = 8.0
    breakout_volume_mult: float = 2.3
    pullback_min: float = 2.0
    pullback_max: float = 8.0

    # --- Risk / trade levels ---
    atr_period: int = 14
    stop_atr_mult: float = 1.5
    recent_low_buffer: float = 0.975   # stop can't be looser than this * recent_low
    # A backstop against an absurd stop, NOT a risk preference, and currently
    # INERT: across 1,540 qualifying observations risk_pct runs p50 6.9%,
    # p99 11.5%, max 13.6%, so 15% removes 0 rows. It must stay loose because
    # stop distance and volatility are the same variable -- at 8.0 it rejected 7
    # of the 8 most volatile stocks in the universe. Per-trade risk is
    # controlled by POSITION SIZE; Risk_Pct is in the output for that.
    max_risk_pct: float = 15.0
    # Zerodha NSE delivery, built up not guessed: STT 0.2000 + exchange 0.0059
    # + SEBI 0.0002 + stamp 0.0150 + GST 0.0011 = 0.2222%, plus a FLAT ~Rs.15.93
    # DP charge per scrip per sell day. 0.25% therefore assumes ~Rs.25,000 a
    # position; trade smaller and this is too kind. Corroborated live: charges
    # over Jul-Sep roughly equalled the entire realized P&L.
    round_trip_cost_pct: float = 0.25
    # Near non-binding on purpose (99% of candidates clear it):
    # min_typical_move_pct decides membership, and a rescale should change the
    # ORDER of the list, not which stocks appear. Survivor counts if you want
    # "fewer but better": 10 keeps 95%, 20 keeps 75%, 25 keeps 50%, 30 keeps 40%.
    # NOT comparable to any pre-rescale value.
    min_score: float = 5.0

    # --- Processing ---
    chunk_size: int = 100
    max_batch_retries: int = 2
    # Share of tickers that may raise inside scan_ticker_data before scan_tickers
    # treats it as a code fault. Thin history returns None rather than raising,
    # so a healthy run sits near zero.
    max_ticker_error_rate: float = 0.25
    # STORAGE cap. Deliberately high: the stored history feeds the signal-history
    # test, which needs the LOW scores too.
    top_n: int = 200
    # DISPLAY cap -- attention knob, not a filter. Caps by RANK so storage is
    # unaffected and cheap-but-good names are not deleted for being cheap.
    display_top_n: int = 50
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.min_typical_move_pct < 0:
            raise ValueError(
                f"min_typical_move_pct must be >= 0 (got {self.min_typical_move_pct}). "
                "0 is valid here and means 'record it, do not gate on it' -- "
                "momentum.py is the file that gates on it."
            )
        if self.min_dip_pct <= 0:
            raise ValueError(f"min_dip_pct must be > 0 (got {self.min_dip_pct})")
        if self.move_horizon_days < 5:
            raise ValueError(
                f"move_horizon_days must be >= 5 (got {self.move_horizon_days}) -- "
                "shorter windows cannot contain a move worth holding for"
            )
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
    data = data.dropna(subset=[c for c in REQUIRED_COLUMNS if c in data.columns])

    # Drop Yahoo's fabricated holiday bars. Zero volume means no trade took
    # place, so this is not a session and must not be averaged in as one.
    #
    # Every error from keeping them points the same way, because a zero-volume
    # day is maximally calm (range 0, close_move 0) AND a zero sample in the
    # 20-day mean. Measured over 19 tickers / 2 years: ATR understated 5.7%,
    # traded value 6.1% (41.9% worst), avg_volatility 1.8%, and Volume_Spike
    # INFLATED -- INDIAGLYCO read 6.39x against a true 4.39x.
    if "Volume" in data.columns:
        data = data[data["Volume"] > 0]
    return data

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


def _last_bar_date(data: pd.DataFrame) -> Optional[dt.date]:
    if data.empty:
        return None
    return pd.Timestamp(data.index[-1]).date()


def market_session_calendar(yf: Any, tickers: list[str],
                            config: ScannerConfig) -> Optional[list[dt.date]]:
    """The run's NSE session calendar, ending at the newest WIDELY-COVERED session.

    Yahoo does not say which sessions it is missing, so the run works it out by
    measuring itself. It samples the universe and keeps the dates that at least
    min_session_coverage of the sample has a bar for.

    The key word is *widely*. An earlier version asked ten megacaps for their
    newest bar, on the theory that if any real stock traded that day it was a
    session. That is true and useless: Yahoo's NSE feed runs 1-2 sessions BEHIND
    for most of the market, so the megacaps' newest bar is a date almost nothing
    else has yet, and requiring it rejected 84% of the universe every run.

    Measured 2026-09-19, 265 tickers, per-date coverage:
        2026-09-11  100%     2026-09-16  100%
        2026-09-15  100%     2026-09-17   38%     2026-09-18   30%
    Coverage is bimodal -- a date is either everywhere or barely anywhere --
    so any threshold between 60% and 99% picks the same reference session and
    the exact value is not delicate.

    Rejecting the thin dates costs nothing and buys everything: with 2026-09-16
    as the reference, 100% of tickers were current and 99% also cleared the
    20-session contiguity check, against 16% under the megacap rule.

    The trade is honest and worth naming: signals are computed on the newest
    session Yahoo has for the WHOLE market, which during a lag is 1-2 sessions
    behind the live market. Bar_Date records which session that was, so nothing
    is misrepresented -- and a uniformly 2-day-old scan is far more useful than
    a mix of fresh and stale rows that cannot be told apart. For a 30-session
    move profile a two-day lag is noise; for the entry trigger it is real, which
    is a reason to read Bar_Date rather than to pretend the bar is today's.

    Returns None if the probe fails, so the caller falls back to the old
    permissive behaviour rather than rejecting everything over a network blip.
    """
    if not tickers:
        return None
    stride = max(1, len(tickers) // SESSION_SAMPLE_SIZE)
    sample = tickers[::stride][:SESSION_SAMPLE_SIZE]
    try:
        probe = download_batch(yf, sample, ScannerConfig(period="3mo"))
    except Exception as exc:
        LOGGER.warning("Session probe download failed (%s) -- bar-recency checks disabled", exc)
        return None

    coverage: dict[dt.date, int] = {}
    usable = 0
    for ticker in sample:
        frame = get_ticker_frame(probe, ticker)
        if frame.empty:
            continue
        usable += 1
        for stamp in frame.index:
            day = pd.Timestamp(stamp).date()
            coverage[day] = coverage.get(day, 0) + 1

    if not usable:
        LOGGER.warning("Session probe returned no usable bars -- bar-recency checks disabled")
        return None

    floor = config.min_session_coverage * usable
    calendar = sorted(day for day, seen in coverage.items() if seen >= floor)

    # Sessions that are unarguably REAL -- a meaningful minority of the market
    # traded -- but that Yahoo lacks for too much of it to treat as a session.
    #
    # These are the blind spot, and it is worth stating rather than hiding.
    # 2026-09-17 was one: Kite has it in full, and Yahoo had it for 42%. It is
    # therefore excluded from the calendar, which means a ticker missing it is
    # NOT flagged as gapped -- there is no majority to compare against. Their
    # one-day-lookback figures silently span it: AEROFLEX's 2026-09-18 close
    # move reads +6.19% instead of +2.67%, and prev["High"] becomes 495.00
    # instead of 512.10, which can fabricate a pullback_bounce confirmation.
    #
    # Including it instead would reject the ~58% of tickers that lack it, which
    # is worse than the harm. A feed-wide hole is not a per-ticker defect and
    # cannot be filtered like one. So: name it, and let the reader discount the
    # one-day terms for that run.
    if calendar:
        recent = calendar[-config.contiguous_sessions_required:]
        span_start = recent[0] if recent else calendar[0]
        thin = sorted(
            day for day, seen in coverage.items()
            if day > span_start and seen < floor and seen >= 0.05 * usable
        )
        if thin:
            LOGGER.warning(
                "Real sessions Yahoo has for only part of the market, excluded from the "
                "calendar: %s. Tickers missing these are NOT flagged as gapped, so their "
                "close-move and prior-high terms span the hole.",
                ", ".join(f"{d} ({coverage[d] / usable:.0%})" for d in thin),
            )
    if not calendar:
        LOGGER.warning(
            "No session reached %.0f%% coverage across %d probe tickers -- "
            "bar-recency checks disabled", config.min_session_coverage * 100, usable,
        )
        return None

    return calendar


def is_partial_session(session: Optional[dt.date],
                       now_ist: Optional[dt.datetime] = None) -> bool:
    """Is this session's bar still being written?

    True for a bar dated today before the session has settled. Such a bar holds
    a real, current price and a real, current trend -- but only PART of a day's
    volume and part of its range, so anything that compares it against a
    full-day baseline is comparing unlike things.

    Note what this does NOT do: it does not discard the bar. Dropping it was
    tried and was wrong. On market days Yahoo has today's bar for the entire
    universe on a single Bar_Date -- verified across the 11:00, 12:00, 13:00 and
    15:00 runs on 2026-09-16/17/18 -- while the PREVIOUS session is still
    missing for most of it. So discarding today's bar does not step back one
    session, it steps back two, and throws away the only current data there is.
    """
    if session is None:
        return False
    now = now_ist or (dt.datetime.now(dt.timezone.utc) + IST_OFFSET).replace(tzinfo=None)
    return session == now.date() and now.hour < SESSION_SETTLED_IST_HOUR


def align_to_session(data: pd.DataFrame, calendar: Optional[list[dt.date]],
                     contiguous_required: int = 0) -> tuple[pd.DataFrame, str]:
    """Trim a ticker to the run's calendar and report whether its recent bars are sound.

    Three jobs, because they are all the same question -- does this ticker's
    recent history match the sessions the market actually held.

    1. Bars AFTER the newest complete session are the live partial bar, dropped.
    2. What remains must END on that session, or this ticker is running behind
       the market and its "latest" reading is really days old.
    3. The last `contiguous_required` sessions must have no holes. This is the
       check that catches AEROFLEX on 2026-09-18: its last bar IS current, so
       (2) passes, yet 2026-09-17 is missing from the middle of its history and
       every rolling window silently closes over the gap.

    `contiguous_required` is measured in SESSIONS, not calendar days, so a
    holiday or weekend is never mistaken for a hole -- the probe basket did not
    trade on those days either, so they are not in the calendar to begin with.

    Returns (trimmed, state), state one of "current", "stale", "gapped", "empty".
    A None calendar disables all three checks and passes everything through.
    """
    if not calendar:
        return data, "current"
    if data.empty:
        return data, "empty"

    reference_session = calendar[-1]
    dates = pd.Index([pd.Timestamp(i).date() for i in data.index])
    trimmed = data[dates <= reference_session]
    if trimmed.empty:
        return trimmed, "empty"
    if _last_bar_date(trimmed) != reference_session:
        return trimmed, "stale"

    if contiguous_required > 0:
        have = {pd.Timestamp(i).date() for i in trimmed.index}
        if set(calendar[-contiguous_required:]) - have:
            return trimmed, "gapped"
    return trimmed, "current"
# =========================================================
# INDICATORS
# =========================================================
def add_indicators(data: pd.DataFrame, config: ScannerConfig,
                   bar_is_partial: bool = False) -> pd.DataFrame:
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

    # Computed ONCE, here, because get_entry_trigger and build_candidate both
    # need it and previously each derived it inline -- two copies of the same
    # formula that had to be kept in step by a comment. The breakout gate and
    # the stored Volume_Spike column now cannot disagree by construction.
    data["volume_spike"] = data["Volume"] / data["avg_volume20_prior"]
    if bar_is_partial and len(data):
        # NaN = UNKNOWN, and every reader must treat it that way. A partial
        # day's volume against a full day's baseline is a clock reading, not a
        # volume reading (0.21 at 11:00 rising to 0.95 after the close), so the
        # honest value is "not measured yet" rather than a small number that
        # looks like genuinely thin trade. See trust_partial_volume.
        data.iloc[-1, data.columns.get_loc("volume_spike")] = np.nan
    # Liquidity deliberately still uses the inclusive average -- it is a
    # "is this tradeable" measure, not a deviation-from-normal one, and
    # including today is the more current answer.
    data["avg_traded_value20_cr"] = (
        data["Close"].rolling(20).mean() * data["Volume"].rolling(20).mean()
    ) / 10_000_000

    # The swing rule's two price conditions, both trailing.
    #
    # dip_from_high is NEGATIVE when below the 20-session high, and is measured
    # on CLOSES rather than on high_20 (which uses intraday highs) because that
    # is what the backtest measured. Using the intraday high would report a
    # deeper dip than was tested and quietly loosen min_dip_pct.
    data["high_20_close"] = data["Close"].rolling(config.breakout_lookback).max()
    data["dip_from_high"] = (data["Close"] / data["high_20_close"] - 1) * 100
    data["trend_ma"] = data["Close"].rolling(config.trend_ma_period).mean()
    data["above_trend"] = data["Close"] > data["trend_ma"]

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

    # Precomputed for run_move_profile: was this stock already "volatile"
    # (trailing average, no lookahead) as of each day.
    data["past_vol_mean"] = (
        data["volatility_measure"].rolling(config.volatility_lookback, min_periods=20).mean().shift(1)
    )

    # Trade levels for every row (vectorized, no lookahead -- ATR/recent_low/
    # Close are all trailing). Stop is the TIGHTEST of an ATR stop and a
    # structural floor.
    #
    # A third candidate, rr_floor_stop, capped risk at exactly 3% and is gone:
    # a 3% stop on a stock with 4-6% daily range sits inside its own noise, so
    # a multi-week hold was stopped out by ordinary fluctuation. It also made
    # max_risk_pct unreachable -- 384 of 384 stored rows had Risk_Pct == 3.00.
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
def run_move_profile(
    data: pd.DataFrame, config: ScannerConfig
) -> tuple[float, int, float, float, float]:
    """How far this stock typically travels over move_horizon_days, both ways.

    Nothing is thresholded. For every historically-eligible day (trailing avg
    volatility already >= min_avg_volatility as of that day -- no lookahead),
    this looks forward move_horizon_days and records how far the close got
    above entry at best and below at worst, both net of round_trip_cost_pct.

    It replaced a 3%-floor win rate, which came from the retired "sell at 3%
    and repeat" goal and was actively deselecting the winners: the four best
    trades scored 5.6, 10.0 and 17.4 out of 100 and DYCL was rejected outright.
    A pass/fail bar also scores a +19% window and a -5% window identically.

    Both directions are reported because the hardest movers also fall hardest:
    the top quintile by big-move frequency hit +20% in 23.8% of windows but
    ALSO -20% in 9.4%, against 1.2% for the bottom quintile.

    Sub-periods are anchored to the END of the array, so a stock whose
    behaviour changed recently is not judged on stale history.

    Returns (typical_move_pct, sample_size, typical_drawdown_pct, move_stability,
    tail_drawdown_pct). tail_drawdown_pct is the 10th-percentile trough -- a bad
    window, not a typical one -- because a collapse does not live in a median:
    across live candidates the median drawdown was -3.5% while p10 was -12.2%
    and the worst reached -30.7%.
    """
    lookback = config.persistence_lookback
    horizon = config.move_horizon_days
    cost = config.round_trip_cost_pct
    if len(data) < lookback + horizon:
        return 0.0, 0, 0.0, 999.0, 0.0

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
        return 0.0, 0, 0.0, 999.0, 0.0

    typical_move = round(float(np.median(peaks)), 2)
    typical_drawdown = round(float(np.median(troughs)), 2)
    tail_drawdown = round(float(np.percentile(troughs, 10)), 2)
    stability = round(float(np.std(move_by_period)), 1) if len(move_by_period) > 1 else 0.0

    return typical_move, len(peaks), typical_drawdown, stability, tail_drawdown


def upside_rate(data: pd.DataFrame, config: ScannerConfig) -> float:
    """
    Share of forward windows in which the best gain beat the worst loss in
    magnitude -- "when this stock moves, how often does it move UP first/further".

    This is the honest replacement for the old win rate. It is not a probability
    of profit and must not be read as one: it says the upside dominated the
    downside in that window, not that a trade would have been exited there.
    Exits are a human decision (see position_check), so nothing here simulates
    one.

    Costs are charged on the same convention as run_move_profile -- deducted
    from BOTH ends. Because the trough is negative, subtracting the cost widens
    the loss while shrinking the gain, so the comparison becomes
    peak > |trough| + 2*cost. Leaving costs out here (as this originally did)
    made the measure quietly more optimistic than the move profile it sits
    beside, which is exactly the kind of inconsistency that makes two numbers
    in the same row disagree.
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
        cost = config.round_trip_cost_pct
        peak = (window.max() - entry) / entry * 100 - cost
        trough = (window.min() - entry) / entry * 100 - cost
        total += 1
        wins += peak > abs(trough)
    return round(wins / total * 100, 1) if total else 0.0


def zigzag_pivots(closes: np.ndarray, pct: float) -> list[tuple[int, str]]:
    """Turning points, confirmed only after price reverses by `pct`.

    Returns [(index, 'H'|'L')]. A pivot is only emitted once the reversal is
    confirmed, so this never uses information from beyond the bar it is called
    with -- the confirming move has already happened by then.
    """
    if len(closes) < 2:
        return []
    pivots: list[tuple[int, str]] = []
    ext = closes[0]; ext_i = 0; direction = 0
    for i in range(1, len(closes)):
        if direction > 0:
            if closes[i] > ext:
                ext, ext_i = closes[i], i
            elif closes[i] <= ext * (1 - pct / 100):
                pivots.append((ext_i, "H")); direction = -1; ext, ext_i = closes[i], i
        elif direction < 0:
            if closes[i] < ext:
                ext, ext_i = closes[i], i
            elif closes[i] >= ext * (1 + pct / 100):
                pivots.append((ext_i, "L")); direction = 1; ext, ext_i = closes[i], i
        else:
            if closes[i] >= ext * (1 + pct / 100):
                direction = 1; ext, ext_i = closes[i], i
            elif closes[i] <= ext * (1 - pct / 100):
                direction = -1; ext, ext_i = closes[i], i
            elif closes[i] > ext:
                ext, ext_i = closes[i], i
    return pivots


def measure_bounce_history(data: pd.DataFrame, config: ScannerConfig) -> tuple[float, int]:
    """This stock's own record of falling and coming back.

    Returns (median completed trough->peak gain %, number of such legs) over the
    last persistence_lookback sessions.

    This is the swing analogue of Expected_Move, and it is what this file ranks
    on. The distinction matters: Expected_Move asks "how far does it travel",
    which a stock going up in a straight line answers well; this asks "when it
    falls, how far does it come back", which only an oscillator answers well. A
    straight ramp scores 0 here because it never completes a trough->peak leg.

    Measured: ranking the dip cohort by this returned +10.06% against +9.06%
    for ranking the same cohort by Expected_Move and +8.76% for ranking by
    deepest dip. Across the whole qualified pool it correlates +0.048 with
    forward 30-session return -- weaker than Expected_Move's +0.064, so it is
    NOT a better general predictor. It is better specifically at ordering
    stocks that have already dipped, which is the only population this file
    ever ranks.
    """
    window = data["Close"].tail(config.persistence_lookback).to_numpy()
    window = window[np.isfinite(window)]
    if len(window) < config.persistence_lookback // 2:
        return 0.0, 0
    pivots = zigzag_pivots(window, config.zigzag_pct)
    legs = [
        (window[b] - window[a]) / window[a] * 100
        for (a, ka), (b, kb) in zip(pivots, pivots[1:])
        if ka == "L" and kb == "H" and window[a] > 0
    ]
    return (round(float(np.median(legs)), 2) if legs else 0.0), len(legs)


def measure_spike_ratio(data: pd.DataFrame, config: ScannerConfig) -> float:
    """How concentrated this stock's movement is in a few wild days.

    95th-percentile absolute daily close move over the median one. Around 2
    means most sessions look alike; above 4 means the stock is quiet until it
    gaps, and a gap is not something you can enter.

    This separates two things Stage 2 cannot tell apart. Both a steady 5%-a-day
    mover and a flat stock that jumps 15% twice a month can clear the same
    avg/median volatility gates, but only one of them is tradeable.

    Returns inf when there is too little history to judge, so the caller
    rejects rather than guesses.
    """
    # spike_lookback, NOT volatility_lookback -- see that field for why this one
    # stayed at 180 when the volatility window shortened to 60.
    hist = data["abs_close_move"].tail(config.spike_lookback).dropna()
    if len(hist) < config.spike_lookback // 2:
        return float("inf")
    median_move = float(hist.median())
    if median_move <= 0:
        return float("inf")
    return float(np.percentile(hist.to_numpy(), 95)) / median_move


def measure_max_drawdown(data: pd.DataFrame, config: ScannerConfig) -> float:
    """Worst peak-to-trough fall within drawdown_lookback, as a negative percent.

    Backward-looking on purpose. It is not a forecast that the stock will fall
    again -- it is the observation that it already has, which is the only
    evidence available that a collapse is inside its range of behaviour.
    """
    window = data["Close"].tail(config.drawdown_lookback).to_numpy()
    if len(window) < 60:
        return 0.0
    peak = np.maximum.accumulate(window)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, (window - peak) / peak, 0.0)
    return round(float(dd.min()) * 100, 2)


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
    # NaN here means the session is still open and volume is not yet measurable.
    volume_spike = float(latest["volume_spike"])
    volume_known = bool(np.isfinite(volume_spike))

    bullish = close > float(latest["Open"])
    closed_above_prev_high = close > float(prev["High"])

    if pd.isna(ema20) or pd.isna(high_20_prev) or pd.isna(pullback):
        return "WATCH", "no_data", "Indicators not ready"

    # breakout and pullback_bounce BUY; reclaim stays WATCH-only. No win-rate
    # claim in that split -- across three 400-500 stock samples the three setups
    # were indistinguishable and their ranking FLIPPED between samples.
    #
    # Worth knowing when acting on a breakout: it fires a median +6.35% AFTER
    # the move (979 trigger days, 87% closed >3% up). The 2.3x volume
    # requirement guarantees that lateness. The backtest enters at the next
    # open, so this is already priced in, not an extra penalty.
    #
    # While the session is open the volume requirement is WAIVED, not failed --
    # 2.3x of a partial count is arithmetically unavailable before 15:30, which
    # is why midday runs produced almost no breakouts.
    breakout_volume_ok = volume_spike >= config.breakout_volume_mult if volume_known else True
    if close > high_20_prev and breakout_volume_ok and bullish:
        reason = (f"Broke 20D high on {volume_spike:.1f}x volume" if volume_known
                  else "Broke 20D high - volume unconfirmed, session still open")
        return "BUY", "breakout", reason

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
# Score point budget. Positive weights sum to 100; SUPPORT_PENALTY_PTS subtracts.
#
# Two rules produced these:
#  1. NEVER SCORE WHAT THE GATES ALREADY ENFORCE. An earlier budget spent 38 of
#     100 points on four measures that were already hard gates, so every
#     survivor scored the same on them -- they averaged 27.2 points of a typical
#     45 while barely varying, which is why scores all landed 40-60.
#  2. RISK RANKS, IT DOES NOT VETO. TAIL_PTS is large on purpose; it replaced
#     hard gates that were removing good stocks outright.
#
# Stability is deliberately small: it returns only 0, half or full, and under an
# older budget that 3-step flag drove 37.8% of all ranking variance.
#
# NOT A PREDICTIVE CLAIM -- no version of this has been shown to predict forward
# returns (the previous one correlated -0.153).
MOVE_PTS = 40
UPSIDE_PTS = 15
TAIL_PTS = 20
STABILITY_PTS = 8
VOLUME_PTS = 5
SETUP_PTS = 12
SUPPORT_PENALTY_PTS = 10

# ANCHORS. Each component scores between its MINIMUM ADMISSIBLE value and a
# genuinely excellent one, never between 0 and an unreachable ideal. The old
# version scored from a 100% win rate no stock reached, so half the budget was
# unreachable and the first quarter was free -- every score landed 40-60
# (mean 44.97, sd 4.01). Now spans 2.4-76.0, sd 13.4.
EXCELLENT_MOVE_PCT = 20.0           # travels 20% in 30 sessions = full marks
# Tail anchors. -20% scores nothing: a stock whose bad windows run that deep is
# a different animal from one that dips 5%, even if both travel 20% up. Range
# observed across live candidates was -23.0% to -5.9%.
WORST_TAIL_PCT = -20.0
EXCELLENT_TAIL_PCT = -5.0
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
    tail_drawdown: float,
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
    ) * MOVE_PTS
    # How often the upside actually dominated the downside over the horizon.
    # Anchored from 50% -- below that the stock fell further than it rose in
    # most windows, which deserves nothing, not partial credit.
    upside_score = _span(upside_pct, 50.0, EXCELLENT_UPSIDE_RATE) * UPSIDE_PTS
    # Risk, as a ranking term rather than a veto. tail_drawdown is the
    # 10th-percentile forward trough -- a bad window, not a typical one -- so
    # this is what separates "moves 20% and dips 5%" from "moves 20% and dips
    # 20%". Both are volatile; only one of them is comfortable to hold.
    tail_score = _span(tail_drawdown, WORST_TAIL_PCT, EXCELLENT_TAIL_PCT) * TAIL_PTS

    if move_stability < config.move_stability_threshold:
        stability_bonus = STABILITY_PTS
    elif move_stability < config.move_stability_threshold * 1.5:
        stability_bonus = STABILITY_PTS / 2
    else:
        stability_bonus = 0.0

    # From 1.0x, not from 0 -- trading your own average volume is the null
    # result, and used to collect a third of this budget for it.
    #
    # NaN means the session is still open, so volume is not measurable yet (see
    # add_indicators). That scores 0 of 5 -- the same as an average day, which
    # is the right neutral: it neither rewards a stock for a spike it has not
    # been shown to have, nor punishes it for one the clock has hidden. It does
    # mean a midday score is at most 95 of the 100 an after-close score can
    # reach, and that 5-point ceiling is the honest cost of scoring early.
    volume_bonus = (
        _span(volume_spike, NORMAL_VOLUME_RATIO, EXCELLENT_VOLUME_RATIO) * VOLUME_PTS
        if np.isfinite(volume_spike) else 0.0
    )

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
        move_score + upside_score + tail_score + stability_bonus
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

    # Character gates -- see max_spike_ratio / min_drawdown_pct in ScannerConfig.
    # Placed here, immediately after the volatility identity, because they
    # qualify the SAME measurement: Stage 2 established that this stock moves,
    # these two establish that it moves in a way you can hold.
    spike_ratio = measure_spike_ratio(data, config)
    if not np.isfinite(spike_ratio) or spike_ratio > config.max_spike_ratio:
        return None
    max_drawdown_pct = measure_max_drawdown(data, config)
    if max_drawdown_pct < config.min_drawdown_pct:
        return None

    # --- THE SWING RULE. These two conditions decide membership in this file,
    # --- where momentum.py uses min_typical_move_pct instead.
    dip_from_high = float(latest["dip_from_high"])
    above_trend = bool(latest["above_trend"]) if pd.notna(latest["above_trend"]) else False
    if pd.isna(dip_from_high) or pd.isna(latest["trend_ma"]):
        return None
    # Has it actually fallen far enough to be a dip worth buying?
    if dip_from_high > -config.min_dip_pct:
        return None
    # Is this a pullback in an uptrend rather than a falling knife? Worth 4
    # points of return (+10.06% with, +6.09% without).
    if not above_trend:
        return None

    bounce_median, bounce_legs = measure_bounce_history(data, config)
    if bounce_legs < config.min_bounce_legs or bounce_median <= 0:
        return None

    typical_move, sample_size, typical_drawdown, move_stability, tail_drawdown = run_move_profile(data, config)
    # Quality floor, not the membership gate -- see min_typical_move_pct.
    if typical_move < config.min_typical_move_pct:
        return None
    upside_dominance = upside_rate(data, config)
    # min_persistence_sample is NOT enforced here. It guards the RELIABILITY of
    # Expected_Move, which this file records but does not rank on -- and in the
    # gate test it rejected a cohort that went on to return +8.30% against the
    # survivors' +3.95%, because a thin sample means the stock only recently
    # became volatile. momentum.py still enforces it.

    # Action is OVERRIDDEN to BUY for everything that reaches here, and the
    # reason is structural rather than optimistic.
    #
    # get_entry_trigger returns deep_pullback -> WATCH for any pullback over
    # 12%, which is precisely what the swing rule selects for. So its Action was
    # WATCH on 100% of rows -- a column fully determined by the gate that
    # admitted the row, telling the reader nothing. On the first live run all 7
    # candidates read WATCH.
    #
    # In this file the LIST IS THE SIGNAL: a qualified stock that has fallen
    # >= min_dip_pct while holding above its trend average is the setup, and
    # nothing further needs to confirm it. Measured, waiting for confirmation
    # is actively worse -- deep_pullback (still falling) returned +3.65%
    # against pullback_bounce (confirmed turn) at +0.02%.
    #
    # The raw trigger output is still stored in Setup_Type and Reason, so
    # nothing is lost; only the word the phone shows changes.
    _raw_action, setup_type, reason = get_entry_trigger(data, config)
    action = "BUY"
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
    # The same column get_entry_trigger gated on -- one definition, so the
    # stored value and the gate cannot drift apart. NaN while the session is
    # open, and written to BigQuery as NULL.
    volume_spike = float(latest["volume_spike"])

    score = calculate_score(
        typical_move, move_stability, upside_dominance, tail_drawdown, volume_spike,
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
        # Renamed to say what they now hold. The old Persistence_* names were
        # carried over from the retired 3%-floor backtest and had stopped
        # describing their contents. NOTE: renaming these requires a manual
        # Dataform workspace pull before the views pick them up -- see
        # trigger_dataform_run for why that sync is manual.
        "Upside_Dominance_Pct": upside_dominance,  # % of windows upside beat downside
        "Upside_Dominance_Sample": sample_size,
        "Expected_Move": typical_move,             # median 30-session peak gain %
        "Move_Stability": move_stability,          # variability of that gain
        "Typical_Drawdown_Pct": typical_drawdown,  # median worst loss in the same windows
        "Tail_Drawdown_Pct": tail_drawdown,        # 10th-percentile (bad) window
        "Bounce_Median": bounce_median,                # median trough->peak leg %
        "Bounce_Legs": bounce_legs,                    # completed legs behind it
        "Dip_From_High_Pct": round(dip_from_high, 2),  # negative: below 20D high
        "Spike_Ratio": round(spike_ratio, 2),          # p95 / median daily move
        "Max_Drawdown_1Y_Pct": max_drawdown_pct,       # worst peak-to-trough, trailing year
        "Traded_Value_Cr": round(avg_traded_value20_cr, 2),
        "Volume_Spike": round(volume_spike, 2),
        "Pullback_Pct": round(float(latest["pullback_pct"]), 2),
        "Dist_Support_Pct": round(distance_from_support, 2),
        "Dist_MA20_Pct": round(distance_from_ma20, 2),
        "Entry": risk["entry"],
        "Stop_Loss": risk["stop_loss"],
        "Typical_Move_Price": risk["target"],
        "Risk_Per_Share": risk["risk_per_share"],
        "Risk_Pct": risk["risk_pct"],
        "RR_Ratio": risk["rr_ratio"],
        "ATR": risk["atr"],
        "Move_Horizon_Days": config.move_horizon_days,
    }
# =========================================================
# SCANNER
# =========================================================
def scan_ticker_data(ticker: str, data: pd.DataFrame, config: ScannerConfig, run_date: str,
                     run_timestamp: str, bar_is_partial: bool = False) -> Optional[dict[str, Any]]:
    if not has_enough_data(data, config):
        return None
    data = add_indicators(data, config, bar_is_partial=bar_is_partial)
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
    # Tickers dropped for bar recency rather than for quality. Counted, not
    # silently discarded: a day when Yahoo is missing a session for most of the
    # universe looks identical to a quiet market unless someone prints this.
    # On 2026-09-18 it would have been ~73% of tickers.
    bar_state_counts: dict[str, int] = {"current": 0, "stale": 0, "gapped": 0, "empty": 0}
    batches = chunks(tickers, config.chunk_size)
    total = len(batches)
    attempted = 0

    calendar = (
        market_session_calendar(yf, tickers, config) if config.require_current_bar else None
    )
    bar_is_partial = (
        is_partial_session(calendar[-1]) and not config.trust_partial_volume
        if calendar else False
    )
    if calendar:
        today_ist = (dt.datetime.now(dt.timezone.utc) + IST_OFFSET).date()
        lag = (today_ist - calendar[-1]).days
        # 3 days absorbs an ordinary weekend. Past that, Yahoo is genuinely
        # behind and the whole run is working on older bars than the date
        # suggests -- worth saying out loud rather than leaving in Bar_Date.
        lag_note = (f" -- {lag} calendar days behind today, Yahoo has not caught up"
                    if lag > 3 else "")
        state = "session still open, volume unmeasured" if bar_is_partial else "complete"
        print(f"Reference session: {calendar[-1]} ({state}){lag_note} | requiring the "
              f"last {config.contiguous_sessions_required} sessions hole-free")

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
                    frame, bar_state = align_to_session(
                        get_ticker_frame(batch_data, ticker), calendar,
                        config.contiguous_sessions_required,
                    )
                    bar_state_counts[bar_state] += 1
                    if bar_state != "current":
                        continue
                    candidate = scan_ticker_data(ticker, frame, config, run_date,
                                                 run_timestamp, bar_is_partial)
                    if candidate:
                        results.append(candidate)
                except Exception as exc:
                    ticker_errors[ticker] = f"{type(exc).__name__}: {exc}"
                    if config.verbose:
                        print(f"\nSkipped {ticker}: {exc}")

        print(f"\rProgress: {int(bn/total*100)}% ({bn}/{total}) | {len(results)} found", end="", flush=True)
    print()

    # Two independent triggers, because a code fault does not always show up as
    # a HIGH failure rate.
    #
    # RATE catches total failure: renaming support_distance_threshold made all
    # 2,574 tickers raise, and five scheduled runs reported "0 candidates found"
    # with nothing worse than a warning.
    #
    # TYPE catches partial failure, which the rate misses: deleting max_risk_pct
    # broke only the 16 of 400 tickers that reached the risk check -- 4%, under
    # the 25% threshold, silently reported as zero candidates.
    # AttributeError/NameError/TypeError essentially never come from market data.
    CODE_FAULT_TYPES = ("AttributeError", "NameError", "TypeError")
    CODE_FAULT_MIN_COUNT = 5
    kind_counts: dict[str, int] = {}
    for msg in ticker_errors.values():
        kind_counts[msg.split(":")[0]] = kind_counts.get(msg.split(":")[0], 0) + 1
    code_faults = {k: v for k, v in kind_counts.items()
                   if k in CODE_FAULT_TYPES and v >= CODE_FAULT_MIN_COUNT}
    rate_exceeded = bool(attempted) and len(ticker_errors) / attempted > config.max_ticker_error_rate

    if code_faults or rate_exceeded:
        top_kind = max(code_faults or kind_counts, key=(code_faults or kind_counts).get)
        sample = next(m for m in ticker_errors.values() if m.startswith(top_kind))
        why = (f"{top_kind} raised {kind_counts[top_kind]}x -- that exception type comes "
               f"from code, not from market data"
               if code_faults else
               f"{len(ticker_errors) / attempted:.0%} of tickers raised, over the "
               f"{config.max_ticker_error_rate:.0%} limit")
        raise RuntimeError(
            f"{len(ticker_errors)}/{attempted} tickers raised inside scan_ticker_data. "
            f"{why}. This is a code fault, not missing data. Example: {sample}"
        )

    failures = batch_failures + list(ticker_errors)
    if failures:
        LOGGER.warning(
            "%d/%d tickers unusable (%d failed batch download, %d raised while scanning)",
            len(failures), len(tickers), len(batch_failures), len(ticker_errors),
        )

    unusable = bar_state_counts["stale"] + bar_state_counts["gapped"]
    checked = unusable + bar_state_counts["current"]
    if unusable:
        stale_rate = unusable / checked if checked else 0.0
        message = (
            f"{unusable} of {checked} tickers ({stale_rate:.0%}) skipped on bar recency: "
            f"{bar_state_counts['stale']} never reached {calendar[-1]}, "
            f"{bar_state_counts['gapped']} reached it but are missing a session inside "
            f"the last {config.contiguous_sessions_required}. Yahoo's prices are fine; "
            f"the sessions simply are not there."
        )
        # The reference session is chosen so that the large majority of the
        # universe clears it -- measured, 100% current and 99% contiguous. So
        # unlike the earlier megacap-based rule, a high rate here is genuinely
        # anomalous rather than the normal state of the feed, and it means the
        # coverage probe disagreed with the full universe. Rerunning will not
        # help; the thing to check is min_session_coverage.
        if stale_rate > 0.25:
            print(f"\nWARNING: {message}\n"
                  f"         That is far above the expected ~1%. The probe sample and the "
                  f"full universe disagree about {calendar[-1]} -- check min_session_coverage "
                  f"({config.min_session_coverage:.0%}) before trusting this run.")
        LOGGER.warning("%s", message)

    if not results:
        return pd.DataFrame(), failures, 0

    df = pd.DataFrame(results)
    # Ranked by Bounce_Median -- this stock's own history of recovering from a
    # dip -- NOT by Expected_Move, Score or Action.
    #
    # Measured on the dip cohort, top 3 per day, 30-session hold:
    #   rank by Bounce_Median  +10.06%  win 65.0%  halves 9.61 / 10.62
    #   rank by Expected_Move   +9.06%  win 62.4%  halves 10.87 / 6.87
    #   rank by deepest dip     +8.76%  win 60.5%  halves 11.40 / 5.95
    # Bounce history wins on return, win rate AND balance between the two
    # disjoint ticker halves.
    #
    # Note it is the better ranker only for THIS population. Across the whole
    # qualified pool it correlates +0.048 with forward return against
    # Expected_Move's +0.064. It is better at ordering stocks that have already
    # fallen, which is the only population this file ever sees.
    #
    # Score and Action are stored but rank nothing: Score correlated -0.065
    # with forward return, and BUY returned +3.54% against WATCH's +4.42%.
    df = df.sort_values(["Bounce_Median", "Expected_Move"], ascending=[False, False])

    total_quality_candidates = len(df)
    # top_n caps STORAGE only; the display cap is applied by the caller. These
    # were the same number until the list grew past what one person can read,
    # at which point shortening the report also started shortening the history.
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
    bigquery.SchemaField("Upside_Dominance_Pct", "FLOAT64"),
    bigquery.SchemaField("Upside_Dominance_Sample", "INT64"),
    bigquery.SchemaField("Expected_Move", "FLOAT64"),
    bigquery.SchemaField("Move_Stability", "FLOAT64"),
    bigquery.SchemaField("Typical_Drawdown_Pct", "FLOAT64"),
    bigquery.SchemaField("Tail_Drawdown_Pct", "FLOAT64"),
    bigquery.SchemaField("Bounce_Median", "FLOAT64"),
    bigquery.SchemaField("Bounce_Legs", "INT64"),
    bigquery.SchemaField("Dip_From_High_Pct", "FLOAT64"),
    bigquery.SchemaField("Spike_Ratio", "FLOAT64"),
    bigquery.SchemaField("Max_Drawdown_1Y_Pct", "FLOAT64"),
    bigquery.SchemaField("Traded_Value_Cr", "FLOAT64"),
    bigquery.SchemaField("Volume_Spike", "FLOAT64"),
    bigquery.SchemaField("Pullback_Pct", "FLOAT64"),
    bigquery.SchemaField("Dist_Support_Pct", "FLOAT64"),
    bigquery.SchemaField("Dist_MA20_Pct", "FLOAT64"),
    bigquery.SchemaField("Entry", "FLOAT64"),
    bigquery.SchemaField("Stop_Loss", "FLOAT64"),
    bigquery.SchemaField("Typical_Move_Price", "FLOAT64"),
    bigquery.SchemaField("Risk_Per_Share", "FLOAT64"),
    bigquery.SchemaField("Risk_Pct", "FLOAT64"),
    bigquery.SchemaField("RR_Ratio", "FLOAT64"),
    bigquery.SchemaField("ATR", "FLOAT64"),
    bigquery.SchemaField("Move_Horizon_Days", "INT64"),
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

    # De-duplication keyed on Run_Timestamp, not (Ticker, Bar_Date).
    #
    # The old key destroyed the intraday history Run_Timestamp exists to
    # capture: with 4+ runs a day each run DELETEd earlier rows for any ticker
    # it also found, so only the last run per ticker per bar survived. Confirmed
    # live: 0 (Ticker, Bar_Date) pairs ever appeared twice.
    #
    # A whole scan shares one timestamp, so this is a single equality. Re-running
    # produces a new row, which is correct -- that IS a second observation. The
    # DELETE still protects a retried load reusing the same timestamp.
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


def format_telegram_digest(candidates: pd.DataFrame, run_date: str,
                           limit: int = 10) -> Optional[str]:
    """Name and Action for the top `limit` rows, in the order the console prints.

    Returns None only when the scan found nothing at all.

    Sends the top of the ranked list, not fresh BUYs. The old
    (Action == BUY and Setup_Age_Days == 1) filter selected the worse cohort:
    BUY returned +0.33% against WATCH's +2.75% over 605 stored signals at a
    10-session horizon, and through BLISSGVS's +177% run and E2E's +85% run the
    scanner said WATCH every day, so Telegram said nothing at all.

    Plain text, no parse_mode -- see send_telegram_notification.
    """
    if candidates.empty:
        return None

    rows = candidates.head(limit)
    run_time_ist = (dt.datetime.now(dt.timezone.utc) + IST_OFFSET).strftime("%Y-%m-%d %H:%M IST")
    width = max((len(str(t).replace(".NS", "")) for t in rows["Ticker"]), default=10)
    lines = [f"Swing (dip) {run_time_ist}", ""]
    for n, (_, r) in enumerate(rows.iterrows(), start=1):
        ticker = str(r["Ticker"]).replace(".NS", "")
        lines.append(f"{n}. {ticker:<{width}}  {r['Action']}")
    if len(candidates) > len(rows):
        lines += ["", f"Top {len(rows)} of {len(candidates)}."]
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
        description="Volatility swing scanner - finds stocks that genuinely move, "
                    "measures how far they travel over a 30-session window in both "
                    "directions, and ranks them. It does not model an exit."
    )
    p.add_argument("--tickers", nargs="+", default=None)
    p.add_argument("--symbols-source", default=NSE_EQUITY_LIST_URL)
    p.add_argument("--limit", default=0, type=int)
    p.add_argument("--only-buy", action="store_true")

    p.add_argument("--min-avg-volatility", default=ScannerConfig.min_avg_volatility, type=float)
    p.add_argument("--range-event-threshold", default=ScannerConfig.range_event_threshold, type=float)
    # Defaults MUST read the dataclass, never a literal. --min-volatile-days sat
    # at a hardcoded 30 after the field moved to 12, so every local CLI run was
    # silently stricter than the deployed HTTP path, which reads the dataclass.
    p.add_argument("--min-volatile-days", default=ScannerConfig.min_volatile_days, type=int)

    p.add_argument("--min-dip", default=ScannerConfig.min_dip_pct, type=float,
                    help="How far below the 20-session high the stock must have fallen")
    p.add_argument("--trend-ma", default=ScannerConfig.trend_ma_period, type=int,
                    help="Moving average the stock must still be above")
    p.add_argument("--min-typical-move", default=ScannerConfig.min_typical_move_pct, type=float,
                    help="Recorded, not gated on, in this file -- see momentum.py")
    p.add_argument("--move-horizon-days", default=ScannerConfig.move_horizon_days, type=int,
                    help="Forward window the move profile measures over")
    p.add_argument("--min-persistence-sample", default=ScannerConfig.min_persistence_sample, type=int)

    p.add_argument("--min-price", default=ScannerConfig.min_price, type=float)
    p.add_argument("--max-price", default=ScannerConfig.max_price, type=float)
    p.add_argument("--min-traded-value-cr", default=ScannerConfig.min_avg_traded_value_cr, type=float)

    p.add_argument("--round-trip-cost-pct", default=ScannerConfig.round_trip_cost_pct, type=float,
                    help="Round-trip transaction cost %% charged to every backtest trial. "
                         "Raise it if you trade smaller than ~Rs.25,000 a position, since the "
                         "flat DP charge is a bigger share of a smaller trade")
    p.add_argument("--max-risk-pct", default=ScannerConfig.max_risk_pct, type=float)
    p.add_argument("--min-score", default=ScannerConfig.min_score, type=float,
                    help="Hard quality gate (0-100); candidates scoring below this are rejected outright")

    p.add_argument("--top-n", default=ScannerConfig.top_n, type=int,
                    help="Max rows written to BigQuery. Keep high -- stored history feeds "
                         "the signal-history test, which needs low scores too")
    p.add_argument("--display-top-n", default=ScannerConfig.display_top_n, type=int,
                    help="How many rows to print and send to Telegram. Attention knob, "
                         "not a filter -- storage is unaffected")
    p.add_argument("--output", default="", help="CSV output path")
    p.add_argument("--no-bq", action="store_true", help="Skip writing results to BigQuery")
    p.add_argument("--no-dataform-trigger", action="store_true",
                    help="Skip triggering a Dataform run after a successful BigQuery write")
    p.add_argument("--no-telegram", action="store_true",
                    help="Skip sending the Telegram notification")
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
    and argparse's prefix-matching silently mapped "--target=main" onto a
    --target-pct flag this file defined at the time, then crashed trying to
    parse "main" as a float, failing every single HTTP request. That specific
    flag is gone, but the guard must stay: any future flag whose name prefixes
    one of functions-framework's own would reintroduce it. Config here comes from
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
            min_dip_pct=float(body.get("min_dip", ScannerConfig.min_dip_pct)),
            trend_ma_period=int(body.get("trend_ma", ScannerConfig.trend_ma_period)),
            move_horizon_days=int(body.get("move_horizon_days", ScannerConfig.move_horizon_days)),
            min_persistence_sample=int(body.get("min_persistence_sample", ScannerConfig.min_persistence_sample)),
            min_price=float(body.get("min_price", ScannerConfig.min_price)),
            max_price=float(body.get("max_price", ScannerConfig.max_price)),
            min_avg_traded_value_cr=float(body.get("min_traded_value_cr", ScannerConfig.min_avg_traded_value_cr)),
            round_trip_cost_pct=float(body.get("round_trip_cost_pct", ScannerConfig.round_trip_cost_pct)),
            max_risk_pct=float(body.get("max_risk_pct", ScannerConfig.max_risk_pct)),
            min_score=float(body.get("min_score", ScannerConfig.min_score)),
            top_n=int(body.get("top_n", ScannerConfig.top_n)),
            display_top_n=int(body.get("display_top_n", ScannerConfig.display_top_n)),
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
            min_dip_pct=args.min_dip,
            trend_ma_period=args.trend_ma,
            move_horizon_days=args.move_horizon_days,
            min_persistence_sample=args.min_persistence_sample,
            min_price=args.min_price,
            max_price=args.max_price,
            min_avg_traded_value_cr=args.min_traded_value_cr,
            round_trip_cost_pct=args.round_trip_cost_pct,
            max_risk_pct=args.max_risk_pct,
            min_score=args.min_score,
            top_n=args.top_n,
            display_top_n=args.display_top_n,
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
    print(f"Volatility: last {config.volatility_lookback} sessions, avg >= {config.min_avg_volatility}% | "
          f"median >= {config.min_median_volatility}% | {config.min_volatile_days}+ volatile days "
          f"(>= {config.min_volatility_ratio:.0%})")
    print(f"Stop: ATR/structure only, risk capped at {config.max_risk_pct}% "
          f"(no fixed profit floor -- exits are a human decision)")
    print(f"Swing rule: fallen >= {config.min_dip_pct:.0f}% from the 20-session high, "
          f"still above the {config.trend_ma_period}-day average")
    print(f"Quality floor: typical 30-session gain >= {config.min_typical_move_pct:.0f}% "
          f"(a floor under the ranker, not the membership gate)")
    print(f"Ranked by: median recovery from a dip, over the last "
          f"{config.persistence_lookback} sessions ({config.zigzag_pct:.0f}% zigzag, "
          f"{config.min_bounce_legs}+ completed legs)")
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
        # not listed: it no longer exists, and the stop is ATR/structural.
        #
        # --min-score is also not listed: at 5.0 it is already near-zero on a
        # 0-100 scale (see the field comment), so there is no lower value left
        # that means anything as a relaxation hint. min_typical_move_pct is
        # what actually decides membership now.
        print(f"  --min-dip 8                  (shallower dip, now {config.min_dip_pct:.0f}%)")
        print(f"  --min-avg-volatility 3.0     (lower volatility bar, now {config.min_avg_volatility})")
        print(f"  --max-price 2000             (widen the universe, now {config.max_price:.0f})")
        print("  A quiet day here is NORMAL: this rule fired on 75 of 164 tested")
        print("  sessions. A dip that deep is not an everyday event.")
        return (message, 200) if request is not None else None

    display = candidates
    if only_buy:
        display = candidates[candidates["Action"] == "BUY"]
        if display.empty:
            print("\nNo BUY signals today. Showing top WATCH candidates instead:")
            display = candidates
    shown = len(display)
    display = display.head(config.display_top_n)

    print("\n" + "-" * 100)
    # This label must keep matching the sort in scan_tickers. It has been wrong
    # twice: it claimed "by score" after Score stopped being the key, then
    # "BUY first" after Action stopped being one. A header that describes a
    # different ordering than the rows below it is worse than no header, because
    # it is believed.
    print("RESULTS (ranked by Bounce% -- this stock's median recovery from a dip)")
    print(f"Every row has fallen >= {config.min_dip_pct:.0f}% from its 20-session high and is still")
    print(f"above its {config.trend_ma_period}-day average. Action/Setup do NOT order the list.")
    print("-" * 100)
    header = (f"{'Ticker':<13} {'Action':<7} {'Setup':<16} {'Dip%':>7} {'Bounce%':>8} {'Legs':>5} "
              f"{'Score':>6} {'Vol%':>6} {'Move%':>7} {'TailDD':>8} {'RR':>5} "
              f"{'Entry':>9} {'SL':>9} {'MovePx':>9}")
    print(header)
    print("-" * 100)
    for _, r in display.iterrows():
        print(f"{r['Ticker']:<13} {r['Action']:<7} {r['Setup_Type']:<16} "
              f"{r['Dip_From_High_Pct']:>6.1f}% {r['Bounce_Median']:>7.1f}% {r['Bounce_Legs']:>5} "
              f"{r['Score']:>6.1f} {r['Avg_Volatility']:>5.1f}% "
              f"{r['Expected_Move']:>6.1f}% {r['Tail_Drawdown_Pct']:>7.1f}% {r['RR_Ratio']:>5.1f} "
              f"{r['Entry']:>9.2f} {r['Stop_Loss']:>9.2f} {r['Typical_Move_Price']:>9.2f}")

    if output_path:
        candidates.to_csv(output_path, index=False)
        print(f"\nSaved {len(candidates)} rows to {output_path}")

    if shown > len(display):
        print(f"\nShowing the top {len(display)} of {shown} by Move%. "
              f"All {len(candidates)} are written to BigQuery -- raise --display-top-n to see more.")
    else:
        print(f"\nTotal quality candidates: {total_quality}")
    if total_quality > len(candidates):
        print(f"WARNING: {total_quality} candidates found but only {len(candidates)} stored "
              f"(top_n={config.top_n}). Raise --top-n or history is being lost.")

    if not no_telegram:
        try:
            bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
            chat_id_raw = os.getenv("TELEGRAM_CHAT_ID")
            if not bot_token or not chat_id_raw:
                print("\nTelegram notification skipped: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set.")
            else:
                digest_text = format_telegram_digest(candidates, run_date, config.display_top_n)
                if digest_text is None:
                    # Only reachable when the scan found nothing at all. It used
                    # to say "no fresh Age==1 BUY signals", which stopped being
                    # true when the digest switched to sending the top of the
                    # ranked list rather than fresh BUYs.
                    print("\nNo candidates -- Telegram notification skipped.")
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
