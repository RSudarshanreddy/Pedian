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
    # Data
    period: str = "2y"            # needs enough runway for two stacked 180d lookbacks
    interval: str = "1d"
    min_days: int = 220

    # --- Data integrity (verified against Kite broker history, 2026-09-19) ---
    #
    # Yahoo's NSE daily series has two defects that the scanner used to consume
    # silently. Both were confirmed by diffing yfinance against Zerodha Kite,
    # which is the same data the account actually trades on. On 33 overlapping
    # bars for AEROFLEX and SHILPAMED, OHLCV matched to 0.0000% -- so Yahoo's
    # PRICES are right. What is wrong is which bars exist.
    #
    # 1. FABRICATED HOLIDAY BARS. On an NSE holiday Yahoo emits a bar with
    #    Open=High=Low=Close=previous close and Volume=0. Kite has no such
    #    session. 2026-09-14 was a holiday; Yahoo invented a flat bar for 55%
    #    of tickers. Across a 250-ticker sample these are 2.96% of all bars and
    #    affect 100% of tickers. They are not NaN, so the old dropna() kept
    #    them, and each one reads as a perfectly calm, zero-volume day.
    #
    # 2. MISSING REAL SESSIONS. The opposite failure, and the worse one. On
    #    2026-09-17 Yahoo had no bar for 63% of the sample and on 2026-09-18
    #    none for 73% -- both were normal NSE sessions that Kite has in full.
    #
    # require_current_bar controls the fix for (2). See market_session_calendar
    # and align_to_session for the mechanism.
    #
    # Why (2) has to be a hard reject rather than a score penalty: a stale bar
    # does not make a signal lower-quality, it makes it WRONG. Every
    # day-over-day quantity silently spans the gap. With 2026-09-17 absent,
    # AEROFLEX's 2026-09-18 close move computes as +6.19% (481.10 -> 510.90)
    # against a true +2.67% (497.60 -> 510.90), which more than doubles it and
    # flips is_volatile_day. get_entry_trigger's prev["High"] becomes 495.00
    # instead of 512.10, so a close of 510.90 reads as clearing the prior high
    # when it did not -- a fabricated breakout.
    #
    # It also silently mislabels the stored history. The 2026-09-18 run wrote
    # an AEROFLEX row stamped Bar_Date 2026-09-16 at Score 78.0, the highest
    # score that ticker recorded all week, sitting in the same table as genuine
    # same-day rows with nothing to tell them apart. Since the whole point of
    # storing every candidate is the eventual signal-history test, admitting
    # rows whose Bar_Date is a guess corrupts the one dataset that test needs.
    require_current_bar: bool = True
    # Share of the probe sample that must have a bar before a date counts as a
    # session the whole market traded. This is what stops the scanner chasing a
    # date only megacaps have yet -- see market_session_calendar.
    #
    # Not delicate. Coverage is bimodal (100% or ~30%, nothing between), so
    # every threshold from 60% to 99% picked the same reference session on the
    # measured sample. 80% sits in the empty middle.
    min_session_coverage: float = 0.80
    # How many recent sessions must be hole-free, not just how recent the last
    # bar is. Checking only the last bar is not enough: on 2026-09-18 AEROFLEX
    # HAD a current bar and was still missing 2026-09-17 from the middle of its
    # history, which is the case that fabricates a breakout.
    #
    # 20 covers every rolling window add_indicators computes -- avg_volume20,
    # avg_volume20_prior, high_20_prev, recent_low, MA20, ATR(14) -- so a
    # surviving ticker has genuinely contiguous data everywhere those look.
    #
    # It is nearly free. Measured on 300 tickers, of the 94 with a current last
    # bar, requiring 2 clean sessions rejected 36 and requiring 20 rejected 37.
    # Yahoo's holes cluster at the live edge of the feed; older history is
    # clean. So the strict setting costs one ticker over the loose one and buys
    # correctness for every indicator in the file.
    contiguous_sessions_required: int = 20
    # Treat a partial day's volume as if it were a full day's.
    #
    # The scanner runs at 09:07, 11:00, 13:30 and 15:00 IST, so from 11:00
    # onward today's bar is real but half-formed. Volume is the measure this
    # ruins, because it is the only one compared against a FULL-day baseline.
    # Median Volume_Spike by run hour IST, from the stored runs: 0.21 at 11:00,
    # 0.34 at 12:00, 0.36 at 13:00, 0.62 at 15:00, 0.71-0.95 after the close.
    # It is not a volume signal, it is a clock.
    #
    # A breakout needs breakout_volume_mult = 2.3x against a baseline the live
    # bar cannot reach until after 15:30, so the midday runs essentially could
    # not fire one -- a large part of why "Telegram goes quiet". And the same
    # stock on the same Bar_Date scored differently in every run: SHILPAMED on
    # 2026-09-18 was written at scores 52.7 to 65.5.
    #
    # Left False, a partial bar's volume_spike is NaN -- UNKNOWN, not zero and
    # not small. Unknown means: do not gate a breakout on it, do not pay score
    # points for it, and write NULL rather than a number that is really a
    # timestamp. Price, trend, pullback and setup all still work, because those
    # read a current price rather than a day's accumulation.
    #
    # Rescaling the baseline by elapsed session time was considered and
    # rejected: NSE intraday volume is U-shaped, so a linear fraction badly
    # overstates spikes in the morning.
    trust_partial_volume: bool = False
    # Universe (hard filters)
    # The band is an affordability/position-sizing preference, not a predictor:
    # realized outcome correlated -0.030 with entry price across the tested
    # sample, i.e. nothing. Widening it changes how many names you see, not
    # their quality. Liquidity is policed separately by
    # min_avg_traded_value_cr, so lowering the floor does not admit thin
    # stocks -- a 250-rupee name still has to trade 10 Cr/day.
    # Held at 250. Briefly raised to 350 on 2026-09-19 to keep small caps out,
    # then put back once it was measured properly: price is not what protects
    # you, and it was never doing that job.
    #
    # Swept against the FULL filter stack (spike, drawdown, typical move) over
    # 12 months on the whole universe, 15-session hold net of costs:
    #     0 -> +4.03%   150 -> +3.73%   250 -> +4.44%
    #   300 -> +4.50%   350 -> +4.63%   500 -> +3.77%
    # 250-350 is a flat plateau -- 0.19pp across it, inside the noise -- with
    # the curve falling away below 250 and above 400. Price itself correlates
    # -0.016 to -0.031 with forward return, i.e. nothing.
    #
    # What actually removes the collapse-prone names is min_drawdown_pct and
    # max_spike_ratio below. In the 250-350 band specifically, the worst 1-year
    # drawdown among survivors falls from -74% WITHOUT those filters to -48%
    # WITH them, and returns go from +2.41% to +3.34%. That is the small-cap
    # protection, measured directly, rather than inferred from share price.
    #
    # Price was never a proxy for that anyway: the account's single largest
    # loss is BSE at Rs.3,266 (-26,552, -9%), which any floor waves through,
    # while MILKYMIST at Rs.274 is +38%.
    #
    # The floor's real cost is that it is the ONLY filter a stock crosses by
    # GOING UP, so it admits cheap movers late. Of 247 big up-legs (>=25%) in
    # volatile, liquid stocks over 2 years, 51% began below 250 and 33% never
    # reached 250 at all; of those that did cross, the move was already up a
    # median 27.5%. BLISSGVS was listed after 30% of a +177% run, VIYASH after
    # 43% of a +63% run -- and VIYASH's one genuine BUY, at 230.40, was thrown
    # away for being cheap. Raising the floor deepens that; 250 is the point
    # where the return curve stops paying for it.
    min_price: float = 250.0
    max_price: float = 1400.0
    # NOT raised, despite the intuition that more volume is safer. Measured, it
    # is the wrong direction: at a 350 floor, tightening this to 20/30/50/100 Cr
    # moved returns +2.14% -> 1.62 -> 1.22 -> 1.34 -> 0.89, monotonically worse,
    # and corr(traded value, forward return) is -0.042. Heavily traded stocks
    # are more efficiently priced, so there is less left to capture. 10 Cr is
    # here to guarantee you can get in and out, and that is all it should do.
    min_avg_traded_value_cr: float = 10.0

    # Volatility identity (what makes a stock "volatile")
    #
    # 180 -> 60 on 2026-09-22. This is the change that addresses the blind spot
    # this file has documented since the beginning: "a stock that became
    # volatile only 30 days ago has its 180-day average diluted by 150 quiet
    # sessions, so it reads as calm and is rejected."
    #
    # It is not an abstract concern. ANTELOPUS, one of the four target trades,
    # was blocked at EVERY point before its +56.2% leg (585.90 -> 915.30). At
    # the leg start its typical move read 5.6 on a 180-day window -- far below
    # the 14 gate -- while the same measurement over the most recent sessions
    # read 31.8 at 60 days and 38.9 at 50. It was already moving hard; the long
    # window averaged that against six quiet months until it disappeared.
    #
    # Swept over 12 months on the full universe, 15-session hold, net, entry at
    # next open, with the lookahead bug that flattered earlier runs removed:
    #   180d (sample>=60)  n=1168  ret +1.92%  top5 +1.55%  halves 3.26 / 0.69
    #    90d (sample>=30)  n=1982  ret +1.59%  top5 +3.23%  halves 1.76 / 1.43
    #    60d (sample>=20)  n=2456  ret +2.25%  top5 +4.42%  halves 2.83 / 1.69
    #    50d (sample>=17)  n=2624  ret +2.32%  top5 +6.18%  halves 2.97 / 1.70
    # Shorter is better on every column AND the two disjoint ticker halves come
    # closer together -- 180d has the widest split of any setting, which is what
    # an edge resting on one half looks like. 50 measured best; 60 is chosen
    # because it keeps more windows behind each median and this file has a
    # history of the most extreme setting being the one that had to be undone.
    #
    # Known limit, stated so nobody re-litigates it: this does NOT rescue
    # AEGISLOG, the +142.4% trade. It was genuinely dormant before that move --
    # typical move 1.4-2.9 at every lookback from 40 to 180 days. No window
    # length sees a stock that has not moved yet, and any gate loose enough to
    # admit it would admit most of the market.
    volatility_lookback: int = 60
    range_event_threshold: float = 3.5
    close_move_threshold: float = 2.5
    min_avg_volatility: float = 3.5
    min_median_volatility: float = 2.5
    # Scaled 30 -> 12 WITH the lookback, to hold the gate's behaviour fixed
    # rather than its number. These two gates are redundant by design and only
    # the tighter one ever binds: at a 180-day window the ratio demanded 36 days
    # against this field's 30, so the RATIO bound. Leaving 30 in place at a
    # 60-day window would silently flip that -- 30 of 60 sessions is 50%, two
    # and a half times stricter than the 20% ratio -- and quietly become the
    # strictest gate in the file. 12 is 0.20 * 60, which keeps the ratio binding
    # exactly as before.
    min_volatile_days: int = 12
    min_volatility_ratio: float = 0.20
    # Kept at 180 deliberately, and NOT tied to volatility_lookback.
    #
    # measure_spike_ratio reads this. It is a p95/median ratio, and a p95 drawn
    # from 60 observations is a far shakier statistic than one drawn from 180 --
    # the 95th percentile of 60 points is essentially the third-largest value.
    # The 4.5 threshold was calibrated on a 180-day window, so shortening the
    # window would re-calibrate a gate nobody asked to change and quietly alter
    # which stocks it rejects.
    spike_lookback: int = 180

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
    # the old Persistence_Rate (now retired) correlated -0.080 with the
    # 30-session forward peak,
    # while a stock's own history of large moves correlated +0.158 and produced
    # a monotonic gradient (bottom quintile 7.6% hit rate, top 23.8%). It also
    # survived a control for volatility -- within a FIXED volatility band the
    # low-move-rate tercile stayed worst in all three bands -- so it is not
    # just "volatile stocks are volatile" restated.
    # 180 -> 60, alongside volatility_lookback. See that field for the sweep and
    # for the ANTELOPUS case that motivated it: the same stock reads typical
    # move 5.6 over 180 days and 31.8 over 60, because the long window averages
    # a live move against six dead months.
    persistence_lookback: int = 60
    # 60 -> 20, and this one is arithmetic rather than preference: a 60-day
    # lookback can yield at most 60 forward windows, and eligibility (trailing
    # volatility already >= min_avg_volatility) removes more, so a floor of 60
    # would reject almost everything and the lookback change would look like it
    # had broken the scanner.
    #
    # 20 is one third of the window, the same proportion the old 60-of-180 was.
    # The honest cost: a median built from 20 windows is noisier than one from
    # 76 (the old measured median). That is the price of seeing a move while it
    # is happening instead of six months after.
    min_persistence_sample: int = 20
    # Forward window the move profile measures over. 30 sessions ~ 6 weeks,
    # which is the timescale the winning trades actually played out on
    # (SHILPAMED reached +20.7% in 8 sessions and kept running to +58.8%).
    move_horizon_days: int = 30
    # Gate on typical move SIZE, not on a pass/fail rate. This is the
    # "capture great ones" filter: it selects stocks that habitually travel,
    # rather than rejecting those that fail to clear an arbitrary bar.
    # Calibrated below against the live universe.
    # Raised 10 -> 14 on 2026-09-19. The single biggest contributor to the
    # filter stack: dropping it costs 0.80pp, more than any other gate.
    #
    # Swept over 12 months on the full universe, 15-session hold net of costs,
    # requiring BOTH independent halves of the period to agree at every step:
    #   gate 10 -> +3.07% (H1 +1.29, H2 +4.60)
    #   gate 12 -> +3.79% (H1 +1.87, H2 +5.04)
    #   gate 14 -> +4.30% (H1 +2.40, H2 +5.61)
    #   gate 15 -> +4.56% (H1 +2.40, H2 +5.92)
    #   gate 16 -> +4.15% (H1 +1.02, H2 +5.85)   <- degrades
    # Smooth and monotonic to 14-15, then it turns over, which is what an
    # honest optimum looks like rather than a single lucky spike. 14 is chosen
    # over 15 to keep more candidates at nearly identical return.
    #
    # This is also the gate that earns Stage 3 its place at all: Stage 1 alone
    # returned +0.66%, Stage 1+2 +1.19%, and Stage 1+2+3 +3.07%.
    min_typical_move_pct: float = 14.0
    # Gain/pain and tail drawdown are NOT gates. They are scored (see
    # calculate_score), so a stock with an ugly tail ranks low instead of
    # disappearing.
    #
    # They were gates briefly and it was a mistake: min_move_ratio at 3.0
    # excluded ANTELOPUS (2.50) and a -12% tail floor excluded SHILPAMED
    # (-15.8%) -- the two trades held up as the target pattern, each killed by
    # a different gate. A filter that removes the thing you are looking for is
    # miscalibrated, and tightening it further only hides more.
    #
    # The scanner's job is the simple one it started with: find good stocks
    # that can move. Whether a mover is worth YOUR money is a ranking question,
    # and the risk side belongs in the ranking rather than in a silent veto.
    move_stability_threshold: float = 4.0

    # --- Character filters: WHAT KIND of volatility, not how much ---
    #
    # Added 2026-09-19. Everything above asks how far a stock travels. These
    # two ask whether the travelling is the sort you can hold through, which is
    # the job the price floor was being asked to do and never did.
    #
    # max_spike_ratio: 95th-percentile daily move divided by the median daily
    # move, over volatility_lookback. A stock whose volatility is spread across
    # most sessions scores low; one that sits still and then gaps 15% on news
    # scores high. The second kind satisfies every volatility gate in Stage 2
    # while being untradeable -- you cannot enter a gap.
    #
    # Set to 4.0, NOT the 3.5 the first sweep suggested. Swept on top of the
    # rest of the stack, with the period split into two disjoint TICKER halves:
    #   no filter -> +3.86% (n=1530)  halves 3.99 / 3.77
    #   <=4.5     -> +4.19% (n=1292)  halves 5.14 / 3.52
    #   <=4.0     -> +4.63% (n=1022)  halves 5.78 / 3.86
    #   <=3.5     -> +4.86% (n= 488)  halves 5.75 / 4.12
    #   <=3.0     -> +5.57% (n= 155)  halves 10.93 / 4.07   <- overfit, ignore
    #
    # SETTLED AT 4.5, after the threshold had to be loosened twice -- which is
    # itself the finding. 3.5 rejected ANTELOPUS (3.55); 4.0 then rejected DYCL
    # (4.18). Those are two of the four trades this scanner exists to find,
    # each lost to a gate by less than two tenths of a point. A cutoff that has
    # to be moved every time it meets a known-good stock is not measuring a
    # real boundary, it is being fitted to noise.
    #
    # The FILTER is real; the exact cutoff is not well determined:
    #   no filter -> +3.86%   <=4.5 -> +4.19%   <=4.0 -> +4.63%   <=3.5 -> +4.86%
    # 4.5 keeps roughly half the measured benefit (+0.33pp over no filter
    # against +0.77pp at 4.0). That is the price of not rejecting DYCL, and it
    # is worth paying: a locked model that excludes a stock the owner made 21%
    # on will not be trusted or followed, and an untrusted list is worth zero
    # regardless of its backtest.
    #
    # 4.5 still sits well above the qualifying pool's MEDIAN of 3.72, so it
    # removes the genuinely gap-driven tail rather than cutting into the middle
    # of the population -- which is the correct shape for a safety filter.
    #
    # All four target trades now clear it: ANTELOPUS 3.55, SHILPAMED 3.84,
    # AEGISLOG 3.28, DYCL 4.18.
    max_spike_ratio: float = 4.5
    # Reject anything that has already fallen this far from a peak within
    # drawdown_lookback. Not a prediction -- a stock that halved in the last
    # year has demonstrated it can halve. Worth 0.32pp in the ablation, and it
    # improves the MEDIAN and win rate more than the mean, which is exactly
    # what "stop me being slammed" should look like.
    min_drawdown_pct: float = -50.0
    drawdown_lookback: int = 250
    # A median/average volatility ratio was tested alongside these as a
    # "steadiness" measure and DROPPED: it is redundant with max_spike_ratio
    # (both measure dispersion of daily moves) and adding it on top changed
    # returns by -0.08pp, i.e. slightly negative. Two filters for one idea is
    # one filter too many.

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
    # Absolute ceiling on stop distance -- a backstop against an absurd stop,
    # NOT a risk preference.
    #
    # It must be loose, because stop distance and volatility are the SAME
    # variable: the stop is 1.5x ATR below entry, so a volatile stock
    # necessarily needs a wide one. Set too tight, this gate rejects stocks for
    # being volatile in a scanner whose entire purpose is finding volatile
    # stocks.
    #
    # That is not hypothetical. At 8.0 it rejected 7 of the 8 most volatile
    # stocks in the universe, and for most of them it was the ONLY failing
    # gate: STLTECH (typical move 48.9%, risk 9.1%), DEEDEV (28.8%, 8.2% --
    # over by two tenths of a point), AEROFLEX (27.9%, 8.8%), INDSWFTLAB
    # (27.1%, 8.8%), ANTELOPUS (22.9%, 13.9%). Candidates clustered in the
    # SECOND volatility decile rather than the first.
    #
    # 8.0 was never a considered choice for this role. It sat inert for the
    # scanner's whole life because rr_floor_stop pinned every stop at exactly
    # target_pct/min_rr = 3% (384 of 384 stored rows had Risk_Pct == 3.00).
    # Removing rr_floor_stop made it live for the first time, at a value chosen
    # when it could never bind.
    #
    # 15.0 admits the whole volatile cohort while still catching a genuinely
    # broken stop. Per-trade risk is controlled by POSITION SIZE, not by
    # refusing the stock: a 14% stop on Rs.50,000 risks Rs.7,000, or Rs.3,500
    # on a Rs.25,000 position. Risk_Pct is in the output for exactly that.
    # VALIDATED 2026-09-19 and deliberately left alone. Under the locked filter
    # stack it is INERT: across 1,540 qualifying observations the risk_pct
    # distribution runs p50 6.9%, p90 9.3%, p99 11.5% and a MAXIMUM of 13.6%,
    # so a 15% ceiling removed exactly 0 rows. It is a backstop against an
    # absurd stop, not a live filter, which is the role intended for it.
    #
    # That also resolves the standing objection that this is a hard reject in a
    # file whose philosophy is "risk ranks, it does not gate": it does not
    # actually gate anything any more, so there is nothing to convert.
    #
    # Not tightened. Stop distance does not predict return here
    # (corr -0.039, and the widest quintile is noisy rather than uniformly
    # worse), and dropping to 8.0 would remove 26% of candidates for +0.64pp --
    # the exact trade the comment above records as a mistake the first time.
    max_risk_pct: float = 15.0

    # target_pct / max_extension_days / max_hold_days / min_rr were deleted
    # here. They defined the retired rule -- buy, never sell below a +3%
    # floor, extend while still closing higher, give up after 5 sessions --
    # which the scanner no longer models at all. It measures how far a stock
    # travels over move_horizon_days and leaves the exit to a human.
    # position_check.py still replays that rule and now carries its own
    # LEGACY_* constants for it.

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
    # purpose: min_typical_move_pct is the gate that decides membership, and a
    # rescale should change the ORDER and SPREAD of the list, not silently
    # change which stocks appear.
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
    # STORAGE cap -- how many rows reach BigQuery. Deliberately high: the
    # stored history is the raw material for the signal-history test (does a
    # BUY at score 60+ actually beat one at 30-40?), and that test needs the
    # LOW scores too. Truncating storage would permanently destroy the ability
    # to answer it.
    top_n: int = 200
    # DISPLAY cap -- how many rows are printed and pushed to Telegram.
    #
    # This is the attention knob, and it is deliberately NOT a filter. A live
    # scan returns ~100 candidates; that is a list you skim, not one you act
    # on. Cutting the list with min_score or the price band would also cut the
    # stored history, and cutting by price is actively wrong -- it is
    # orthogonal to quality (entry price correlated -0.030 with outcome), so a
    # 400-1500 band would have deleted GANDHAR at score 71.1, PFOCUS at 64.7,
    # INDSWFTLAB at 64.2 and CUPID at 64.2 purely for being cheap, while still
    # leaving 76 names to read.
    #
    # Capping by RANK keeps the best N whatever they cost. On the live scan the
    # top 15 spanned scores 60.0-71.4 and prices Rs.275-1,169.
    display_top_n: int = 50
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.min_typical_move_pct <= 0:
            raise ValueError(
                f"min_typical_move_pct must be > 0 (got {self.min_typical_move_pct})"
            )
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

    # FIX: drop Yahoo's fabricated holiday bars -- zero volume means no trade
    # took place, so this is not a session for this stock and it must not be
    # averaged in as one. Kite confirms no such bar exists. See the
    # require_current_bar comment in ScannerConfig for how these are produced.
    #
    # Measured cost of keeping them, across 19 tickers over 2 years:
    #   ATR              understated  5.7% on average, 11.3% worst
    #   traded value     understated  6.1% on average, 41.9% worst (INDIAGLYCO)
    #   avg_volatility   understated  1.8% on average
    #   Volume_Spike     INDIAGLYCO read 6.39x against a true 4.39x -- +46%
    #
    # Every one of those errors points the same way. A zero-volume day is a
    # maximally calm day (range_pct 0, close_move 0) and a zero-volume sample
    # in the 20-day mean, so it drags avg_volatility, volatility_ratio, ATR and
    # traded value DOWN -- pushing stocks toward the wrong side of four Stage
    # 1/2 gates -- while dragging avg_volume20_prior down too, which inflates
    # the next day's Volume_Spike. Understated ATR also tightens the 1.5x ATR
    # stop, so Risk_Pct was reported smaller than the stop actually is.
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
def run_move_profile(
    data: pd.DataFrame, config: ScannerConfig
) -> tuple[float, int, float, float, float]:
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

    Returns: (typical_move_pct, sample_size, typical_drawdown_pct, move_stability,
              tail_drawdown_pct)
      - typical_move_pct: MEDIAN best gain reached within the horizon. Median,
        not mean, because one 300% window would otherwise define a stock.
      - typical_drawdown_pct: MEDIAN worst loss within the same windows
        (negative). What you would have had to sit through.
      - move_stability: std of typical_move_pct across the 4 sub-periods. Low
        means the stock behaves consistently; high means it had one good spell.
      - tail_drawdown_pct: the 10th-percentile trough -- a bad window, not a
        typical one. This exists because typical_drawdown_pct is a MEDIAN and a
        collapse does not live in the median. Measured on the 29 candidates the
        live gates produce, median drawdown was -3.5% while the p10 tail was
        -12.2% and the worst single candidate reached -30.7%. GANDHAR showed a
        -1.4% median against a -23.2% worst; E2E a -5.1% median against -30.7%.
        On a 50,000 INR position that is the difference between losing 1,700
        and losing 15,300.
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
    # While the session is open the volume requirement is WAIVED, not failed.
    # Requiring 2.3x of a partial count is requiring something arithmetically
    # unavailable before 15:30, which is why the midday runs produced almost no
    # breakouts. Waiving it means the price condition -- a close above the 20-day
    # high, on a bullish bar -- stands on its own, and the reason string says
    # plainly that volume has not confirmed it yet.
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
# Score point budget. The positive weights sum to 100 so the number reads as a
# percentage-like figure; SUPPORT_PENALTY_PTS is subtracted on top.
#
# Two rules produced these, both learned the hard way:
#
# 1. NEVER SCORE SOMETHING THE GATES ALREADY ENFORCE. An earlier budget spent
#    38 of its 100 points on avg_volatility, median_volatility,
#    volatility_ratio and today_range -- all four already hard gates. Scoring a
#    filter you have applied cannot rank the survivors, because every one of
#    them passed it. Measured: those four averaged 27.2 points, 60% of a
#    typical score of 45, while barely varying. That alone is why every score
#    used to land between 40 and 60 (mean 44.97, sd 4.01).
#
# 2. RISK BELONGS IN THE RANKING, NOT IN A VETO. TAIL_PTS is large on purpose.
#    It replaced hard gates that were removing good stocks outright -- a -12%
#    tail floor excluded SHILPAMED at -15.8%, a 3.0 gain/pain floor excluded
#    ANTELOPUS at 2.50. Ranking a risky mover below a clean one keeps it
#    visible; gating it out did not.
#
# Move size takes the largest share because it is the thing actually being
# looked for. Stability is deliberately small: it returns only 0, half or full
# (a 3-step flag, not a measurement), yet under an older budget that coarse
# flag drove 37.8% of all ranking variance.
#
# NOT A PREDICTIVE CLAIM. The score orders candidates by how well they fit what
# was measured. No version of it has been shown to predict forward returns --
# the previous one correlated -0.153 with realized outcome. That test needs a
# signal-history table, which does not exist yet.
MOVE_PTS = 40
UPSIDE_PTS = 15
TAIL_PTS = 20
STABILITY_PTS = 8
VOLUME_PTS = 5
SETUP_PTS = 12
SUPPORT_PENALTY_PTS = 10

# ANCHORS -- what counts as 0 and what counts as full marks for each component.
#
# Each component is scored between the MINIMUM ADMISSIBLE value and a genuinely
# excellent one, never between 0 and a theoretical ideal that cannot occur. The
# earlier version scored persistence from a 100% win rate that no real stock
# reached, so half the budget was unreachable while the first quarter of it was
# handed free to anything that cleared the gate. Measured consequence: every
# score landed between 40 and 60 (mean 44.97, sd 4.01 on a 100-point scale).
#
# Move size anchors from min_typical_move_pct, the gate, because clearing the
# gate is the price of admission rather than an achievement. Volume anchors
# from 1.0x for the same reason: 1x IS the average, and scoring it from zero
# paid a third of that budget to a stock for being unremarkable.
#
# The budget itself: move 40, tail 20, upside 15, setup 12, stability 8,
# volume 5. Tail carries real weight deliberately -- it is the only term that
# expresses risk, and it replaced a hard gate that was excluding good stocks
# outright (a -12% tail floor removed SHILPAMED at -15.8%). Ranking a risky
# mover below a clean one keeps it visible; gating it out did not.
#
# NOT A PREDICTIVE CLAIM. The score orders candidates by how well they fit what
# was measured; it is not a forecast, and no version of it has been shown to
# predict forward returns. That test needs a signal-history table, which does
# not exist yet.
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

    typical_move, sample_size, typical_drawdown, move_stability, tail_drawdown = run_move_profile(data, config)
    if sample_size < config.min_persistence_sample:
        return None
    # The one gate here: does it actually travel? Asymmetry and tail drawdown
    # are scored, not gated -- see the config comment above.
    if typical_move < config.min_typical_move_pct:
        return None
    upside_dominance = upside_rate(data, config)
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

    # A near-total per-ticker failure rate is not a data condition, it is a
    # broken build, and it must not be allowed to read as "0 candidates found".
    # This is exactly how a field rename (support_distance_threshold ->
    # support_distance) took the scanner down: every one of 2,574 tickers
    # raised AttributeError inside calculate_score, the broad except below
    # swallowed all of them, and five consecutive scheduled runs reported zero
    # results with nothing worse than a warning in the log.
    # Two independent triggers, because a code fault does not always show up as
    # a HIGH failure rate.
    #
    # The rate trigger catches a total failure -- the support_distance_threshold
    # rename made all 2,574 tickers raise, which surfaced as "0 candidates
    # found" and a warning.
    #
    # The type trigger catches a PARTIAL one, which the rate alone misses.
    # Deleting max_risk_pct broke only the tickers that got far enough to reach
    # the risk check: 16 of 400, just 4%, silently under the 25% threshold, and
    # the run reported zero candidates as though that were a market condition.
    # AttributeError/NameError/TypeError essentially never come from bad market
    # data -- they mean the code referenced something that does not exist -- so
    # a handful of them is already proof of a broken build.
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
    # Score is the ranker, so it must also be the sort key. It previously sat
    # FOURTH, behind what were then Persistence_Rate and Persistence_Stability
    # (now Upside_Dominance_Pct / Move_Stability), which meant it
    # almost never affected the order at all: on a live 84-candidate run only
    # 6 rows shared an (Action, win-rate, stability) tuple
    # for Score to break, the displayed position correlated -0.366 with Score,
    # and the top 8 shown shared just 3 names with the top 8 by Score. The
    # report therefore ranked by one number while printing another beside it.
    # The expectancy measure is still an output column and still drives 50 of
    # Score's 100 points, so it keeps most of its influence -- explicitly
    # rather than accidentally.
    # Ranked by Expected_Move, NOT by Score. Score is still computed and stored,
    # but it stopped being the sort key on 2026-09-19 because it does not
    # predict: across 991 stored signals its correlation with forward return was
    # -0.065, it was positive on only 4 of 15 bar dates, and the 60+ bucket
    # (+0.98%) underperformed the 40-50 bucket (+2.84%). The three trades held
    # up as the target pattern -- ANTELOPUS, SHILPAMED, AEGISLOG -- averaged
    # 48.1, BELOW the 49.0 population mean. It never identified them.
    #
    # Worse, it moved the wrong way during the moves that mattered. Through
    # BLISSGVS's +177% run it scored 7.9 and 9.2 out of 100; through E2E's +85%
    # run its score FELL from 52.2 at the bottom to 38.9 midway. With
    # display_top_n = 10 against ~100 candidates, ranking by Score is what
    # buried both of them.
    #
    # Expected_Move is the honest replacement: it is the one measured quantity
    # that correlated positively and repeatably with forward return (+0.126 to
    # +0.143 across samples), and it is already what min_typical_move_pct gates
    # on, so the list is now ordered by the same thing that decides membership.
    # Sorted PURELY by Expected_Move. Action is deliberately NOT a sort key any
    # more -- BUY no longer floats to the top.
    #
    # It had to go the moment this list became something to act on top-down.
    # Measured over 605 stored signals at a 10-session horizon:
    #     BUY    n=163   +0.33%   median -1.04%   win 45.4%
    #     WATCH  n=442   +2.75%   median +0.80%   win 52.9%
    # Sorting BUY first therefore handed the reader the WORSE half of the list
    # first. By setup the same inversion: extended +4.30% and deep_pullback
    # +3.65% are the best, while pullback_bounce (+0.02%) and breakout (+0.74%)
    # -- the only two setups that produce a BUY -- are the worst.
    #
    # Action is still computed and still stored, because it describes today's
    # price structure and that is worth recording. It just must not decide what
    # you look at first until it is shown to predict something, which it
    # currently does not.
    df = df.sort_values(["Expected_Move", "Score"], ascending=[False, False])

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


# =========================================================
# SHORTLIST -- the names to actually act on
# =========================================================
# The full scan returns ~73 candidates. That is a research output, not a
# decision, and a list nobody can act on is worth the same as no list. This
# cuts it to a number a person can hold positions in.
#
# Measured over 2,383 qualifying observations, 247 tickers, 108 decision dates
# (2025-12-09 .. 2026-08-06), entry at next open, net of costs, ranked by
# Expected_Move exactly as scan_tickers already sorts. ret15 / ret30 are mean
# returns at those holding periods; halves are two disjoint TICKER halves:
#     N     trades   ret15   ret30   win30   halves (15s)
#      1       108   +3.81   +9.57   62.0%   -0.08 / 8.50   <- one half negative
#      3       282   +4.88   +8.56   60.6%    4.82 / 4.95
#      5       411   +4.05   +7.01   57.9%    4.26 / 3.81
#     10       628   +3.69   +5.75   55.4%    3.30 / 4.07   <- set here
#     15       781   +3.53   +5.98   56.3%    3.96 / 3.15
#     25      1026   +3.32   +5.82   56.2%    3.77 / 2.83
#
# SET TO 10 BY OWNER PREFERENCE, and the cost is real and stated: 10 returns
# +3.69% against 3's +4.88% at 15 sessions, and +5.75% against +8.56% at 30.
# Roughly a third of the measured edge, given up for seven more names to
# choose among. That is a legitimate trade -- more names means capital spread
# across more positions, which cuts single-name risk, and the account's worst
# damage to date came from concentration (BSE alone was 30.8% of capital and
# -26,552). Nothing here measures position sizing, so the backtest cannot see
# that benefit; it only sees the per-trade return going down.
#
# 10 does hold up on the ticker split (3.30 / 4.07) better than 1 does
# (-0.08 / 8.50), so it is not a fragile setting -- just a less sharp one.
#
# DELIBERATELY NO EXTRA GATE. Three were tested on top of this ranking and all
# three made it worse, which is the opposite of the intuition (figures at N=3,
# where they were swept):
#     top3, no filter          +4.88%   halves  4.82 / 4.95
#     + tail drawdown >= -12%  +2.38%   halves  3.27 / 1.48
#     + tail drawdown >= -10%  +2.19%   halves -0.37 / 4.78
#     + move stability <= 20   +4.14%   halves  5.27 / 3.12
#     + move stability <= 15   +3.59%   halves  3.95 / 3.07
# An ABSOLUTE tail floor is the wrong shape: a -17% tail on a stock that
# travels 50% is not the same risk as a -17% tail on one that travels 15%, and
# a flat floor scores them identically. It removes the big movers, which is
# exactly where the return is.
#
# The RELATIVE version (Expected_Move / |Tail_Drawdown_Pct|) does look better
# -- +6.50% at >= 2.0, and both ticker halves improve (5.54 / 7.68). It is NOT
# applied, because it fails the other split: by TIME it returned +8.87% in the
# first half of the period against +2.65% in the second, where unfiltered ran
# 5.70 / 3.27. So it helped in one regime and slightly hurt in the other, on
# 189 trades. That is not enough to gate on. It is reported as a COLUMN
# instead, so the pain is visible without being acted on. Revisit when
# signal_outcomes carries real forward data (~December).
SHORTLIST_N = 10
# How far back to count previous appearances. ~45 calendar days covers roughly
# 30 sessions, the same horizon the move profile measures over.
STREAK_LOOKBACK_DAYS = 45


def lookup_list_streak(tickers: list[str], project_id: str, dataset_id: str,
                       table_id: str, bar_date: str) -> dict[str, int]:
    """How many earlier scans put each ticker in the top SHORTLIST_N.

    This exists because repeat appearances turned out to be a POSITIVE signal,
    which is the opposite of how a recurring name usually reads. Measured on
    the top-10 list, mean 30-session return by how many times the name had
    already appeared:
        1st appearance   n= 90   +4.58%   win 47.8%
        2nd              n= 73   +5.26%   win 53.4%
        3rd              n= 64   +8.61%   win 56.2%
        4th-6th          n=144   +8.34%   win 64.6%
        7th or later     n=257   +4.14%   win 53.3%
    So a name on its 3rd to 6th scan is the best of them -- roughly double a
    first appearance -- and a familiar name is not a used-up one.

    Note the DECAY after the 6th. At N=3 that tail-off did not appear (4th+
    held at +9.41%), so it shows up only once the list is wide enough to carry
    names that have drifted down the ranking and are sitting near the cut. Read
    a count above ~6 as neutral rather than as more of a good thing.

    Read all of it as weak evidence, not a rule. Those rows come from only 90
    distinct tickers, so they are far from independent observations, and the
    whole sample is 8 months of one market. It is enough to stop treating a
    familiar name as stale. It is not enough to size a position on.

    Best-effort by design: returns {} on any failure. A missing streak column
    must never cost you a scan that already succeeded.
    """
    if not tickers:
        return {}
    try:
        client = bigquery.Client(project=project_id)
        query = f"""
        WITH ranked AS (
          SELECT Ticker, Bar_Date,
                 ROW_NUMBER() OVER (
                   PARTITION BY Run_Timestamp ORDER BY Expected_Move DESC, Score DESC
                 ) AS rn
          FROM `{project_id}.{dataset_id}.{table_id}`
          WHERE Bar_Date >= DATE_SUB(@bar, INTERVAL @days DAY)
            AND Bar_Date < @bar
        )
        SELECT Ticker, COUNT(DISTINCT Bar_Date) AS appearances
        FROM ranked
        WHERE rn <= @n AND Ticker IN UNNEST(@tickers)
        GROUP BY Ticker
        """
        job = client.query(query, job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("bar", "DATE", bar_date),
            bigquery.ScalarQueryParameter("days", "INT64", STREAK_LOOKBACK_DAYS),
            bigquery.ScalarQueryParameter("n", "INT64", SHORTLIST_N),
            bigquery.ArrayQueryParameter("tickers", "STRING", tickers),
        ]))
        return {row["Ticker"]: int(row["appearances"]) for row in job.result()}
    except Exception as exc:
        LOGGER.warning("Shortlist streak lookup failed (non-fatal): %s", exc)
        return {}


def _ordinal(n: int) -> str:
    """2 -> '2nd'. 11-13 are the exceptions that a bare suffix table gets wrong."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def build_shortlist(candidates: pd.DataFrame, streaks: Optional[dict[str, int]] = None,
                    n: int = SHORTLIST_N) -> pd.DataFrame:
    """The top n of the already-ranked list, plus the two decision columns.

    Adds nothing to the ranking -- `candidates` arrives sorted by Expected_Move
    and this takes the head of it. What it adds is the context needed to act on
    a name rather than merely read it: how much pain per unit of move
    (Pain_Ratio) and how long it has been on the list (Days_On_List).
    """
    short = candidates.head(n).copy()
    if short.empty:
        return short
    short["Pain_Ratio"] = (
        short["Expected_Move"] / short["Tail_Drawdown_Pct"].abs().clip(lower=0.01)
    ).round(1)
    streaks = streaks or {}
    short["Days_On_List"] = [int(streaks.get(t, 0)) + 1 for t in short["Ticker"]]
    return short


def format_shortlist(short: pd.DataFrame, config: ScannerConfig) -> list[str]:
    """Console block. Deliberately the same content the Telegram message sends,
    so the phone and the terminal can never disagree about what to buy."""
    if short.empty:
        return ["SHORTLIST: nothing qualified today."]
    lines = [
        "=" * 78,
        f"SHORTLIST -- the {len(short)} to act on. Hold ~{config.move_horizon_days} sessions.",
        "=" * 78,
        f"{'':<3}{'Ticker':<13}{'Action':<8}{'Buy below':>10}{'Stop':>9}{'Move%':>7}"
        f"{'Pain':>6}{'Scans':>6}  Note",
    ]
    for i, (_, r) in enumerate(short.iterrows(), start=1):
        ticker = str(r["Ticker"]).replace(".NS", "")
        note = (f"{_ordinal(int(r['Days_On_List']))} scan on the list"
                if r["Days_On_List"] > 1 else "new today")
        lines.append(
            f"{i:<3}{ticker:<13}{r['Action']:<8}{r['Entry']:>10.2f}{r['Stop_Loss']:>9.2f}"
            f"{r['Expected_Move']:>6.1f}%{r['Pain_Ratio']:>6.1f}"
            f"{r['Days_On_List']:>6}  {note}"
        )
    lines.append("")
    lines.append("Pain = Move% / tail drawdown -- higher is a smoother ride. Shown, "
                 "not filtered on: it failed a time split.")
    lines.append("Scans = times on this list in the last 45 days. 3rd-6th has done "
                 "BEST (+8.5% over 30 sessions against +4.6% for a first); past "
                 "the 6th it flattens back out.")
    return lines


def format_telegram_digest(candidates: pd.DataFrame, run_date: str,
                           short: Optional[pd.DataFrame] = None,
                           horizon_days: int = 30) -> Optional[str]:
    """
    Builds the notification text: the shortlist, which is the same thing the
    console prints at the top. Returns None only when the scan found nothing
    at all.

    Includes the actual run time (IST, not just the date) in the header --
    with 4 scheduled runs a day, the same ticker can legitimately appear in
    more than one, and "9:07 AM run" vs "3 PM run" for the same name is
    very different information: was this signal just caught, or has it
    already had hours to move since it first showed up.
    """
    if candidates.empty:
        return None

    # Sends the SHORTLIST -- the same three names the console prints, in the
    # same order, with the same stops. Not the top 10, not fresh BUYs.
    #
    # The history of this function is a history of sending the wrong rows. It
    # began as (Action == BUY and Setup_Age_Days == 1), and both halves of that
    # selected badly: BUY returned +0.33% against WATCH's +2.75%, and the two
    # setups that can produce a BUY are the worst two of the seven. Through
    # BLISSGVS's +177% run and E2E's +85% run the scanner said WATCH every day,
    # so Telegram said nothing at all. It then sent the top 10 by Expected_Move,
    # which was honest but still a list to study rather than a decision.
    #
    # Three names, a price to buy below, and a stop. Everything else -- score,
    # setup, upside dominance, the other 70 candidates -- is in the console
    # output and in BigQuery for whoever wants to dig. This is the 5-second
    # phone read that a position can be opened from.
    run_time_ist = (dt.datetime.now(dt.timezone.utc) + IST_OFFSET).strftime("%Y-%m-%d %H:%M IST")

    lines = [f"Swing scan {run_time_ist}", ""]
    if short is None or short.empty:
        lines.append("Nothing qualified today.")
        return "\n".join(lines)

    # Two lines per name, not three. At SHORTLIST_N = 10 a third line would
    # push this past 30 lines, which is a scroll rather than a glance, and the
    # whole point of this message is that it can be read at a traffic light.
    width = max(len(str(t).replace(".NS", "")) for t in short["Ticker"])
    for n, (_, r) in enumerate(short.iterrows(), start=1):
        ticker = str(r["Ticker"]).replace(".NS", "")
        streak = (f"  ({_ordinal(int(r['Days_On_List']))} scan)"
                  if r["Days_On_List"] > 1 else "")
        lines.append(f"{n:>2}. {ticker:<{width}}  {r['Action']}")
        lines.append(f"    {r['Entry']:.0f} -> {r['Typical_Move_Price']:.0f}"
                     f", stop {r['Stop_Loss']:.0f}{streak}")
    lines.append("")
    lines.append(f"Hold ~{horizon_days} sessions. {len(candidates)} candidates scanned.")
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
    print(f"Move profile: typical gain >= {config.min_typical_move_pct}% over "
          f"{config.move_horizon_days} sessions "
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
        # not listed: it no longer exists, and the stop is ATR/structural.
        #
        # --min-score is also not listed: at 5.0 it is already near-zero on a
        # 0-100 scale (see the field comment), so there is no lower value left
        # that means anything as a relaxation hint. min_typical_move_pct is
        # what actually decides membership now.
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
            display = candidates
    shown = len(display)
    display = display.head(config.display_top_n)

    # --- shortlist: computed before anything is printed, because it is the
    # --- output that matters and the 73-row table is the appendix to it.
    streaks = {}
    if not no_bq:
        try:
            streaks = lookup_list_streak(
                candidates.head(SHORTLIST_N)["Ticker"].tolist(),
                _resolve_project_id(project_id), dataset_id, table_id,
                candidates.iloc[0]["Bar_Date"],
            )
        except Exception as exc:
            LOGGER.warning("Streak lookup skipped: %s", exc)
    short = build_shortlist(candidates, streaks)

    print()
    for line in format_shortlist(short, config):
        print(line)

    print("\n" + "-" * 100)
    # This label must keep matching the sort in scan_tickers. It has been wrong
    # twice: it claimed "by score" after Score stopped being the key, then
    # "BUY first" after Action stopped being one. A header that describes a
    # different ordering than the rows below it is worse than no header, because
    # it is believed.
    print("RESULTS (ranked by Move% -- typical 30-session gain. Not by Score, not by Action)")
    print("Age = consecutive days this Setup has held. Action/Setup describe today's price")
    print("structure; they do NOT order the list and BUY has not been shown to beat WATCH.")
    print("-" * 100)
    header = (f"{'Ticker':<13} {'Action':<7} {'Setup':<16} {'Age':>4} {'Score':>6} {'Vol%':>6} "
              f"{'Upside%':>9} {'Stabil':>7} {'Move%':>8} {'TailDD':>8} {'RR':>5} "
              f"{'Entry':>9} {'SL':>9} {'MovePx':>9}")
    print(header)
    print("-" * 100)
    for _, r in display.iterrows():
        print(f"{r['Ticker']:<13} {r['Action']:<7} {r['Setup_Type']:<16} {r['Setup_Age_Days']:>4} "
              f"{r['Score']:>6.1f} {r['Avg_Volatility']:>5.1f}% "
              f"{r['Upside_Dominance_Pct']:>8.0f}% {r['Move_Stability']:>6.1f} "
              f"{r['Expected_Move']:>7.1f}% {r['Tail_Drawdown_Pct']:>7.1f}% {r['RR_Ratio']:>5.1f} "
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
                digest_text = format_telegram_digest(
                    candidates, run_date, short, config.move_horizon_days)
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
