"""The rules in README.md, and nothing else. Change them there first."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rules:
    v_pct: float = 10.0          # a V is a fall of this much, then a rise of this much
    lookback: int = 60           # both completed Vs within this many sessions
    min_traded_cr: float = 10.0  # 20-session average traded value, Rs crore
    buy_at: float = 0.25         # buy at floor + this share of the range
    sell_at: float = 0.75        # sell at floor + this share of the range
    stop_below: float = 0.25     # stop on a close below floor - this share of the range
    max_days: int = 15           # sessions held, counting the day of buying
    cost_pct: float = 0.4        # per round trip


@dataclass(frozen=True)
class Pivot:
    i: int          # bar of the swing high or low
    price: float
    kind: str       # "H" or "L"
    confirmed: int  # bar on which the reversal reached v_pct -- known from this close on


def swings(high: np.ndarray, low: np.ndarray, pct: float) -> list[Pivot]:
    """Alternating swing highs and lows, each dated by when it became certain.

    A low is only a low once price has risen pct% off it, and a high once price
    has fallen pct% from it. Until then it is just the lowest (or highest)
    price so far, which is all a trader standing on that day could know.
    """
    up, down = 1 + pct / 100, 1 - pct / 100
    pivots: list[Pivot] = []
    if len(high) == 0:
        return pivots
    hi, hi_i, lo, lo_i = high[0], 0, low[0], 0
    trend = 0  # 0 = no swing yet, 1 = rising from a low, -1 = falling from a high
    for i in range(1, len(high)):
        if trend == 0:
            if high[i] > hi:
                hi, hi_i = high[i], i
            if low[i] < lo:
                lo, lo_i = low[i], i
            if hi_i > lo_i and hi >= lo * up:
                pivots.append(Pivot(lo_i, lo, "L", i))
                trend, ext, ext_i = 1, hi, hi_i
            elif lo_i > hi_i and lo <= hi * down:
                pivots.append(Pivot(hi_i, hi, "H", i))
                trend, ext, ext_i = -1, lo, lo_i
        elif trend == 1:
            if high[i] > ext:
                ext, ext_i = high[i], i
            elif low[i] <= ext * down:
                pivots.append(Pivot(ext_i, ext, "H", i))
                trend, ext, ext_i = -1, low[i], i
        else:
            if low[i] < ext:
                ext, ext_i = low[i], i
            elif high[i] >= ext * up:
                pivots.append(Pivot(ext_i, ext, "L", i))
                trend, ext, ext_i = 1, high[i], i
    return pivots


def setup(piv: list[Pivot], k: int, t: int, close: float, traded_cr: float,
          used: set[int], r: Rules) -> dict | None:
    """Tomorrow's buy order if rules 1-5 hold at bar t's close, else None.

    piv[:k] are the pivots confirmed by bar t. They alternate, so when the
    latest is a high the four before read L1, H1, L2, H2 (oldest first).
    """
    if k < 4:
        return None
    h2, l2, l1 = piv[k - 1], piv[k - 2], piv[k - 4]
    if h2.kind != "H" or h2.i in used:                  # rule 4, one trade per dip
        return None
    if l1.i < t - r.lookback:                           # rule 2
        return None
    if l2.price < l1.price:                             # rule 3
        return None
    if close < l2.price:                                # rule 5
        return None
    if not traded_cr >= r.min_traded_cr:                # rule 1 (NaN fails too)
        return None
    floor, rng = l2.price, h2.price - l2.price
    return {
        "floor": floor, "ceiling": h2.price, "ceiling_bar": h2.i,
        "buy": floor + r.buy_at * rng,
        "target": floor + r.sell_at * rng,
        "stop": floor - r.stop_below * rng,
    }


def trades(df: pd.DataFrame, r: Rules = Rules()) -> list[dict]:
    """Every trade the rules would have made in one stock, oldest first."""
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    traded_cr = pd.Series(c * v).rolling(20).mean().to_numpy() / 1e7
    dates = df.index
    piv = swings(h, l, r.v_pct)
    out: list[dict] = []
    used: set[int] = set()
    k, order, pos = 0, None, None

    def close_out(t: int, px: float, why: str) -> None:
        pos.update(exit_date=dates[t], exit=px, reason=why,
                   ret=(px / pos["entry"] - 1) * 100 - r.cost_pct)
        out.append(pos)

    for t in range(len(df)):
        # The session: an open position trades first, else yesterday's order may fill.
        if pos is not None:
            pos["days"] += 1
            if pos.pop("stop_hit", False):
                close_out(t, o[t], "stop"); pos = None
            elif o[t] >= pos["target"]:
                close_out(t, o[t], "target"); pos = None
            elif h[t] >= pos["target"]:
                close_out(t, pos["target"], "target"); pos = None
            elif pos["days"] >= r.max_days:
                close_out(t, c[t], "time"); pos = None
            elif c[t] < pos["stop"]:
                pos["stop_hit"] = True
        elif order is not None:
            fill = o[t] if o[t] <= order["buy"] else order["buy"] if l[t] <= order["buy"] else None
            if fill is not None:
                used.add(order["ceiling_bar"])
                pos = {**order, "signal_date": dates[t - 1], "entry_date": dates[t],
                       "entry": fill, "days": 1, "traded_cr": traded_cr[t - 1]}
                if c[t] < pos["stop"]:
                    pos["stop_hit"] = True
        order = None

        # After the close: what is now confirmed, and tomorrow's order.
        while k < len(piv) and piv[k].confirmed <= t:
            k += 1
        if pos is None and t + 1 < len(df):
            order = setup(piv, k, t, c[t], traded_cr[t], used, r)

    if pos is not None:
        pos.pop("stop_hit", None)
        close_out(len(df) - 1, c[-1], "open")
    return out
