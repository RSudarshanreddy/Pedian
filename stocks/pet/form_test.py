"""Replays form.py over the last 12 months, judged month by month. The rules are in README.md.

  model   : persistent movers in a healthy 4-7% pullback (the candidates)
  control : the same healthy pullback, but the leg was NOT persistent (emerging or burst)
            -- shows whether finding the mover adds anything
  ordinary: any liquid stock, bought at the next open, held 10 sessions

    python form_test.py      # uses the 4-year price cache in data/
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

import data
import form

COST, HOLD = 0.4, 10
START = "2025-10-01"


def trade(o, h, c, t, target, stop):
    """Buy at the open after t. Exit at the target (from the next day), a close below stop
    (sold at the next open), or the close of the HOLD-th session. None if it can't be judged."""
    e = t + 1
    entry = o[e]
    if entry >= target or entry <= stop:   # the pullback is already over, or broken, at the open
        return None
    pending = c[e] < stop
    for j in range(e + 1, e + HOLD):
        if j >= len(c):
            return None
        if pending:
            return j, entry, o[j], "stop"
        if o[j] >= target:
            return j, entry, o[j], "target"
        if h[j] >= target:
            return j, entry, target, "target"
        if j == e + HOLD - 1:
            return j, entry, c[j], "time"
        pending = c[j] < stop
    return None


def stats(t: pd.DataFrame) -> dict:
    win = t.ret > 0
    return {"trades": len(t), "win %": win.mean() * 100, "avg %": t.ret.mean(), "median %": t.ret.median(),
            "avg win %": t.ret[win].mean(), "avg loss %": t.ret[~win].mean(),
            "hit target %": (t.why == "target").mean() * 100, "stopped %": (t.why == "stop").mean() * 100,
            "days held": t.days.mean()}


def main() -> None:
    bars = data.load_bars()
    trades, ordinary = [], []
    shown = defaultdict(set)
    for sym, df in bars.items():
        o, h, l, c, v = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
        dates = df.index
        tv = pd.Series(c * v).rolling(20).mean().to_numpy() / 1e7
        hi = pd.Series(c).rolling(form.WINDOW).max().to_numpy()
        lo = pd.Series(c).rolling(form.WINDOW).min().to_numpy()
        busy = {"model": -1, "control": -1}
        first = int(np.searchsorted(dates, pd.Timestamp(START)))
        for t in range(max(first, form.WINDOW - 1), len(c) - HOLD - 1):
            if not (c[t] >= form.MIN_PRICE and tv[t] >= form.MIN_TRADED_CR):
                continue
            ordinary.append((dates[t], (c[t + HOLD] / o[t + 1] - 1) * 100 - COST))
            if hi[t] / lo[t] < 1.10:                       # no 10% leg is possible: noise
                continue
            w = slice(t - form.WINDOW + 1, t + 1)
            m = form.analyse(o[w], h[w], l[w], c[w], v[w])
            if m["form"] == "noise":
                continue
            group = ("model" if m["candidate"] else
                     "control" if m["healthy_pullback"] and m["form"] != "persistent" else None)
            if m["candidate"]:
                shown[dates[t]].add(sym)
            if group is None or t <= busy[group]:
                continue
            r = trade(o, h, c, t, m["peak"], m["halfway"])
            if r is None:
                continue
            j, entry, exit_px, why = r
            trades.append({"group": group, "sym": sym, "signal": dates[t], "entry": entry, "exit": exit_px,
                           "why": why, "days": j - t, "ret": (exit_px / entry - 1) * 100 - COST})
            busy[group] = j

    t = pd.DataFrame(trades)
    ordn = pd.DataFrame(ordinary, columns=["date", "ret"])
    pd.set_option("display.width", 220)
    print(f"\nsignals {START} to {t.signal.max().date()}\n")
    print(pd.DataFrame({g: stats(x) for g, x in t.groupby("group")}).T.round(1).to_string(), "\n")

    month = lambda s: s.dt.to_period("M")
    by = pd.DataFrame({
        "model avg %": t[t.group == "model"].groupby(month(t[t.group == "model"].signal)).ret.mean(),
        "model trades": t[t.group == "model"].groupby(month(t[t.group == "model"].signal)).size(),
        "control avg %": t[t.group == "control"].groupby(month(t[t.group == "control"].signal)).ret.mean(),
        "ordinary avg %": ordn.groupby(month(ordn.date)).ret.mean(),
    })
    by["model beat control"] = by["model avg %"] > by["control avg %"]
    by["model beat ordinary"] = by["model avg %"] > by["ordinary avg %"]
    print("MONTH BY MONTH")
    print(by.round(2).to_string())
    print(f"\nmodel ahead of control in {int(by['model beat control'].sum())} of {len(by)} months, "
          f"ahead of an ordinary stock in {int(by['model beat ordinary'].sum())} of {len(by)} months")

    days = pd.Series({d: len(s) for d, s in shown.items()}).reindex(
        sorted(set(ordn.date)), fill_value=0)
    keys = sorted(shown)
    new = [len(shown[keys[i]] - shown[keys[i - 1]]) for i in range(1, len(keys))]
    print(f"\ncandidates per day: median {days.median():.0f}, max {days.max()}, "
          f"days with none {(days == 0).mean() * 100:.0f}%, new names per day median {np.median(new):.0f}")


if __name__ == "__main__":
    main()
