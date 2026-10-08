"""Movers in form, entered on a healthy pullback. The rules are in README.md; change them there first.

Scans every NSE stock using only its last 25 sessions:
  find the active leg -> judge its form -> keep the persistent movers
  -> READY: those in a healthy 4-7% pullback; WATCH: those still near their peak.
The daily list is 10 names: every READY name first (best leg first), then WATCH.
Telegram gets the names only; the details (targets, exits, buy zones) go to the log.

    python form.py        # downloads ~3 months for every NSE stock (a few minutes)

Deployed as the `form` Cloud Run service, called at 10:00 IST on weekdays; it
sends the list to the same Telegram chat as the momentum scanner. At 10:00 the
day's bar is only 45 minutes old, so a bar for today is dropped until after
the close -- the list is always judged on complete sessions, as in the test.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import urllib.request
from typing import Any

import numpy as np
import pandas as pd

import data

WINDOW = 25
MIN_PRICE = 100.0
MIN_TRADED_CR = 10.0
MAX_SHOWN = 30
LIST_SIZE = 10
PROJECT = "sudarshan-442212"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def analyse(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray) -> dict:
    """The leg, its form and the pullback since its peak, from one WINDOW of sessions (oldest first)."""
    p = int(np.argmax(c))
    if p == 0:
        return {"form": "noise"}
    s = int(np.argmin(c[:p + 1]))
    size = c[p] / c[s] - 1
    length = p - s
    m = {"start": c[s], "peak": c[p], "halfway": c[s] + (c[p] - c[s]) / 2, "size": size, "length": length}
    if size <= 0 or length == 0:
        return {**m, "form": "noise"}

    r = c[s + 1:p + 1] / c[s:p] - 1
    total = np.log(c[p] / c[s])
    m["persistence"] = float(np.mean(r > 0))
    m["biggest_day"] = float(np.max(np.log1p(r)) / total)
    m["gap_share"] = float(np.sum(np.maximum(0.0, np.log(o[s + 1:p + 1] / c[s:p]))) / total)
    leg = c[s:p + 1]
    m["inner_dip"] = float(np.min(leg / np.maximum.accumulate(leg)) - 1)
    m["volume"] = float(np.mean(v[s + 1:p + 1]) / np.mean(v)) if np.mean(v) > 0 else np.nan
    mid = s + length // 2
    m["acceleration"] = float((c[p] / c[mid] - 1) - (c[mid] / c[s] - 1)) if length >= 2 else np.nan

    pb_days = len(c) - 1 - p
    m["pullback"] = c[-1] / c[p] - 1
    m["pullback_days"] = pb_days
    if pb_days:
        m["pullback_worst_day"] = float(np.min(c[p + 1:] / c[p:-1] - 1))
        leg_vol = np.mean(v[s + 1:p + 1])
        m["pullback_volume"] = float(np.mean(v[p + 1:]) / leg_vol) if leg_vol > 0 else np.nan
        m["below_halfway"] = bool(np.any(c[p + 1:] <= m["halfway"]))
    else:
        m.update(pullback_worst_day=0.0, pullback_volume=np.nan, below_halfway=False)

    if size < 0.10:
        form = "noise"
    elif length < 5 or m["biggest_day"] >= 0.5 or m["gap_share"] >= 0.5:
        form = "burst"
    elif m["pullback"] < -0.12 or pb_days > 10 or m["below_halfway"]:
        form = "exhausted"
    elif size < 0.15:
        form = "emerging"
    elif (m["persistence"] >= 0.55 and m["biggest_day"] < 0.4 and m["gap_share"] < 0.4
          and m["inner_dip"] > -0.10 and m["volume"] >= 1.0):
        form = "persistent"
    else:
        form = "burst"
    m["form"] = form
    m["healthy_pullback"] = healthy_pullback(m)
    m["candidate"] = form == "persistent" and m["healthy_pullback"]
    m["score"] = size * m["persistence"]
    return m


def healthy_pullback(m: dict) -> bool:
    return (-0.07 <= m["pullback"] <= -0.04 and 1 <= m["pullback_days"] <= 5
            and m["pullback_worst_day"] > -0.05 and m["pullback_volume"] < 1.0 and not m["below_halfway"])


def in_universe(c: np.ndarray, v: np.ndarray) -> bool:
    return c[-1] >= MIN_PRICE and np.mean(c[-20:] * v[-20:]) / 1e7 >= MIN_TRADED_CR


def scan(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every stock's form as of its latest bar."""
    latest = max(df.index[-1] for df in bars.values())
    rows = {}
    for sym, df in bars.items():
        if len(df) < WINDOW or df.index[-1] < latest - pd.Timedelta(days=4):
            continue
        w = df.iloc[-WINDOW:]
        o, h, l, c, v = (w[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
        if not in_universe(c, v):
            continue
        rows[sym] = {"date": w.index[-1].date(), "price": c[-1], **analyse(o, h, l, c, v)}
    return pd.DataFrame(rows).T


def shortlist(t: pd.DataFrame, n: int = LIST_SIZE) -> pd.DataFrame:
    """READY names first, then WATCH: persistent movers still within 4% of their peak,
    waiting for a 4-7% dip. Best leg (size x persistence) first within each."""
    ready = t[t.candidate.eq(True)].sort_values("score", ascending=False).assign(status="READY")
    watch = t[t.form.eq("persistent") & ~t.candidate.eq(True) & (t.pullback.astype(float) > -0.04)]
    watch = watch.sort_values("score", ascending=False).assign(status="WATCH")
    return pd.concat([ready, watch]).head(n)


def digest(s: pd.DataFrame, prices_to: dt.date) -> str:
    """Plain text: Telegram's Markdown parser chokes on stray underscores in data."""
    lines = [f"form.py -- {dt.datetime.now(IST):%d %b %Y} (prices to {prices_to:%d %b})", ""]
    for status, title in (("READY", "READY -- in a healthy 4-7% pullback"),
                          ("WATCH", "WATCH -- in form, wait for a 4-7% dip")):
        rows = s[s.status == status]
        if not len(rows):
            continue
        lines.append(f"{title}:")
        for sym, r in rows.iterrows():
            head = f"{sym} {r.price:,.1f} | leg +{r['size'] * 100:.0f}% in {int(r.length)}d"
            if status == "READY":
                lines.append(f"{head} | {r.pullback * 100:+.1f}% from peak | target {r.peak:,.1f} | "
                             f"exit below {r.halfway:,.1f}")
            else:
                lines.append(f"{head} | buy zone {r.peak * 0.93:,.1f}-{r.peak * 0.96:,.1f}")
        lines.append("")
    if not len(s):
        lines.append("No persistent movers today.")
    lines.append("Check live prices in Kite before acting.")
    return "\n".join(lines)


def universe() -> list[str]:
    """NSE's equity list; NSE often refuses cloud servers, so fall back to the
    copy the momentum scanner keeps in BigQuery."""
    try:
        return data.load_universe()
    except Exception:
        from google.cloud import bigquery
        rows = bigquery.Client(project=PROJECT).query(
            f"SELECT Ticker FROM `{PROJECT}.data_options.nse_ticker_cache`").result()
        return sorted({r.Ticker.removesuffix(".NS") for r in rows})


def complete_sessions_only(bars: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    now = dt.datetime.now(IST)
    if now.time() >= dt.time(15, 45):
        return bars
    return {s: df[df.index.date < now.date()] for s, df in bars.items()}


def send_telegram(text: str) -> None:
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("Telegram skipped: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set.")
        return
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage",
                                 data=json.dumps({"chat_id": chat, "text": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=15).read()


def names_only(s: pd.DataFrame) -> str:
    """The Telegram message: just the names. The details stay in digest(), printed to the logs."""
    lines = [f"form {dt.datetime.now(IST):%d %b}"]
    for status in ("READY", "WATCH"):
        names = list(s.index[s.status == status])
        if names:
            lines.append(f"{status}: {', '.join(names)}")
    if len(lines) == 1:
        lines.append("No persistent movers today.")
    return "\n".join(lines)


def run() -> tuple[str, str]:
    """(details for the log, names-only message for Telegram)"""
    bars = complete_sessions_only(data.download(universe(), period="3mo", min_rows=WINDOW))
    bars = {s: df for s, df in bars.items() if len(df) >= WINDOW}
    t = scan(bars)
    prices_to = max(df.index[-1] for df in bars.values()).date()
    print(f"{len(t)} liquid stocks priced Rs {MIN_PRICE:.0f}+; form: {t.form.value_counts().to_dict()}")
    s = shortlist(t)
    return digest(s, prices_to), names_only(s)


def main(request: Any = None):
    """Cloud Run entry point (functions-framework) and command line."""
    details, names = run()
    print(details)
    if request is not None:
        send_telegram(names)
        return names, 200
    return None


if __name__ == "__main__":
    main()
