"""
POSITION MOMENTUM CHECK
=======================
Separate from swings.py on purpose -- this checks REAL holdings (your actual
entry price/date from data_options.holdings) against the same floor+extension
exit rule the scanner is built on, not scanner candidates. Different concern,
different lifecycle: swings.py finds new trades; this tracks whether trades
you're already in are still worth holding.

Answers the question asked by hand, over and over, for BALUFORGE/HEG/DYCL
this session: is this position still in its momentum window (worth keeping
capital in), or did it already hit the rule's real exit trigger?

Usage:
    python position_check.py                    # check all active holdings
    python position_check.py --project-id XYZ    # override default project
"""

from __future__ import annotations

import argparse
import datetime as dt
from typing import Optional

import pandas as pd
from google.cloud import bigquery

import swings


def check_position_momentum(
    ticker: str, entry_price: float, entry_date: str, config: swings.ScannerConfig
) -> dict:
    """
    Traces the actual day-by-day price action from a REAL entry point (not
    a scanner-signal day) using the exact same floor+extension rule the
    exit logic is built on.

    Mirrors the scanner's own extension bounds exactly -- the window is
    capped at config.max_extension_days past the day the floor was first
    crossed. If a non-higher close already happened inside that window,
    the verdict reflects that the rule's exit trigger already fired (even
    if price recovered later -- the rule exits on the FIRST break, it
    doesn't wait to see what happens next), not just "did it close lower
    today."

    Returns a dict; does not raise on missing data (returns a NO DATA
    verdict instead), since this is meant to run across many holdings
    unattended.
    """
    yf = swings.load_yfinance()
    try:
        raw = yf.download(ticker, period="3mo", interval="1d", progress=False, auto_adjust=True, threads=False)
        data = swings.normalize_single_ticker_columns(raw)
    except Exception as exc:
        return {"Ticker": ticker, "Verdict": f"DATA ERROR: {exc}"}

    entry_ts = pd.Timestamp(entry_date)
    post_entry = data.loc[data.index >= entry_ts]
    if post_entry.empty:
        return {"Ticker": ticker, "Verdict": "NO DATA since entry date"}

    closes = post_entry["Close"].to_numpy()
    dates = post_entry.index
    current_price = float(closes[-1])
    pnl_pct = (current_price - entry_price) / entry_price * 100
    floor = entry_price * (1 + config.target_pct / 100)

    floor_day_idx = next((i for i, c in enumerate(closes) if c >= floor), None)

    if floor_day_idx is None:
        return {
            "Ticker": ticker, "Entry_Price": entry_price, "Entry_Date": entry_date,
            "Current_Price": round(current_price, 2), "PnL_Pct": round(pnl_pct, 2),
            "Days_Since_Entry": len(closes), "Verdict": "NOT YET AT FLOOR",
        }

    ext_end = min(floor_day_idx + 1 + config.max_extension_days, len(closes))
    prev_close = closes[floor_day_idx]
    broke_at_idx = None
    for i in range(floor_day_idx + 1, ext_end):
        if closes[i] <= prev_close:
            broke_at_idx = i
            break
        prev_close = closes[i]

    if broke_at_idx is not None:
        exit_price = float(closes[broke_at_idx])
        exit_pnl = (exit_price - entry_price) / entry_price * 100
        verdict = (
            f"MOMENTUM BROKEN on {dates[broke_at_idx].date()} at {exit_price:.2f} "
            f"({exit_pnl:+.2f}%) -- rule's exit already triggered, even though "
            f"current price may differ"
        )
        still_in_momentum = False
    elif ext_end - floor_day_idx - 1 >= config.max_extension_days:
        verdict = f"EXTENSION WINDOW EXPIRED ({config.max_extension_days} days past floor) -- rule says exit now"
        still_in_momentum = False
    else:
        verdict = "STILL IN MOMENTUM -- hold"
        still_in_momentum = True

    return {
        "Ticker": ticker,
        "Entry_Price": entry_price,
        "Entry_Date": entry_date,
        "Current_Price": round(current_price, 2),
        "PnL_Pct": round(pnl_pct, 2),
        "Floor_Crossed_On": str(dates[floor_day_idx].date()),
        "Still_In_Momentum": still_in_momentum,
        "Days_Since_Entry": len(closes),
        "Verdict": verdict,
    }


def run_position_check(project_id: str, config: Optional[swings.ScannerConfig] = None) -> pd.DataFrame:
    """
    Reads active rows from data_options.holdings, runs check_position_momentum
    on each, prints a summary, and writes the results to
    data_options.position_momentum_check (timestamped, append-only history --
    so "was I told to exit this three days ago" is answerable later, not
    just "what does it say right now").
    """
    config = config or swings.ScannerConfig()
    client = bigquery.Client(project=project_id)

    holdings = list(client.query(
        f"SELECT Ticker, Entry_Price, Entry_Date, Quantity, Notes "
        f"FROM `{project_id}.data_options.holdings` WHERE Active = TRUE"
    ).result())

    if not holdings:
        print("No active holdings in data_options.holdings.")
        return pd.DataFrame()

    checked_at = dt.datetime.now(dt.timezone.utc).isoformat()
    rows = []
    print(f"\n{'Ticker':<14} {'Entry':>9} {'Current':>9} {'PnL%':>7} {'Verdict'}")
    print("-" * 100)
    for h in holdings:
        result = check_position_momentum(h["Ticker"], float(h["Entry_Price"]), str(h["Entry_Date"]), config)
        result["Checked_At"] = checked_at
        result["Quantity"] = h["Quantity"]
        rows.append(result)
        pnl = result.get("PnL_Pct")
        current = result.get("Current_Price")
        print(f"{result['Ticker']:<14} {h['Entry_Price']:>9.2f} "
              f"{current if current is not None else '-':>9} "
              f"{(f'{pnl:+.2f}' if pnl is not None else '-'):>7} {result['Verdict']}")

    results_df = pd.DataFrame(rows)
    try:
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
        client.load_table_from_dataframe(
            results_df, f"{project_id}.data_options.position_momentum_check", job_config=job_config
        ).result()
        print(f"\nWrote {len(results_df)} rows to {project_id}.data_options.position_momentum_check")
    except Exception as exc:
        print(f"\nposition_momentum_check write skipped/failed: {exc}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Check real holdings against the floor+extension exit rule.")
    parser.add_argument("--project-id", default="sudarshan-442212")
    args = parser.parse_args()
    run_position_check(args.project_id)


if __name__ == "__main__":
    main()
