"""
OSCILLATING RANGE SCANNER
=========================
Finds stocks trading in a definable band and reports where in that band they
currently sit -- the "buy near 1200, sell near 1300" method, applied across
the universe instead of by eye on one stock.

Separate from swings.py on purpose. swings.py hunts trend CONTINUATION
(pullback_bounce into an uptrend); this hunts MEAN REVERSION inside a range.
They are opposite premises and want opposite things from the same price
action, so keeping them apart avoids one rule quietly contaminating the
other. Universe gates, indicators and the regime classifier are imported
rather than duplicated.

WHAT THE BACKTEST ACTUALLY SHOWED -- read before trusting any output here.
Tested over 500 tickers, entries at band bottom (<=25%), exits at band top
(>=75%), regime flip to BREAKDOWN as invalidation, no overlapping trades:

    n=316 trades   win(>=3%) = 33.5%   mean P&L = -0.06%   median = -3.32%

    reached sell zone : n=104 (33%)  win 97.1%  mean +11.17%
    range broke down  : n=187 (59%)  win  0.5%  mean  -5.91%
    timed out         : n=25  ( 8%)  win 16.0%  mean  -3.03%

So: when a range holds this works extremely well, and 59% of the time it
doesn't. The two roughly cancel -- BEFORE transaction costs, which are real
at this frequency. A follow-up test compared the 104 that held against the
187 that broke on seven features knowable at entry (band width, ATR relative
to band width, range age, position vs the 200-day, volume drying up, drift
inside the band, share of days inside the band). Breakdown rate sat at
55-63% in every bucket of every feature -- nothing separated them. The
strongest mechanical candidate, ATR-vs-band-width, was flat (57.1 / 60.6 /
58.6 / 56.0) and its mean difference pointed the wrong way.

Treat this as a WATCHLIST GENERATOR, not a signal: it narrows ~2,570 tickers
to ~25 sitting at the bottom of a real range, which is a tractable number to
follow. It cannot tell you which of those ranges will hold.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
from typing import Any, Optional

import pandas as pd
from google.cloud import bigquery

import position_check
import swings

logging.basicConfig(level=logging.WARNING)
LOGGER = logging.getLogger(__name__)

DEFAULT_BQ_TABLE_ID = "oscillating"

# Zone thresholds, as a percentage of band width. These match what the
# backtest above was run with -- changing them invalidates those numbers.
BUY_ZONE_MAX_POS = 25.0
SELL_ZONE_MIN_POS = 75.0


def find_oscillating(
    tickers: list[str],
    config: swings.ScannerConfig,
    run_timestamp: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Applies swings.py's universe gates (price band, liquidity, volatility
    identity) then classifies regime, keeping only stocks that are actually
    OSCILLATING. Returns (dataframe, failed_tickers).

    Deliberately does NOT apply the persistence backtest or min_score gate:
    those score a stock on the floor+extension MOMENTUM rule, which is the
    wrong question for a range trade and would reject range-bound names for
    failing at a strategy we aren't running on them.
    """
    yf = swings.load_yfinance()
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    batches = swings.chunks(tickers, config.chunk_size)

    for bn, batch in enumerate(batches, start=1):
        try:
            batch_data = swings.download_batch(yf, batch, config)
        except Exception as exc:
            failures.extend(batch)
            if config.verbose:
                print(f"\nBatch {bn} failed: {exc}")
            continue

        for ticker in batch:
            try:
                data = swings.get_ticker_frame(batch_data, ticker)
                if not swings.has_enough_data(data, config):
                    continue
                ind = swings.add_indicators(data, config)
                latest = ind.iloc[-1]

                price = float(latest["Close"])
                traded_value = float(latest["avg_traded_value20_cr"])
                if pd.isna(price) or pd.isna(traded_value):
                    continue
                if not (config.min_price < price < config.max_price):
                    continue
                if traded_value < config.min_avg_traded_value_cr:
                    continue

                avg_vol, med_vol, vol_days, vol_ratio = swings.verify_volatility_identity(ind, config)
                if avg_vol < config.min_avg_volatility or med_vol < config.min_median_volatility:
                    continue
                if vol_days < config.min_volatile_days or vol_ratio < config.min_volatility_ratio:
                    continue

                regime = position_check.classify_regime(ind["Close"].to_numpy())
                if regime is None or regime["Regime"] != "OSCILLATING":
                    continue

                pos = regime["Pos_In_Band_Pct"]
                zone = ("BUY_ZONE" if pos <= BUY_ZONE_MAX_POS
                        else "SELL_ZONE" if pos >= SELL_ZONE_MIN_POS
                        else "MID_BAND")

                rows.append({
                    "Ticker": ticker,
                    "Bar_Date": pd.Timestamp(ind.index[-1]).date().isoformat(),
                    "Run_Timestamp": run_timestamp,
                    "Zone": zone,
                    "Price": round(price, 2),
                    "Band_Low": regime["Band_Low"],
                    "Band_High": regime["Band_High"],
                    "Pos_In_Band_Pct": pos,
                    "Band_Width_Pct": regime["Band_Width_Pct"],
                    # what reaching the top of the band would be worth from here
                    "Upside_To_Top_Pct": round((regime["Band_High"] - price) / price * 100, 2),
                    "Downside_To_Bottom_Pct": round((price - regime["Band_Low"]) / price * 100, 2),
                    "Drift_Pct": regime["Drift_Pct"],
                    "Inside_Band_Pct": regime["Inside_Band_Pct"],
                    "Avg_Volatility": round(avg_vol, 2),
                    "Traded_Value_Cr": round(traded_value, 2),
                    "Note": regime["Note"],
                })
            except Exception as exc:
                failures.append(ticker)
                if config.verbose:
                    print(f"\nSkipped {ticker}: {exc}")

        print(f"\rProgress: {int(bn / len(batches) * 100)}% ({bn}/{len(batches)}) | "
              f"{len(rows)} oscillating", end="", flush=True)
    print()

    if not rows:
        return pd.DataFrame(), failures

    df = pd.DataFrame(rows)
    zone_rank = {"BUY_ZONE": 0, "SELL_ZONE": 1, "MID_BAND": 2}
    df["_zone_rank"] = df["Zone"].map(zone_rank).fillna(3)
    # Within a zone, sort by how much room is left to the other side of the
    # band. That is a MAGNITUDE ordering (what the trade is worth if it
    # works), explicitly NOT a probability ordering -- nothing tested
    # predicts which ranges hold, so this must not be read as a ranking of
    # likelihood.
    df = df.sort_values(["_zone_rank", "Upside_To_Top_Pct"], ascending=[True, False])
    return df.drop(columns=["_zone_rank"]).reset_index(drop=True), failures


def write_to_bigquery(df: pd.DataFrame, project_id: str, dataset_id: str, table_id: str) -> int:
    """Idempotent on (Ticker, Bar_Date), same pattern as swings.write_to_bigquery:
    re-running on the same trading day replaces rather than duplicates."""
    if df.empty:
        return 0

    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    output = df.copy()
    output["Bar_Date"] = pd.to_datetime(output["Bar_Date"]).dt.date
    output["Run_Timestamp"] = pd.to_datetime(output["Run_Timestamp"], utc=True)

    try:
        client.get_table(table_ref)
        pairs = output[["Ticker", "Bar_Date"]].drop_duplicates()
        conditions = " OR ".join(
            f"(Ticker = '{row.Ticker.replace(chr(39), chr(39) * 2)}' "
            f"AND Bar_Date = DATE('{row.Bar_Date.isoformat()}'))"
            for row in pairs.itertuples(index=False)
        )
        client.query(f"DELETE FROM `{table_ref}` WHERE {conditions}").result()
    except Exception:
        pass   # table doesn't exist yet -- the load below creates it

    client.load_table_from_dataframe(
        output, table_ref,
        job_config=bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
        ),
    ).result()
    return len(output)


def format_telegram_digest(df: pd.DataFrame) -> Optional[str]:
    """
    Buy-zone names only -- a mid-band stock has no edge either way and a
    sell-zone one is only actionable if you already hold it (the holdings
    regime check covers that case). Returns None if there's nothing to send.

    Plain text, no parse_mode -- see swings.send_telegram_notification for
    why Markdown breaks on ticker/setup content.
    """
    if df.empty:
        return None
    buys = df[df["Zone"] == "BUY_ZONE"]
    if buys.empty:
        return None

    run_time_ist = (dt.datetime.now(dt.timezone.utc) + swings.IST_OFFSET).strftime("%Y-%m-%d %H:%M IST")
    lines = [f"Oscillating scan -- {run_time_ist}",
             f"{len(buys)} stock(s) at the BOTTOM of their range:", ""]
    for _, r in buys.head(15).iterrows():
        lines.append(f"{r['Ticker']}  {r['Price']:.0f}  band {r['Band_Low']:.0f}-{r['Band_High']:.0f} "
                     f"({r['Pos_In_Band_Pct']:.0f}%)  +{r['Upside_To_Top_Pct']:.0f}% to top")
    lines.append("")
    # This warning is not boilerplate -- it is the measured base rate. See the
    # module docstring: 59% of these ranges broke before reaching the top.
    lines.append("Watchlist, not signals: ~59% of ranges break before reaching the top. "
                 "Band position says nothing about which ones hold.")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Finds stocks oscillating in a band and reports where in the band they sit."
    )
    p.add_argument("--tickers", nargs="+", default=None)
    p.add_argument("--limit", default=0, type=int)
    p.add_argument("--min-price", default=swings.ScannerConfig.min_price, type=float)
    p.add_argument("--max-price", default=swings.ScannerConfig.max_price, type=float)
    p.add_argument("--min-traded-value-cr", default=swings.ScannerConfig.min_avg_traded_value_cr, type=float)
    p.add_argument("--min-avg-volatility", default=swings.ScannerConfig.min_avg_volatility, type=float)
    p.add_argument("--zone", default="all", choices=["all", "buy", "sell", "mid"],
                    help="Filter the printed table (BigQuery always gets every oscillating row)")
    p.add_argument("--no-bq", action="store_true")
    p.add_argument("--no-telegram", action="store_true")
    p.add_argument("--project-id", default=None)
    p.add_argument("--dataset-id", default="data_options")
    p.add_argument("--table-id", default=DEFAULT_BQ_TABLE_ID)
    p.add_argument("--verbose", action="store_true")
    args, _ = p.parse_known_args()
    return args


def main(request: Any = None) -> Optional[tuple[str, int]]:
    """
    CLI and HTTP entry point. Branches on request is not None BEFORE touching
    argparse -- on an HTTP invocation sys.argv holds the container's own
    launch flags (functions-framework's --target/--source/--port), and letting
    argparse see those is what crashed swings.py's Cloud Run deploy.
    """
    if request is not None:
        try:
            body = request.get_json(silent=True) or {}
        except Exception:
            body = {}
        config = swings.ScannerConfig(
            min_price=float(body.get("min_price", swings.ScannerConfig.min_price)),
            max_price=float(body.get("max_price", swings.ScannerConfig.max_price)),
            min_avg_traded_value_cr=float(body.get("min_traded_value_cr",
                                                    swings.ScannerConfig.min_avg_traded_value_cr)),
            min_avg_volatility=float(body.get("min_avg_volatility", swings.ScannerConfig.min_avg_volatility)),
            verbose=bool(body.get("verbose", False)),
        )
        raw_tickers = body.get("tickers")
        tickers = (swings.parse_tickers(raw_tickers) if raw_tickers
                   else swings.load_nse_tickers(project_id=body.get("project_id")))
        limit = int(body.get("limit", 0))
        zone_filter = "all"
        no_bq = bool(body.get("no_bq", False))
        no_telegram = bool(body.get("no_telegram", False))
        project_id = body.get("project_id")
        dataset_id = body.get("dataset_id", "data_options")
        table_id = body.get("table_id", DEFAULT_BQ_TABLE_ID)
    else:
        args = parse_args()
        config = swings.ScannerConfig(
            min_price=args.min_price,
            max_price=args.max_price,
            min_avg_traded_value_cr=args.min_traded_value_cr,
            min_avg_volatility=args.min_avg_volatility,
            verbose=args.verbose,
        )
        tickers = (swings.parse_tickers(args.tickers) if args.tickers
                   else swings.load_nse_tickers(project_id=args.project_id))
        limit = args.limit
        zone_filter = args.zone
        no_bq = args.no_bq
        no_telegram = args.no_telegram
        project_id = args.project_id
        dataset_id = args.dataset_id
        table_id = args.table_id

    if limit > 0:
        tickers = tickers[:limit]

    print(f"Universe: Rs.{config.min_price}-{config.max_price} | "
          f"Liquidity >= {config.min_avg_traded_value_cr} Cr | "
          f"6-month avg volatility >= {config.min_avg_volatility}%")
    print(f"Band: {position_check.REGIME_BAND_LOW_PCTILE}/{position_check.REGIME_BAND_HIGH_PCTILE} "
          f"percentile over {position_check.REGIME_LOOKBACK_DAYS} sessions | "
          f"buy zone <= {BUY_ZONE_MAX_POS:.0f}%, sell zone >= {SELL_ZONE_MIN_POS:.0f}%")
    print(f"Scanning {len(tickers)} tickers for range-bound stocks...")
    print("-" * 80)

    run_timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    df, failures = find_oscillating(tickers, config, run_timestamp)

    if failures:
        print(f"\n{len(failures)} tickers failed or had no usable data.")

    if df.empty:
        message = "No oscillating stocks found."
        print(f"\n{message}")
        return (message, 200) if request is not None else None

    display = df if zone_filter == "all" else df[df["Zone"] == f"{zone_filter.upper()}_ZONE"]
    if zone_filter == "mid":
        display = df[df["Zone"] == "MID_BAND"]

    counts = df["Zone"].value_counts().to_dict()
    print(f"\nFound {len(df)} oscillating: {counts}")
    print("-" * 100)
    print(f"{'Ticker':<15}{'Zone':<11}{'Price':>9}{'Band':>18}{'Pos':>6}"
          f"{'ToTop%':>8}{'Width%':>8}{'Vol%':>7}")
    print("-" * 100)
    for _, r in display.head(40).iterrows():
        band = f"{r['Band_Low']:.0f}-{r['Band_High']:.0f}"
        print(f"{r['Ticker']:<15}{r['Zone']:<11}{r['Price']:>9.1f}{band:>18}"
              f"{r['Pos_In_Band_Pct']:>5.0f}%{r['Upside_To_Top_Pct']:>8.1f}"
              f"{r['Band_Width_Pct']:>8.1f}{r['Avg_Volatility']:>7.1f}")

    print("\nWATCHLIST, NOT SIGNALS. Backtested: 33.5% of these reach the band top "
          "(+11.2% avg), 59% break down first (-5.9% avg), net ~breakeven before costs. "
          "Nothing measurable predicts which -- see the module docstring.")

    if not no_telegram:
        try:
            bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
            chat_id_raw = os.getenv("TELEGRAM_CHAT_ID")
            if not bot_token or not chat_id_raw:
                print("\nTelegram skipped: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set.")
            else:
                text = format_telegram_digest(df)
                if text is None:
                    print("\nNo buy-zone stocks -- Telegram skipped.")
                else:
                    chat_ids = [c.strip() for c in chat_id_raw.split(",") if c.strip()]
                    telegram_failures = swings.send_telegram_to_all(text, bot_token, chat_ids)
                    sent = len(chat_ids) - len(telegram_failures)
                    print(f"\nSent Telegram to {sent}/{len(chat_ids)} recipient(s)"
                          + (f"; failed: {telegram_failures}" if telegram_failures else "."))
        except Exception as exc:
            print(f"\nTelegram skipped/failed: {exc}")

    if not no_bq:
        try:
            resolved = swings._resolve_project_id(project_id)
            n = write_to_bigquery(df, resolved, dataset_id, table_id)
            print(f"Wrote {n} rows to {resolved}.{dataset_id}.{table_id}")
        except Exception as exc:
            print(f"\nBigQuery write skipped/failed: {exc}")

    if request is not None:
        return (f"Oscillating scan completed: {len(df)} range-bound stocks "
                f"({counts.get('BUY_ZONE', 0)} in buy zone).", 200)
    return None


if __name__ == "__main__":
    main()
