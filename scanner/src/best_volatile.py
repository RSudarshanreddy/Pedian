"""
Predictable Swing Trading Engine
================================
Goal:
Core Philosophy:
We are NOT predicting exact prices.
We are ranking high probability swing setups.
"""


import argparse
import datetime as dt
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any
import pandas as pd
import numpy as np
import yfinance as yf

# =========================================================
# CONFIG
# =========================================================

@dataclass
class ScannerConfig:
    # Data
    period: str = "1y"
    interval: str = "1d"
    # Universe
    min_price: float = 400
    max_price: float = 1800
    min_avg_traded_value_cr: float = 25
    # Volatility
    min_atr_pct: float = 1.5
    min_stddev: float = 0.8
    max_stddev: float = 6
    # Trend
    ema_fast: int = 20
    ema_slow: int = 50

    # Expansion Cycle
    expansion_move_pct: float = 3 # keep checking this values, learning 
    pullback_min_pct: float = 0.5 # keep checking this values, learning 
    pullback_max_pct: float = 10  # keep checking this values, learning 

    # Entry
    min_volume_spike: float = 0.8 # keep checking this values, learning / importnant 
    # Trade
    target_pct: float = 4 # minimum 3% is good volatile
    stop_buffer_pct: float = 2
    # Batch
    chunk_size: int = 75

    #input
    limit : int = 10
    # Output
    top_n: int = 27    
    # Event Spike Protection
    max_5d_move_pct: float = 22  #momentun can sustain, lets check
    max_gap_pct: float = 5
    max_range_stddev: float = 4
    max_pullback_volume_ratio: float = 1.2


CONFIG = ScannerConfig()
# =========================================================
# NSE UNIVERSE
# =========================================================

NSE_EQUITY_LIST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
DEFAULT_BQ_PROJECT = "sudarshan-442212"
DEFAULT_BQ_CANDIDATES_TABLE = "sudarshan-442212.data_options.high_volatile_results"
HIGH_VOLATILE_BQ_COLUMNS = [
    "run_date",
    "ticker",
    "score",
    "stddev_20",
    "price",
    "entry",
    "sl",
    "target",
    "atr_pct",
    "move10d_pct",
    "move20d_pct",
    "move30d_pct",
    "move40d_pct",
    "pullback_pct",
    "volume_spike",
    "stretch_penalty",
    "medium_term_stretch_penalty",
    "volatility_stability_score",
    "trend_score",
    "pullback_score",
    "volume_score",
    "expansion_score",
    "volatility_persistence_score",
    "swing_count_score",
    "trend_age_score",
    "volatility_persistence",
    "swing_count",
    "trend_age",
    "regime_ratio",
    "acceleration",
]
HIGH_VOLATILE_BQ_SCHEMA = [
    ("run_date", "DATE"),
    ("ticker", "STRING"),
    ("score", "NUMERIC"),
    ("stddev_20", "NUMERIC"),
    ("price", "NUMERIC"),
    ("entry", "NUMERIC"),
    ("sl", "NUMERIC"),
    ("target", "NUMERIC"),
    ("atr_pct", "NUMERIC"),
    ("move10d_pct", "NUMERIC"),
    ("move20d_pct", "NUMERIC"),
    ("move30d_pct", "NUMERIC"),
    ("move40d_pct", "NUMERIC"),
    ("pullback_pct", "NUMERIC"),
    ("volume_spike", "NUMERIC"),
    ("stretch_penalty", "NUMERIC"),
    ("medium_term_stretch_penalty", "NUMERIC"),
    ("volatility_stability_score", "NUMERIC"),
    ("trend_score", "NUMERIC"),
    ("pullback_score", "NUMERIC"),
    ("volume_score", "NUMERIC"),
    ("expansion_score", "NUMERIC"),
    ("volatility_persistence_score", "NUMERIC"),
    ("swing_count_score", "NUMERIC"),
    ("trend_age_score", "NUMERIC"),
    ("volatility_persistence", "NUMERIC"),
    ("swing_count", "INTEGER"),
    ("trend_age", "INTEGER"),
    ("regime_ratio", "NUMERIC"),
    ("acceleration", "NUMERIC"),
]


def to_yahoo_symbol(symbol):
    symbol = str(symbol).strip().upper()
    if symbol.endswith(".NS"):
        return symbol
    return f"{symbol}.NS"

def chunks(values, size):
    return [values[i : i + size] for i in range(0, len(values), size)]

def normalize_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def load_nse_tickers(source=NSE_EQUITY_LIST_URL):
    df = pd.read_csv(source)
    symbols = df["SYMBOL"].dropna().astype(str).tolist()
    return [to_yahoo_symbol(x) for x in symbols]

# =========================================================
# BIGQUERY
# =========================================================

def load_bigquery_client(project_id: str | None = None) -> Any:
    try:
        from google.cloud import bigquery
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: google-cloud-bigquery. "
            "Install it with: pip install google-cloud-bigquery"
        ) from exc

    return bigquery.Client(project=project_id)


def to_bigquery_column_name(column: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", str(column))
    cleaned = re.sub(r"_+", "_", cleaned).strip("_").lower()
    if not cleaned:
        return "column"
    if cleaned[0].isdigit():
        return f"col_{cleaned}"
    return cleaned


def prepare_high_volatile_for_bigquery(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    output = output.rename(
        columns={
            "ATR%": "ATR_Pct",
            "Move10D%": "Move10D_Pct",
            "Move20D%": "Move20D_Pct",
            "Move30D%": "Move30D_Pct",
            "Move40D%": "Move40D_Pct",
            "Pullback%": "Pullback_Pct",
            "VolumeSpike": "Volume_Spike",
            "TrendScore": "Trend_Score",
            "PullbackScore": "Pullback_Score",
            "VolumeScore": "Volume_Score",
            "ExpansionScore": "Expansion_Score",
            "VolatilityPersistenceScore": "Volatility_Persistence_Score",
            "SwingCountScore": "Swing_Count_Score",
            "TrendAgeScore": "Trend_Age_Score",
        }
    )
    output.columns = [to_bigquery_column_name(column) for column in output.columns]
    output.insert(0, "run_date", pd.Timestamp.today().date())

    for column in HIGH_VOLATILE_BQ_COLUMNS:
        if column not in output.columns:
            output[column] = None

    integer_columns = ["swing_count", "trend_age"]
    for column in integer_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce").astype("Int64")

    return output[HIGH_VOLATILE_BQ_COLUMNS]


def write_to_bigquery(
    data: pd.DataFrame,
    table_id: str,
    project_id: str | None,
) -> None:
    if data.empty:
        print(f"No rows to write to BigQuery table {table_id}.")
        return

    print(f"Writing {len(data)} rows to BigQuery table {table_id}...")
    client = load_bigquery_client(project_id)

    from google.cloud import bigquery

    output = prepare_high_volatile_for_bigquery(data)
    rows = [
        {column: to_json_safe_value(value) for column, value in row.items()}
        for row in output.to_dict(orient="records")
    ]
    schema = [
        bigquery.SchemaField(column_name, column_type)
        for column_name, column_type in HIGH_VOLATILE_BQ_SCHEMA
    ]
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
    )
    job = client.load_table_from_json(rows, table_id, job_config=job_config)
    job.result()
    print(f"Loaded {len(rows)} rows to BigQuery table {table_id}")


def to_json_safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, float) and item.is_integer():
            return int(item)
        return item
    return value

# =========================================================
# HELPERS
# =========================================================

def download_batch(tickers, config):
    data = yf.download(
        tickers=tickers,
        period=config.period,
        interval=config.interval,
        progress=False,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
    )
    return data

def get_ticker_frame(batch_data, ticker):
    if not isinstance(batch_data.columns, pd.MultiIndex):
        df = normalize_columns(batch_data)
        # print(df.head(1))
        return df.dropna(how="all")
    if ticker not in batch_data.columns.get_level_values(0):
        return pd.DataFrame()

    df = batch_data[ticker].copy()
    df = normalize_columns(df)
    df = df.dropna(how="all")
    return df


# =========================================================
# INDICATORS
# =========================================================

def add_indicators(df):
    df = df.copy()    
    # RETURNS    
    df["RETURNS"] = (df["Close"].pct_change(fill_method=None)) * 100
    
    # TRUE RANGE    
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift(1))
    low_close = np.abs(df["Low"] - df["Close"].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)    

    # ATR    
    df["ATR_5"] = tr.rolling(5).mean()
    df["ATR_PCT"] = (df["ATR_5"] / df["Close"]) * 100
    
    # STANDARD DEVIATION    
    df["STDDEV_20"] = df["RETURNS"].rolling(20).std()
    df["VOLATILITY_REGIME"] = ((df["STDDEV_20"] >= 2) & (df["STDDEV_20"] <= 3.5)).astype(int)

    df["VOLATILITY_PERSISTENCE"] = df["VOLATILITY_REGIME"].rolling(90).sum()
    df["VOLATILITY_PERSISTENCE_90"] = df["VOLATILITY_REGIME"].rolling(90).sum()
    df["VOLATILITY_PERSISTENCE_180"] = df["VOLATILITY_REGIME"].rolling(180).sum()  
    # EMAs  
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean() 

    # Trend Age:
    df["ABOVE_EMA50"] = ( df["Close"] > df["EMA50"] ).astype(int)

    trend_age = []
    counter = 0

    for val in df["ABOVE_EMA50"]:
        if val:
            counter += 1
        else:
            counter = 0
        trend_age.append(counter)

    df["TREND_AGE"] = np.array(trend_age)   

    # VOLUME
    df["AVG_VOL_20"] = df["Volume"].rolling(20).mean()
    df["VOL_SPIKE"] = df["Volume"] / df["AVG_VOL_20"] 

    # TRADED VALUE
    df["TRADED_VALUE_CR"] = (df["Close"] * df["AVG_VOL_20"]) / 10_000_000
    
    # RECENT HIGH
    df["HIGH_20"] = df["High"].rolling(20).max() 

    # EXPANSION MOVE
    df["MOVE_10D"] = ((df["Close"] / df["Close"].shift(10)) - 1) * 100
    df["MOVE_20D"] = ((df["Close"] / df["Close"].shift(20)) - 1) * 100
    df["MOVE_30D"] = ((df["Close"] / df["Close"].shift(30)) - 1) * 100   
    df["MOVE_40D"] = ((df["Close"] / df["Close"].shift(40)) - 1) * 100

    # PULLBACK %   
    df["PULLBACK_PCT"] = ((df["HIGH_20"] - df["Close"]) / df["HIGH_20"]) * 100

    # 5D MOVE
    df["MOVE_5D"] = ((df["Close"] / df["Close"].shift(5)) - 1) * 100
    
    # GAP %
    df["GAP_PCT"] = ((df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)) * 100

    # DAILY RANGE %
    df["RANGE_PCT"] = ((df["High"] - df["Low"]) / df["Close"]) * 100

    # RANGE STABILITY
    df["RANGE_STDDEV_20"] = df["RANGE_PCT"].rolling(20).std()

    # PULLBACK VOLUME QUALITY
    df["PULLBACK_VOL_RATIO"] = df["Volume"] / df["AVG_VOL_20"]
    df["VOLATILITY_CONSISTENCY"] = df["RANGE_PCT"].rolling(20).std()
    df["ATR_STDDEV_20"] = df["ATR_PCT"].rolling(20).std()

    # for how long this volatile..?
    df["VOL_REGIME"] = ((df["STDDEV_20"] > 1.5 ) &( df["STDDEV_20"] < 4 ) ).astype(int)
    df["VOL_REGIME_PERSISTENCE"] = ( df["VOL_REGIME"].rolling(60).sum())

    # is this goigng to be exhausted near by..? from past 60 days
    df["MOVE_60D"] = ((df["Close"]/df["Close"].shift(60)) - 1) * 100

    # Calculate swing count
    swings = 0
    for i in range(20, len(df)):
        move_up = ((df["Close"].iloc[i] / df["Close"].iloc[i-10]) - 1) * 100
        high_20 = df["High"].iloc[max(0, i-20):i+1].max()
        pullback = ((high_20 - df["Close"].iloc[i]) / high_20) * 100
        if move_up > 8 and pullback > 3:
            swings += 1

    df["SWING_COUNT"] = swings
    return df

# =========================================================
# FILTERS
# =========================================================

def passes_universe_filter(row, config):
    return (
        config.min_price <= row["Close"] <= config.max_price
        and row["TRADED_VALUE_CR"] >= config.min_avg_traded_value_cr
    )
def passes_trend_filter(row):
    return row["EMA20"] > row["EMA50"] and row["Close"] > row["EMA20"]
def passes_event_spike_filter(row, config):    

    # Reject huge recent spikes 
    if row["MOVE_5D"] > config.max_5d_move_pct:
        return False     
    # Reject abnormal gaps   
    if abs(row["GAP_PCT"]) > config.max_gap_pct:
        return False    
    return True

# =========================================================
# SCORING ENGINE
# =========================================================

def calculate_score(row):
    trend_score = min(((row["Close"] - row["EMA20"]) / row["EMA20"]) * 100 * 1.5, 10) # Average to 10
    atr_score = min(row["ATR_PCT"], 10)
    ideal_volatility = 3
    volatility_score = max(10 - (abs(row["STDDEV_20"] - ideal_volatility) * 2), 0)
    volume_score = min(row["VOL_SPIKE"] * 5, 10) # change here
    expansion_score = min(row["MOVE_10D"] / 1.5, 10) # Average to 10
    pull_back_ratio_volume = min(row["PULLBACK_VOL_RATIO"], 20)
    ideal_pullback = 3
    pullback_score = max( 10 - (abs( row["PULLBACK_PCT"] - ideal_pullback) * 3 ),  0)
    volatility_stability_score = max( 15 - ( row["VOLATILITY_CONSISTENCY"] * 2), 0)
    VOL_REGIME_PERSISTENCE_SCORE = min(row["VOL_REGIME_PERSISTENCE"] / 10, 10)
    medium_term_stretch_penalty = max(row["MOVE_60D"] - 35, 0)
    swing_count_score = min(row["SWING_COUNT"] * 3, 15)
    trend_age_score = min(row["TREND_AGE"] * 0.5, 10)
    volatility_persistence_score = min(row["VOLATILITY_PERSISTENCE"] / 15, 10)    
    regime_ratio = (row["VOLATILITY_PERSISTENCE_90"] / row["VOLATILITY_PERSISTENCE_180"])
    acceleration = (row["MOVE_10D"] - row["MOVE_30D"])

    acceleration_score = min(max(acceleration + 10, 0), 10)
    move_60d = row["MOVE_60D"]

    medium_term_penalty = max(move_60d - 30, 0)

    score = (
    trend_score * 0.20
    + pullback_score * 0.20
    + volatility_persistence_score * 0.30
    + volume_score * 0.10
    + volatility_score * 0.10
    + expansion_score * 0.10
    + acceleration_score * 0.10
)

    score -= medium_term_penalty * 0.05
    stretch_penalty = max( row["MOVE_10D"] - 15,  0)
    score = score - (stretch_penalty * 0.3)
    score = score - (medium_term_stretch_penalty * 0.05)
    return round(score, 2)


# =========================================================
# MAIN STOCK SCAN
# =========================================================

def stocks_from_df(ticker, df, config):
    if df is None or len(df) < 60:
        return None
    df = add_indicators(df)
    latest = df.iloc[-1]

    # FILTERS  
    if not passes_universe_filter(latest, config):
        return None   
    if not passes_event_spike_filter(latest, config):
        return None  
    if not passes_trend_filter(latest):
        return None        
  
    # SCORE 
    score = calculate_score(latest)    
    # TRADE LEVELS    
    entry, sl, target = generate_trade_levels(df, config)
    return {
        "Ticker": ticker,
        "Score": score,  
        "STDDEV_20": round(latest["STDDEV_20"], 2),         
        "Price": round(latest["Close"], 2),
        "Entry": entry,
        "SL": sl,
        "Target": target,
        "ATR%": round(latest["ATR_PCT"], 2), 
        "Move10D%": round(latest["MOVE_10D"], 2),
        "Move20D%": round(latest["MOVE_20D"], 2),
        "Move30D%": round(latest["MOVE_30D"], 2),
        "Move40D%": round(latest["MOVE_40D"], 2),
        "Pullback%": round(latest["PULLBACK_PCT"], 2),
        "VolumeSpike": round(latest["VOL_SPIKE"], 2),          
        "stretch_penalty":round(max(latest["MOVE_10D"] - 15, 0), 2),
        "medium_term_stretch_penalty": round(latest["MOVE_60D"], 2),        
        "volatility_stability_score":round(latest["VOLATILITY_CONSISTENCY"] , 2),
        "TrendScore": round(min(((latest["Close"] - latest["EMA20"]) / latest["EMA20"]) * 100, 15), 2),
        "PullbackScore": round(max( 10 - (abs( latest["PULLBACK_PCT"] - 3) * 3 ),  0) , 2),
        "VolumeScore": round(min(latest["VOL_SPIKE"] * 5, 10), 2),
        "ExpansionScore": round(min(latest["MOVE_10D"] , 15), 2),
        "VolatilityPersistenceScore": round(min(latest["VOLATILITY_PERSISTENCE"] / 15, 10), 2),
        "SwingCountScore": round(min(latest["SWING_COUNT"]* 0.8, 10), 2),
        "TrendAgeScore": round(min(latest["TREND_AGE"] * 0.5, 10), 2),
        "VOLATILITY_PERSISTENCE": latest["VOLATILITY_PERSISTENCE"],
        "SWING_COUNT": latest["SWING_COUNT"],
        "TREND_AGE": latest["TREND_AGE"],
        "regime_ratio": round(latest["VOLATILITY_PERSISTENCE_90"] / latest["VOLATILITY_PERSISTENCE_180"], 2) if latest["VOLATILITY_PERSISTENCE_180"] > 0 else 0,
        "acceleration": round(latest["MOVE_10D"] - latest["MOVE_30D"], 2)


    }


# =========================================================
# MAIN SCANNER
# =========================================================
def run_scanner(tickers, config):
    results = []
    ticker_batches = chunks(tickers, config.chunk_size)
    total_batches = len(ticker_batches)
    print("Scanning data .. ")

    for batch_num, batch in enumerate(ticker_batches, start=1):        
        try:
            batch_data = download_batch(batch, config)
        except Exception as e:
            print(f"BATCH FAILED: {e}")
            continue         

        for ticker in batch:
            try:
                df = get_ticker_frame(batch_data, ticker)
                if df.empty:
                    continue
                candidate = stocks_from_df(ticker, df, config)
                if candidate:
                    results.append(candidate)
            except Exception as e:
                import traceback
                print(f"{ticker} FAILED: {e}")
                traceback.print_exc()
    if not results:
        return pd.DataFrame()
    output = pd.DataFrame(results)
    output = output.sort_values("Score", ascending=False)
    # output = output.filter(output["ATR%"] )
    return output.head(config.top_n)

# =========================================================
# TRADE LEVELS
# =========================================================

def generate_trade_levels(df, config):
    latest = df.iloc[-1]
    entry = round(latest["Close"], 2)
    stop_loss = round(latest["EMA20"] * (1 - config.stop_buffer_pct / 100), 2)
    target = round(entry * (1 + config.target_pct / 100), 2)
    return entry, stop_loss, target

# =========================================================
# MAIN
# =========================================================

def parse_tickers(raw_tickers):
    tickers = []
    for value in raw_tickers:
        tickers.extend(to_yahoo_symbol(part) for part in value.split(",") if part.strip())
    return tickers


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find high-volatility NSE swing candidates and optionally write them to BigQuery."
    )
    
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Optional Yahoo/NSE tickers. If omitted, all NSE equity symbols are scanned.",
    )

    parser.add_argument(
        "--symbols-source",
        default=NSE_EQUITY_LIST_URL,
        help="CSV file or URL with an NSE SYMBOL column.",
    )

    # parser.add_argument(
    #     "--limit",
    #     default=10,
    #     type=int,
    #     help="Optional number of tickers to scan. Useful for testing the full list.",
    # )

    parser.add_argument("--top-n", default=CONFIG.top_n, type=int)
    parser.add_argument(
        "--bq-project",
        default=DEFAULT_BQ_PROJECT,
        help="Optional BigQuery project ID. Uses default credentials if omitted.",
    )
    parser.add_argument(
        "--bq-candidates-table",
        default=DEFAULT_BQ_CANDIDATES_TABLE,
        help="BigQuery table for scanner rows, for example project.dataset.high_volatile_results.",
    )
    parser.add_argument("--output", default="", help="Optional CSV output path.")
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    config = ScannerConfig(top_n=args.top_n)
    tickers = parse_tickers(args.tickers) if args.tickers else load_nse_tickers(args.symbols_source)
    # if args.limit > 0:
    tickers = tickers[: ]

    import traceback
    try:
        print(f"Scanning {len(tickers)} tickers...")   
        output = run_scanner(tickers, config)
        # OUTPUT
        print("\n")
        print("=" * 80)
        print("TOP SWING CANDIDATES")
        print("=" * 80)

        if output.empty:
            print("No candidates matched.")
        else:
            print(output.to_string(index=False))

    # this is for BQ written  -- start #      
        # if args.output:
        #     output.to_csv(args.output, index=False)
        #     print(f"\nSaved {len(output)} rows to {args.output}")
        # if args.bq_candidates_table:
        #     write_to_bigquery(
        #         output,
        #         args.bq_candidates_table,
        #         args.bq_project or None,
        #     )
    # this is for BQ written --- END ---#     
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"{tickers} FAILED")
        traceback.print_exc()
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
