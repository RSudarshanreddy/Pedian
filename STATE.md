# Pedian — where things stand

Frozen 2026-09-23. Read this before changing anything.

## What it does

Two scanners, opposite premises, same universe filter. Their picks overlap 10%.

| | buys | fires | evidence |
|---|---|---|---|
| `scanner/src/momentum.py` | what is already moving hardest | most days, ~80 candidates | **solid** — 163 decision dates, top-3 returned +9.24% over 30 sessions, disjoint ticker halves 8.78 / 9.64 |
| `scanner/src/swings.py` | a stock that fell ≥12% from its 20-session high while still above its 50-day average | ~every other day, ~7 candidates | **provisional** — +10.06% but only 157 trades ≈ 5–6 independent windows |

Both rank the survivors, neither models an exit. Every row carries an entry, a
stop and a target; deciding when to sell is yours. Measured holding period:
5 sessions +1.75%, 15 +6.50%, **30 +11.06%**. Six weeks, not six days.

**Follow momentum. Audit swings.** The swing rule's winning cell is also its
thinnest — the shallower `dip>=8%` variant measured *worse*, which is the shape
of an overfit even though both splits were clean.

## What is running

All in GCP project `sudarshan-442212`, region `europe-west1`,
schedulers in `asia-south1`, all services at **min-instances 0**.

```
09:07  momentum       previous complete session, actionable pre-open
09:10  swings         same, dip list
10:00  momentum       intraday breakouts, same-day entry
14:00  momentum
17:00  forward-test   grades BOTH tables, every weekday
```

BigQuery `data_options`: `momentum`, `swings`, `signal_outcomes`.

Note `swings` holds momentum-era rows up to 2026-09-22 and swing-era rows after.
`Bounce_Median IS NOT NULL` identifies swing rows. Left deliberately unlabelled.

Cost: inside the Cloud Run free tier. The only thing billing is ~504 MB of
stored container images (~₹5/month). `min-instances = 1` was once 95% of a
₹1,069 bill — if a bill appears, check that first.

## The freeze

No config changes, no new strategies, until `signal_outcomes` has enough graded
rows. That is the point of the freeze: everything decided so far rests on
backtests, and backtests have been wrong — the sweep that set the shortlist to
10 was measuring date-groups of 5 candidates when a live run has 75, off by a
factor of two, and had already been written into the config as settled.

A previous freeze (2026-09-20) lasted two days because every change had numbers
behind it. If a backtest argues for a change before there is live data, the
answer is no.

Exempt: a genuine bug, a data-integrity failure, or the grader breaking.

## What December answers

```bash
python scanner/src/forward_test.py --report --horizon 30
```

- Does the top 3 actually beat ranks 4–25, on live signals?
- Does the swing scanner beat momentum, or was +10.06% the thin slice?
- Does `Bounce_Median` predict, or was it fitted to 157 trades?

None of these can be answered by another backtest.

## Do this regardless

**Size every position the same.** Equal-weighting the existing 19 positions
turns −₹10,412 into **+₹6,506** — ₹16,918 from sizing alone, same stocks, same
entries. That is larger than any edge either scanner has demonstrated, and it
needs no validation. Concentration is the hole in the P&L: BSE alone was 30.8%
of capital and −₹26,552.

## Things known to be wrong

- `Today_Range` reports magnitude, not direction — CUPID showed 7.5% on a day
  it fell 3.32%. Display only, feeds no gate.
- On the swing list `Action` is always BUY and `Setup_Type` always
  `deep_pullback` — both are determined by the gate that admitted the row.
  The list itself is the signal.
- `position_check.py` still replays the retired "sell at +3%" rule under its
  own `LEGACY_*` constants. Its scheduler is PAUSED.
- Yahoo runs 1–3 sessions behind for most of the market. `Bar_Date` records
  which session was actually used — read it rather than assuming it is today.

## Why the config comments are long

Each threshold carries the sweep that set it. They exist so a number is not
re-litigated from intuition. Three findings were retracted in one session for
looking good on one split and failing another — the comments are what stops
that repeating.
