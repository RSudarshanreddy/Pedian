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

## How the list is ordered, and how steeply

Rank matters far more than it looks. Each rank on its own, 30-session hold,
163 decision dates, entry at next open, net of costs:

| rank | trades | ret30 | median | win |
|---|---|---|---|---|
| 1 | 163 | **+18.35%** | +13.99 | 66.3% |
| 2 | 163 | **+14.80%** | +14.76 | 66.3% |
| 3 | 163 | +5.23% | +5.28 | 54.6% |
| 4 | 163 | **-2.78%** | -5.22 | **36.2%** |
| 5 | 163 | +4.34% | +1.16 | 52.8% |

Ranks 1 and 2 are the list. Rank 3 returns a third of rank 2. Rank 4 loses
money at a 36% win rate, and returned -16.56% in the second half of the
period. Below the cut is not a near-miss; it is a different quality of name.

**Open question for December: does top 2 beat top 3?** Measured, it did --
ranks 1-2 returned +16.58% with ticker halves 16.09 / 17.23, against the
deployed top 3's +12.79% and 12.72 / 12.92. NOT acted on, deliberately: it is
the same 8-month backtest, and the last `N` change made on backtest evidence
rested on a sweep that turned out to be measuring the wrong thing. Let the
live data decide.

### Two things that read as problems but are not

**Higher Move% is not always better -- EXCEPT at the top of the list.** Pooled
across all 3,983 qualifying observations the bands peak at 27-45% and fall
away (45-60% wins only 46.1% of the time). But that does NOT transfer to the
top 3: capping Move% costs return at every level (below 60% -> +8.08%, below
45% -> +8.76%, below 35% -> +4.65%, against +12.79% uncapped) and wrecks the
half-split. The 45-60% band returned +7.65% pooled but +50.39% when it was a
top-3 pick. Being the best name available today carries information the raw
band number does not.

**~86 candidates qualify but only ~27 are real.** min_typical_move_pct = 14 is
a coarse sieve; the money starts around 27% (Q3 +1.62% -> Q4 +8.78%). The
ranker does the discrimination, not the gate. The other 80-odd rows exist to
be stored, because December cannot prove the top 3 beats rank 40 without
rank 40 on record.

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
