"""Does the EARLY bucket premise actually have an edge?

The 2026-09-02 fix repaired the arithmetic that made the bucket unfireable. That
is a correctness fix and is justified on its own. It says nothing about whether
the SIGNAL is worth trading, and two strategies in this fleet (ORB, fade-bot)
have already been killed by asking that question late instead of early.

WHAT IS BEING TESTED
    "A liquid name trading on unusual volume for the time of day, whose price has
     moved only 1.5-4% and is holding near the day's high, outperforms over the
     rest of the session."

This deliberately tests the SCANNER, not the bot. The LLM's selection is not
simulated: if the candidate set it is handed has no edge, nothing it picks from
that set can have one — the same reasoning that condemned ORB. A positive result
here is necessary, not sufficient.

Constants are imported from market_scanner so the test cannot drift from the
deployed filter.

CRITERIA, FIXED BEFORE THE FIRST RUN (see verdict() — do not edit these after
seeing output; that is how you end up with a curve-fit):
  1. FREQUENCY  >= 100 signal-days, else the strategy is untradable regardless
                of expectancy and the sample cannot support a conclusion.
  2. EXPECTANCY mean NET excess return per trade > 0.
  3. SIGNIFICANCE day-level t-stat >= 2.0.
Significance is computed at the DAY level, never per trade. Fifteen names
selected by one filter on one morning are one observation of that filter, not
fifteen — treating them as independent is what inflated momentum-bot's t from
2.57 to a meaningless 4.75.

KNOWN BIASES, ALL OPTIMISTIC
  - Survivorship: the universe is built from what is liquid TODAY, so names that
    delisted or collapsed during the window are absent.
  - IEX-only bars: volumes are a consistent fraction of consolidated, so ratios
    hold, but dollar-volume floors are conservative and thin names are noisier.
  - Bracket fills are assumed AT the touched price, with no slippage beyond the
    flat haircut, and when a 30-minute bar touches both the stop and the target
    the stop is assumed first (the one pessimistic assumption here).
  - The real bot flattens at 15:55; this exits on the 16:00 close.
"""
import datetime as _dt
import os
import statistics
import sys
import time as _time
import pickle
import hashlib
from collections import defaultdict
from zoneinfo import ZoneInfo

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import market_scanner as ms
from market_scanner import (
    EARLY_MIN_RELVOL, EARLY_MIN_PCT, EARLY_MAX_PCT,
    EARLY_MIN_DOLLAR_VOLUME, REFERENCE_SYMBOLS, MIN_PACE, MAX_PACE, DATA_FEED,
)
from trader import PROFIT_TARGET, STOP_LOSS

ET = ZoneInfo("America/New_York")

MONTHS = int(os.getenv("BT_MONTHS", 12))
UNIVERSE_SIZE = int(os.getenv("BT_UNIVERSE", 800))
COST_HAIRCUT = float(os.getenv("BT_COST", 0.0010))   # 10bp round trip, market orders
BASELINE_DAYS = 20
BAR_BATCH = 40           # symbols per 30-minute-bar request


CACHE_DIR = os.getenv("BT_CACHE", "/tmp/mordy_bt_cache")


def log(msg):
    print(msg, flush=True)


def cached(key, fn):
    """Bar downloads are ~25 minutes; a second question about the same data
    should not cost that again. Keyed on the request, not on wall time."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, hashlib.sha1(key.encode()).hexdigest() + ".pkl")
    if os.path.exists(path):
        log(f"  (cached: {key[:60]})")
        with open(path, "rb") as f:
            return pickle.load(f)
    val = fn()
    with open(path, "wb") as f:
        pickle.dump(dict(val), f)
    return val


# ---------------------------------------------------------------- data loading

def build_universe(trading, data):
    """The most liquid names today. Survivorship-biased by construction."""
    tickers = ms.get_active_tickers(trading)
    snaps = ms._snapshot_all(data, tickers)
    rows = []
    for sym, sn in snaps.items():
        db = sn.daily_bar
        if not db or not db.close or not db.volume:
            continue
        if not (ms.MIN_PRICE <= db.close <= ms.MAX_PRICE):
            continue
        rows.append((db.close * db.volume, sym))
    rows.sort(reverse=True)
    universe = [s for _, s in rows[:UNIVERSE_SIZE]]
    for s in REFERENCE_SYMBOLS:
        if s not in universe:
            universe.append(s)
    return universe


def fetch_daily(data, symbols, start, end):
    """{symbol: {date: (close, volume)}} — for prev closes and volume baselines."""
    out = defaultdict(dict)
    for i in range(0, len(symbols), 200):
        batch = symbols[i:i + 200]
        try:
            res = data.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=batch, timeframe=TimeFrame.Day,
                start=start, end=end, feed=DATA_FEED)).data
        except Exception as e:
            log(f"  daily batch failed: {e}")
            continue
        for sym, bars in res.items():
            for b in bars:
                out[sym][b.timestamp.astimezone(ET).date()] = (b.close, b.volume)
        log(f"  daily {min(i + 200, len(symbols))}/{len(symbols)}")
    return out


def fetch_intraday(data, symbols, start, end):
    """{symbol: {date: [bars in regular session, chronological]}}

    A dropped batch silently removes 40 names from the sample, which is the same
    fail-to-nothing pattern this whole exercise exists to punish. Retry, and if a
    batch still cannot be fetched, record it so the report can say so instead of
    quietly testing a smaller universe than it claims.
    """
    out = defaultdict(lambda: defaultdict(list))
    dropped = []
    tf = TimeFrame(30, TimeFrameUnit.Minute)
    for i in range(0, len(symbols), BAR_BATCH):
        batch = symbols[i:i + BAR_BATCH]
        res = None
        for attempt in range(4):
            try:
                res = data.get_stock_bars(StockBarsRequest(
                    symbol_or_symbols=batch, timeframe=tf,
                    start=start, end=end, feed=DATA_FEED)).data
                break
            except Exception as e:
                log(f"  intraday batch attempt {attempt + 1} failed: {e}")
                _time.sleep(2 ** attempt)
        if res is None:
            dropped.extend(batch)
            log(f"  !! GAVE UP on {len(batch)} symbols after 4 attempts")
            continue
        for sym, bars in res.items():
            for b in bars:
                t = b.timestamp.astimezone(ET)
                # Regular session only. Extended-hours bars would corrupt both
                # the volume clock and the day's high/low.
                mins = t.hour * 60 + t.minute
                if not (9 * 60 + 30 <= mins < 16 * 60):
                    continue
                out[sym][t.date()].append(b)
        log(f"  intraday {min(i + BAR_BATCH, len(symbols))}/{len(symbols)}")
    if dropped:
        log(f"  WARNING: {len(dropped)} of {len(symbols)} symbols have NO intraday "
            f"data and are absent from the test.")
    return out


# ------------------------------------------------------------------ simulation

SLOTS = 13                      # 09:30-16:00 in 30-minute blocks
OPEN_MIN = 9 * 60 + 30


def slot_of(bar):
    """Which 30-minute block of the regular session a bar belongs to."""
    t = bar.timestamp.astimezone(ET)
    return (t.hour * 60 + t.minute - OPEN_MIN) // 30


def session_paces(intraday, avg20, date):
    """Reproduce market_scanner.market_pace() per slot, from historical bars.

    Same cross-sectional construction: what the reference basket has traded so
    far over what it trades in a full session. Indexed by SLOT rather than by
    bar position, because a symbol missing a bar would otherwise silently shift
    every later reading an half-hour earlier.
    """
    traded = [0.0] * SLOTS
    baseline = 0.0
    matched = 0
    for sym in REFERENCE_SYMBOLS:
        bars = intraday.get(sym, {}).get(date)
        avg = avg20.get(sym, {}).get(date)
        if not bars or not avg:
            continue
        for b in bars:
            s = slot_of(b)
            if 0 <= s < SLOTS:
                traded[s] += b.volume
        baseline += avg
        matched += 1
    if matched < len(REFERENCE_SYMBOLS) * 0.5 or baseline <= 0:
        return None
    out, run = [], 0.0
    for s in range(SLOTS):
        run += traded[s]
        out.append(min(max(run / baseline, MIN_PACE), MAX_PACE) if run > 0 else None)
    return out


def simulate_exit(bars, entry_idx, entry_price):
    """Walk forward through the session applying the live bracket.

    When one bar contains both the target and the stop, the stop is taken. Real
    fills are unknowable at this resolution and the pessimistic read is the
    honest one.
    """
    target = entry_price * (1 + PROFIT_TARGET)
    stop = entry_price * (1 + STOP_LOSS)
    for b in bars[entry_idx + 1:]:
        if b.low <= stop:
            return (stop - entry_price) / entry_price, "stop"
        if b.high >= target:
            return (target - entry_price) / entry_price, "target"
    return (bars[-1].close - entry_price) / entry_price, "eod"


def run():
    trading, data = ms.get_clients()
    end = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(minutes=20)
    start = end - _dt.timedelta(days=int(MONTHS * 30.5))
    daily_start = start - _dt.timedelta(days=60)   # room for the 20d baseline

    log(f"Building universe (top {UNIVERSE_SIZE} by dollar volume)...")
    universe = build_universe(trading, data)
    log(f"  {len(universe)} symbols")

    log("Fetching daily bars...")
    ck = f"daily|{MONTHS}|{UNIVERSE_SIZE}|{daily_start.date()}|{end.date()}|{len(universe)}"
    daily = defaultdict(dict, cached(ck, lambda: fetch_daily(data, universe, daily_start, end)))

    log("Precomputing prior closes and 20-day volume baselines...")
    all_dates = sorted({d for h in daily.values() for d in h})
    sess_dates = [d for d in all_dates if d >= start.date()]
    avg20 = {}
    prev_close = {}
    for sym, hist in daily.items():
        ordered = sorted(hist.items())
        a, p = {}, {}
        vols, last_close = [], None
        for d, (c, v) in ordered:
            # Both use ONLY sessions strictly before d — no lookahead.
            if len(vols) >= 10:
                a[d] = sum(vols[-BASELINE_DAYS:]) / len(vols[-BASELINE_DAYS:])
            if last_close:
                p[d] = last_close
            vols.append(v)
            last_close = c
        avg20[sym], prev_close[sym] = a, p

    log(f"Fetching 30-minute bars for {len(sess_dates)} sessions...")
    ck = f"intra|{MONTHS}|{UNIVERSE_SIZE}|{start.date()}|{end.date()}|{len(universe)}"
    intraday = defaultdict(dict, cached(ck, lambda: fetch_intraday(data, universe, start, end)))

    log("Simulating...")
    trades = []
    spy = intraday.get("SPY", {})
    signal_days = set()

    for date in sess_dates:
        spy_bars = spy.get(date)
        if not spy_bars or len(spy_bars) < 6:
            continue
        paces = session_paces(intraday, avg20, date)
        if paces is None:
            continue
        # SPY close by slot, for the benchmark leg of each excess return.
        spy_close = {}
        for b in spy_bars:
            s = slot_of(b)
            if 0 <= s < SLOTS:
                spy_close[s] = b.close
        spy_final = spy_bars[-1].close

        for sym in universe:
            bars = intraday.get(sym, {}).get(date)
            if not bars or len(bars) < 3:
                continue
            pc = prev_close.get(sym, {}).get(date)
            av = avg20.get(sym, {}).get(date)
            if not pc or not av:
                continue

            cum = 0.0
            hi, lo = float("-inf"), float("inf")
            for idx, b in enumerate(bars):
                cum += b.volume
                hi, lo = max(hi, b.high), min(lo, b.low)
                if idx >= len(bars) - 1:
                    break                      # no time left to hold
                s = slot_of(b)
                if not (0 <= s < SLOTS):
                    continue
                pace = paces[s]
                if pace is None or s not in spy_close:
                    continue

                price = b.close
                if not (ms.MIN_PRICE <= price <= ms.MAX_PRICE):
                    continue
                proj_vol = cum / pace
                if price * proj_vol < EARLY_MIN_DOLLAR_VOLUME:
                    continue
                relvol = proj_vol / av
                if relvol < EARLY_MIN_RELVOL:
                    continue
                change_pct = (price - pc) / pc * 100
                if not (EARLY_MIN_PCT <= abs(change_pct) <= EARLY_MAX_PCT):
                    continue
                rng = hi - lo
                range_pos = (price - lo) / rng if rng > 0 else 0.5
                if range_pos < 0.6:
                    continue

                ret, why = simulate_exit(bars, idx, price)
                spy_ret = (spy_final - spy_close[s]) / spy_close[s]
                # The same entry with NO bracket, held to the close. Comparing
                # the two attributes the result to the signal or to the exit —
                # ORB's ladder was right while its premise was wrong, so the
                # distinction is not academic.
                hold = (bars[-1].close - price) / price
                trades.append({
                    "date": date, "symbol": sym, "slot": s,
                    "gross": ret, "net": ret - COST_HAIRCUT,
                    "excess": (ret - COST_HAIRCUT) - spy_ret,
                    "hold_excess": (hold - COST_HAIRCUT) - spy_ret,
                    "exit": why, "relvol": relvol, "change_pct": change_pct,
                })
                signal_days.add(date)
                break      # one entry per symbol per day, as the bot would

    report(trades, sess_dates, signal_days)


# ---------------------------------------------------------------------- output

def tstat(xs):
    if len(xs) < 3:
        return 0.0
    sd = statistics.stdev(xs)
    if sd == 0:
        return 0.0
    return statistics.mean(xs) / (sd / len(xs) ** 0.5)


def verdict(n_days, mean_excess, t):
    """Criteria fixed before the first run. See module docstring."""
    checks = [
        ("FREQUENCY   >= 100 signal-days", n_days >= 100, f"{n_days}"),
        ("EXPECTANCY  mean net excess > 0", mean_excess > 0, f"{mean_excess*100:+.3f}%"),
        ("SIGNIFICANCE day-level t >= 2.0", t >= 2.0, f"{t:.2f}"),
    ]
    for label, ok, val in checks:
        log(f"  [{'PASS' if ok else 'FAIL'}] {label:34} {val}")
    return all(ok for _, ok, _ in checks)


def report(trades, sess_dates, signal_days):
    log("\n" + "=" * 68)
    log(f"SESSIONS TESTED     {len(sess_dates)}")
    log(f"SIGNAL DAYS         {len(signal_days)} "
        f"({len(signal_days)/max(len(sess_dates),1)*100:.0f}% of sessions)")
    log(f"TRADES              {len(trades)}")
    if not trades:
        log("\nNo signals fired. The premise cannot be evaluated and the filter "
            "cannot trade — that is itself a failing result.")
        log("=" * 68)
        return

    gross = [t["gross"] for t in trades]
    net = [t["net"] for t in trades]
    exc = [t["excess"] for t in trades]

    log(f"TRADES PER SIGNAL DAY {len(trades)/len(signal_days):.1f}")
    log("")
    log(f"  gross  mean {statistics.mean(gross)*100:+.3f}%   median {statistics.median(gross)*100:+.3f}%")
    log(f"  net    mean {statistics.mean(net)*100:+.3f}%   median {statistics.median(net)*100:+.3f}%")
    log(f"  excess mean {statistics.mean(exc)*100:+.3f}%   median {statistics.median(exc)*100:+.3f}%")
    log(f"  win rate (net) {sum(1 for x in net if x > 0)/len(net)*100:.1f}%")

    ex = defaultdict(int)
    for t in trades:
        ex[t["exit"]] += 1
    log(f"  exits: " + ", ".join(f"{k}={v}" for k, v in sorted(ex.items())))

    # Day-level aggregation: one observation per session, not per trade.
    byday = defaultdict(list)
    for t in trades:
        byday[t["date"]].append(t["excess"])
    day_exc = [statistics.mean(v) for _, v in sorted(byday.items())]
    t_day = tstat(day_exc)
    t_trade = tstat(exc)

    log("")
    log(f"  DAY-LEVEL   mean excess {statistics.mean(day_exc)*100:+.3f}%  t = {t_day:.2f}  (n={len(day_exc)})")
    log(f"  per-trade t = {t_trade:.2f} (n={len(exc)}) — NOT the number to judge on; "
        f"trades on one day share one signal")

    # Does the ranking carry information, or is it just "liquid stock, midday"?
    log("")
    log("  BY RELVOL DECILE (does more unusual volume mean more edge?)")
    srt = sorted(trades, key=lambda t: t["relvol"])
    k = max(len(srt) // 4, 1)
    for i, name in enumerate(["Q1 lowest", "Q2", "Q3", "Q4 highest"]):
        chunk = srt[i * k:(i + 1) * k] if i < 3 else srt[3 * k:]
        if chunk:
            log(f"    {name:11} relvol {chunk[0]['relvol']:.1f}-{chunk[-1]['relvol']:.1f}  "
                f"n={len(chunk):5d}  mean excess {statistics.mean([c['excess'] for c in chunk])*100:+.3f}%")

    log("")
    log("  BY TIME OF DAY")
    byslot = defaultdict(list)
    for t in trades:
        byslot[t["slot"]].append(t["excess"])
    for s in sorted(byslot):
        h, m = divmod(OPEN_MIN + 30 * (s + 1), 60)
        log(f"    {h:02d}:{m:02d}  n={len(byslot[s]):5d}  mean excess {statistics.mean(byslot[s])*100:+.3f}%")

    hold = [t["hold_excess"] for t in trades]
    byday_h = defaultdict(list)
    for t in trades:
        byday_h[t["date"]].append(t["hold_excess"])
    day_hold = [statistics.mean(v) for _, v in sorted(byday_h.items())]
    log("")
    log("  ATTRIBUTION — is it the signal or the +3%/-2% bracket?")
    log(f"    with bracket     mean excess {statistics.mean(exc)*100:+.3f}%  day-level t {t_day:.2f}")
    log(f"    held to close    mean excess {statistics.mean(hold)*100:+.3f}%  day-level t {tstat(day_hold):.2f}")
    log("    (if BOTH are negative the entry is the problem, and no exit rescues it)")

    log("")
    log("VERDICT (criteria fixed before the run):")
    ok = verdict(len(signal_days), statistics.mean(exc), t_day)
    log("")
    log("  ==> PREMISE SURVIVES — deploy" if ok else
        "  ==> PREMISE FAILS — do not deploy the early bucket as a trade trigger")
    log("=" * 68)


if __name__ == "__main__":
    run()
