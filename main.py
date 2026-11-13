"""polymarket-nvidia-ceo-forecast-01 entry: per-market normal-quantile forecaster.

Design (backtested on the 6 practice cases: pinball loss 0.2726 vs the
house baseline's 0.3636, winning on every practice and holdout checkpoint):

- The median stays at the checkpoint price. Price trends in this event do
  not persist — every drift variant tested (slopes, shrinkage, decaying
  drift) lost on both practice and holdout — so there is no drift term.
- The spread widens sub-diffusively with horizon, spread(h) = sigma *
  h**0.45: prices here are mean-reverting, so uncertainty grows more
  slowly than a random walk's sqrt(h).
- sigma is the sample standard deviation of the last 120 one-minute price
  moves, floored at 0.0002 and capped at 0.005, so thin or wild markets
  still get a sane spread.
- Quantiles are the standard normal's at 5/25/50/75/95%, clamped to
  [0, 1] and rounded to 6 decimals.

Reads every <id>.in in $MAKEFUN_INPUT_DIR (default /input), writes one
<id>.out per case to $MAKEFUN_OUTPUT_DIR (default /output). Standard
library only; no network, no randomness, no clock. Deterministic.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]

WINDOW = 120          # last 1-minute diffs used for sigma
SIGMA_FLOOR = 0.0002  # per-minute spread floor
SIGMA_CAP = 0.005     # per-minute spread cap
ALPHA = 0.45          # spread growth exponent (mean-reverting: < 0.5)


def price6(v):
    """Clamp to [0, 1], round to 6 decimals, never -0."""
    return round(min(1.0, max(0.0, v)) * 1e6) / 1e6 + 0


def market_sigma(prices):
    """Sample std of the last WINDOW one-minute diffs, clamped."""
    diffs = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    recent = diffs[-WINDOW:]
    n = len(recent)
    if n < 2:
        return SIGMA_FLOOR
    mean = sum(recent) / n
    var = sum((d - mean) ** 2 for d in recent) / (n - 1)
    sigma = math.sqrt(var)
    return min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)


def horizon_spread(sigma, h):
    """Spread h minutes after the checkpoint (sub-diffusive growth)."""
    return sigma * h ** ALPHA


def quantile_row(x, spread):
    """One row of five quantiles around the checkpoint price x."""
    return [price6(x + z * spread) for z in Z]


def forecast_market(prices, horizon):
    """Horizon rows (one per minute after the checkpoint) of 5 quantiles."""
    if not prices:
        raise ValueError("market has no prices")
    x = prices[-1]
    sigma = market_sigma(prices)
    return [quantile_row(x, horizon_spread(sigma, h)) for h in range(1, horizon + 1)]


def forecast_case(case):
    if case.get("quantiles") != QUANTILES:
        raise ValueError("unexpected quantiles: %r" % (case.get("quantiles"),))
    horizon = case["horizon_minutes"]
    forecasts = []
    for market in case["markets"]:
        forecasts.append(
            {"id": market["id"], "q": forecast_market(market["prices"], horizon)}
        )
    return {"forecasts": forecasts}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for in_path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(in_path.read_text(encoding="utf-8"))
        out = forecast_case(case)
        out_path = OUTPUT_DIR / (in_path.stem + ".out")
        out_path.write_text(json.dumps(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
