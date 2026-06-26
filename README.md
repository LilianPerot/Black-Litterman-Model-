# Black-Litterman-Model-

A Black-Litterman asset allocation engine built to go a step further than a textbook
Markowitz mean-variance optimizer, anchored to a real market equilibrium prior instead
of noisy historical averages, then tilted with the user's own views, confidences, and fee
structure. Runs on real, live market data (via `yfinance`) and is driven directly from
Excel (via `xlwings`).

> **Status: work in progress.** The model runs end-to-end and produces stable, sane
> output, but the Sharpe ratios it currently outputs on real asset universes are lower
> than I'd like. I'm actively investigating whether that's the views/confidence
> calibration, the covariance estimation, or simply the statistical reality of the
> tested universes. Feedback welcome

# Why Black-Litterman, not plain Markowitz

Mean-variance optimization (Markowitz, 1952) is the standard entry point to portfolio
theory, but feeding it raw historical average returns is a known trap: small estimation
errors get amplified into extreme, unstable weights that overweight whatever happened to
do well in the lookback window. Black-Litterman addresses this by starting from a
market-implied equilibrium prior (what returns *would* have to be, given current market
weights and risk aversion) and only deviating from it where the user expresses an
explicit, confidence-weighted view.

# What this implementation does differently

- **Real market-implied risk aversion (δ).** δ is computed from real S&P 500 (or any
  configurable benchmark index) price history — `δ = (market excess return) / (market
  variance)`, per He & Litterman (1999) — instead of an internal proxy built from the
  candidate assets themselves, which would make δ swing depending on whatever universe
  happens to be under test.
- **Capped market-cap weighting.** When market caps are used to build the equilibrium
  prior, no single asset's weight is allowed to dominate the reconstructed "market"
  portfolio (capped per asset, redistributed iteratively). If reliable market caps
  aren't available for every asset in the universe, the model falls back to an explicit
  equal-weight prior rather than silently imputing a number.
- **Ledoit-Wolf shrinkage covariance** instead of a raw sample covariance matrix.
- **Idzorek's method** to convert percentage view-confidences directly into the view
  uncertainty matrix (Ω), instead of requiring a hand-built uncertainty matrix.
- **Fee-adjusted expected returns** management fees and amortized entry fees are
  subtracted from BL-adjusted returns before optimization.
- **Walk-forward risk metrics** Max Drawdown is reported both as a walk-forward,
  no-look-ahead estimate (rolling 1-year windows, equal weights) and as a look-ahead
  reference value, so the two aren't conflated.
- **Benchmark-relative reporting** Jensen's alpha and beta of the realized portfolio
  against the same real index used to compute δ, so performance is judged against the
  market, not against the model's own prior.
- **Full transparency on data fallbacks** every run reports exactly which tickers fell
  back from live `yfinance` data to Excel-provided historical volatility or a risk-level
  default, instead of silently mixing sources.

# Requirements

pandas
numpy
scipy
scikit-learn
yfinance
xlwings
PyPortfolioOpt
matplotlib

# Output

- A **Results** sheet in the workbook with portfolio-level metrics, per-asset analysis,
  and a summary table.
- A PNG dashboard (`BL_Analytics_Charts.png`) saved next to the workbook, including the
  asset weight breakdown, BL return vs. market-implied prior, and the full efficient
  frontier (random portfolio cloud, GMV, max-Sharpe tangent portfolio, capital market
  line, and the optimized portfolio's position on it).

# Known limitations / open questions

- **Sharpe ratios are currently lower than expected** across the real asset universes
  I've tested. This survives a check of the obvious suspects (δ calibration, the
  Ledoit-Wolf covariance, the Sharpe formula itself, Idzorek's τ-independence), all of
  which check out against the literature and PyPortfolioOpt's own documented behavior.
  If you've pushed Black-Litterman past toy examples and have hit something similar,
  I'd genuinely like to hear what it turned out to be.
- Market-cap-based weighting is all-or-nothing per run (falls back to equal-weight if
  any single ticker's cap is unavailable), this is a deliberate honesty trade-off, not
  a bug, but it does mean the prior is equal-weighted more often than market-cap-weighted
  in practice for mixed-instrument universes (e.g. ETFs alongside single stocks).
- Built around an Excel + `xlwings` workflow; not yet packaged as a standalone CLI/library.

# References

- He, G. & Litterman, R. (1999). *The Intuition Behind Black-Litterman Model Portfolios.*
- Idzorek, T. (2005). *A Step-by-Step Guide to the Black-Litterman Model.*
- [PyPortfolioOpt documentation](https://pyportfolioopt.readthedocs.io/) — covariance
  shrinkage, Black-Litterman, and Idzorek's method implementations used here.
