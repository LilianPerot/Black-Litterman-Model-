import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt import black_litterman, risk_models, expected_returns
from pypfopt.risk_models import CovarianceShrinkage
from logger import log_message


BENCHMARK_TICKER = "^GSPC"
MAX_MARKET_WEIGHT = 0.50


def _fetch_benchmark_prices(history_years, risk_free_rate):
    """
    Downloads real benchmark index prices (BENCHMARK_TICKER) over the same
    lookback window used for the asset universe, and computes the
    market-implied risk-aversion coefficient delta from it.
    """
    try:
        period_str = f"{int(history_years)}y"
        index_data = yf.download(
            BENCHMARK_TICKER, period=period_str, auto_adjust=True, progress=False)
        if index_data.empty or 'Close' not in index_data.columns:
            raise ValueError(f"No data returned for benchmark {BENCHMARK_TICKER}")

        index_prices = index_data['Close']
        if isinstance(index_prices, pd.DataFrame):
            index_prices = index_prices.iloc[:, 0]
        index_prices = index_prices.dropna()

        delta = black_litterman.market_implied_risk_aversion(
            index_prices, risk_free_rate=risk_free_rate)
        delta = float(delta)
        log_message(
            f"Risk aversion (delta) : {delta:.4f}  "
            f"(computed from {BENCHMARK_TICKER}, real market index — "
            f"He & Litterman 1999 methodology)")
        return delta, index_prices

    except Exception as e:
        log_message(
            f"Benchmark index download failed ({e}). "
            f"Falling back to literature-standard delta=2.5.")
        return 2.5, None


def _fetch_market_caps(tickers):
    """
    Fetch current market caps via yfinance for each ticker.
    Returns {ticker: market_cap}, or None if reliable caps aren't available
    for the full universe.
    If even one ticker lacks a real cap, the whole market-cap path is abandoned in
    favour of an equal-weight prior rather than a silently fabricated number distorting every asset's prior return.
    """
    caps = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
            mcap = getattr(info, 'market_cap', None)
            if mcap and mcap > 0:
                caps[t] = float(mcap)
                log_message(f"[{t}] Market cap : {mcap:,.0f}")
            else:
                log_message(f"[{t}] Market cap unavailable.")
                caps[t] = None
        except Exception as e:
            log_message(f"[{t}] Market cap fetch error : {e}")
            caps[t] = None

    missing = [t for t, v in caps.items() if v is None]
    if missing:
        log_message(
            f"Market caps missing for {len(missing)}/{len(tickers)} ticker(s) "
            f"({missing}). Imputing a median value for these would distort "
            f"the equilibrium prior, so market-cap weighting is abandoned "
            f"for this run — falling back to an equal-weight prior instead.")
        return None

    return caps


def _cap_market_weights(market_caps, max_weight=MAX_MARKET_WEIGHT):
    """
    Converts raw market caps into market weights, then caps each weight at
    `max_weight` and renormalises so the weights still sum to 1.
    """
    tickers  = list(market_caps.keys())
    caps_arr = np.array([market_caps[t] for t in tickers], dtype=float)
    weights  = caps_arr / caps_arr.sum()

    for _ in range(len(tickers)):
        over = weights > max_weight
        if not over.any():
            break
        excess = (weights[over] - max_weight).sum()
        weights[over] = max_weight
        under = ~over
        if under.sum() == 0:
            break
        weights[under] += excess * (weights[under] / weights[under].sum())

    return dict(zip(tickers, weights))


def compute_portfolio_alpha(portfolio_returns_series, benchmark_prices, risk_free_rate):
    
    if benchmark_prices is None or portfolio_returns_series is None:
        return np.nan, np.nan

    try:
        bench_ret = benchmark_prices.pct_change().dropna()
        port_ret  = portfolio_returns_series.dropna()

        # Align on common dates
        common_idx = port_ret.index.intersection(bench_ret.index)
        if len(common_idx) < 30:
            return np.nan, np.nan

        port_ret  = port_ret.loc[common_idx]
        bench_ret = bench_ret.loc[common_idx]

        cov = np.cov(port_ret, bench_ret)[0, 1]
        var_bench = np.var(bench_ret)
        beta = cov / var_bench if var_bench > 1e-12 else np.nan

        rf_daily = risk_free_rate / 252
        port_ann  = port_ret.mean() * 252
        bench_ann = bench_ret.mean() * 252

        alpha = port_ann - (risk_free_rate + beta * (bench_ann - risk_free_rate))
        return float(alpha), float(beta)

    except Exception as e:
        log_message(f"Alpha computation failed : {e}")
        return np.nan, np.nan


def compute_black_litterman(prices, assets_df, history=5, risk_free_rate=0.02):
    """
    Compute Black-Litterman returns and covariance
    Covariance method: Ledoit-Wolf.
    """
    tickers = [str(row['Ticker']).strip() for _, row in assets_df.iterrows()]

    #  Covariance Ledoit-Wolf 
    cs         = CovarianceShrinkage(prices, frequency=252)
    cov_matrix = cs.ledoit_wolf()

    try:
        import sklearn.covariance as sk_cov
        daily_ret = prices.pct_change().dropna().values
        _, shrink_coef = sk_cov.ledoit_wolf(daily_ret)
        log_message(f"Ledoit-Wolf shrinkage coefficient : {shrink_coef:.6f}")
    except Exception:
        log_message("Ledoit-Wolf applied (shrinkage coefficient unavailable).")

    try:
        cond = np.linalg.cond(cov_matrix.values)
        log_message(f"Covariance matrix condition number : {cond:.2e}")
    except Exception:
        pass

    #  Risk aversion : calculé sur un vrai indice de marché (S&P500) 
    delta, benchmark_prices = _fetch_benchmark_prices(history, risk_free_rate)

    #  Market weights : market-cap (plafonnées) si fiables, sinon équipondéré 
    raw_caps = _fetch_market_caps(tickers)
    available_tickers = [t for t in tickers if t in prices.columns]

    if raw_caps is not None:
        caps_filtered  = {t: raw_caps[t] for t in available_tickers}
        market_weights = _cap_market_weights(caps_filtered, MAX_MARKET_WEIGHT)
        log_message(
            f"Using market-cap-derived weights (capped at "
            f"{MAX_MARKET_WEIGHT:.0%} per asset) : "
            f"{ {k: round(v, 4) for k, v in market_weights.items()} }")
    else:
        n = len(available_tickers)
        market_weights = {t: 1.0 / n for t in available_tickers}
        log_message(f"Using equal market weights (1/{n} each) — market caps unreliable.")

    market_prior = black_litterman.market_implied_prior_returns(
        market_weights,
        delta,
        cov_matrix,
        risk_free_rate=risk_free_rate)
    log_message(f"Market-implied prior : { {k: round(v, 4) for k, v in market_prior.items()} }")

    #  Views & confidences 
    views       = {}
    confidences = []

    for _, row in assets_df.iterrows():
        ticker    = str(row['Ticker']).strip()
        view_val  = float(row['View%'])  if pd.notna(row['View%'])  else 0.0
        trust_val = float(row['Trust%']) if pd.notna(row['Trust%']) else 0.50
        trust_val = max(0.01, min(0.99, trust_val))
        views[ticker] = view_val
        confidences.append(trust_val)

    log_message(f"Absolute views : {views}")
    log_message(f"Confidences : {confidences}")

    model = black_litterman.BlackLittermanModel(
        cov_matrix,
        pi=market_prior,
        absolute_views=views,
        omega='idzorek',
        view_confidences=confidences)

    bl_returns    = model.bl_returns()
    bl_covariance = model.bl_cov()

    return bl_returns, bl_covariance, market_prior, benchmark_prices
