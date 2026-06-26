import numpy as np
import pandas as pd
from config import VAR_CONFIDENCE


def compute_metrics(weights, expected_returns, covariance, risk_free_rate=0.0,
                    prices=None, market_prior=None, assets_df=None,
                    max_weights=None, portfolio_alpha=None, portfolio_beta=None):
    """
    calculate performance metrics per asset and for the portfolio
    """
    weights = np.array(weights)
    tickers = list(expected_returns.index)
    n = len(weights)

    #  Portfolio metrics 
    port_return = float(np.dot(weights, expected_returns))
    port_vol    = float(np.sqrt(np.dot(weights.T, np.dot(covariance, weights))))
    sharpe      = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0.0
    var_95      = port_return - VAR_CONFIDENCE * port_vol
    diversif    = 1.0 / float(np.sum(weights ** 2)) if np.sum(weights ** 2) > 0 else 1.0

    # Expected Shortfall (CVaR)  95%
    from scipy.stats import norm
    z = VAR_CONFIDENCE
    es_95 = -(port_return - port_vol * norm.pdf(z) / (1 - 0.95))

    #  Max Drawdown 
    # Two MDD values to avoid look-ahead bias:
    # max_dd    (walk-forward, displayed) : rolling 1-year windows with equal
    #           weights. no knowledge of the future BL weights is used.
    #           Conservative estimate of real drawdown risk.
    # max_dd_lh (look-ahead, reference)  : BL optimal weights applied
    #           retrospectively over the full history. Underestimates real
    #           risk because weights were optimised on the same data period.
    #           Kept as a reference column in the Excel table.
    max_dd    = np.nan
    max_dd_lh = np.nan

    if prices is not None and not prices.empty:
        try:
            available   = [t for t in tickers if t in prices.columns]
            daily_ret   = prices[available].pct_change().dropna()
            w_available = weights[[i for i, t in enumerate(tickers) if t in available]]
            w_available = w_available / w_available.sum()

            # Look-ahead MDD (reference — biased downward)
            port_lh   = (daily_ret * w_available).sum(axis=1)
            cum_lh    = (1 + port_lh).cumprod()
            max_dd_lh = float(((cum_lh - cum_lh.cummax()) / cum_lh.cummax()).min())

            # Walk-forward MDD: 1-year windows, quarterly step, equal weights
            n_av    = len(available)
            w_naive = np.full(n_av, 1.0 / n_av)
            window, step = 252, 63
            wf_mdds = []
            for start in range(0, len(daily_ret) - window + 1, step):
                wr  = daily_ret.iloc[start:start + window]
                pw  = (wr * w_naive).sum(axis=1)
                cw  = (1 + pw).cumprod()
                wf_mdds.append(float(((cw - cw.cummax()) / cw.cummax()).min()))
            max_dd = min(wf_mdds) if wf_mdds else max_dd_lh

        except Exception:
            pass

    portfolio_metrics = {
        'Rendement Portefeuille':    port_return,
        'Volatilite Portefeuille':   port_vol,
        'Sharpe Portefeuille':       sharpe,
        'VaR 95%':                   var_95,
        'Expected Shortfall 95%':    es_95,
        'Max Drawdown':              max_dd,       # walk-forward — displayed
        'Max Drawdown LookAhead':    max_dd_lh,   # look-ahead   — reference
        'Indice Diversification':    diversif,
        'Alpha vs Benchmark':        portfolio_alpha if portfolio_alpha is not None else np.nan,
        'Beta vs Benchmark':         portfolio_beta  if portfolio_beta  is not None else np.nan,}

    #  Asset metrics 
    cov_matrix = np.array(covariance)

    # Annualized voliatility
    asset_vols = np.sqrt(np.diag(cov_matrix))

    # Asset sharpe ratio
    asset_sharpes = np.array([
        (expected_returns.iloc[i] - risk_free_rate) / asset_vols[i]
        if asset_vols[i] > 0 else 0.0
        for i in range(n)])

    # Risk Contribution
    marginal_contrib = cov_matrix @ weights
    risk_contrib_abs = weights * marginal_contrib
    risk_contrib_pct = risk_contrib_abs / port_vol if port_vol > 0 else risk_contrib_abs

    # Prior Spread vs BL
    prior_returns = np.array([market_prior.get(t, np.nan) for t in tickers]) \
        if market_prior is not None else np.full(n, np.nan)
    bl_returns_arr = np.array(expected_returns)
    spread = bl_returns_arr - prior_returns

    per_asset_df = pd.DataFrame({
        'Ticker':                  tickers,
        'Poids Optimal':           weights,
        'Rendement Prior (CAPM)':  prior_returns,
        'Rendement BL Ajuste':     bl_returns_arr,
        'Spread BL vs Prior':      spread,
        'Volatilite':              asset_vols,
        'Sharpe':                  asset_sharpes,
        'Risk Contribution (%)':   risk_contrib_pct * 100,})

    return {
        'portfolio':   portfolio_metrics,
        'per_asset':   per_asset_df,
        # Raw arrays needed for efficient frontier in charts
        'covariance':  cov_matrix,     
        'bl_returns':  bl_returns_arr,  
        'risk_free':   risk_free_rate,  
        'tickers':     tickers,        
        'max_weights': max_weights,}    
