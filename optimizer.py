import numpy as np
from scipy.optimize import minimize
from logger import log_message


SLSQP_OPTIONS = {
    "ftol":    1e-12,
    "eps":     1e-10,
    "maxiter": 2_000,}

# Number of random starting points for the multi-start strategy.
# Each restart uses a different random Dirichlet draw as initial weights.

N_RESTARTS = 30

RNG = np.random.default_rng(seed=42)   # reproducible restarts


def optimize_portfolio(expected_returns, covariance, assets_df, risk_free_rate=0.0):
    """
    Calculate optimal weights maximizing Sharpe ratio under max_weight and 100% invested constraint.
    """
    n_assets = len(assets_df)

    # 1. Weight between 0% and max weight
    bounds = tuple(
        (0.0, float(row['Max Weight']))
        for _, row in assets_df.iterrows())

    # 2. Weight sum = 100%
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]


    cov_arr  = np.array(covariance)
    ret_arr  = np.array(expected_returns)

    def portfolio_vol(weights):
        return float(np.sqrt(weights @ cov_arr @ weights))

    def negative_sharpe(weights):
        vol = portfolio_vol(weights)
        if vol < 1e-12:
            return 0.0
        return -(float(ret_arr @ weights) - risk_free_rate) / vol

    #  Build starting points 
    starts = [np.full(n_assets, 1.0 / n_assets)]
    for _ in range(N_RESTARTS):
        raw  = RNG.dirichlet(np.ones(n_assets))
        # Clip to respect Max Weight bounds, then renormalise
        upper = np.array([float(row['Max Weight']) for _, row in assets_df.iterrows()])
        raw   = np.clip(raw, 0.0, upper)
        total = raw.sum()
        starts.append(raw / total if total > 0 else np.full(n_assets, 1.0 / n_assets))

    #  Multi-start optimisation 
    best_result  = None
    best_sharpe  = np.inf       # we minimise negative_sharpe, so lower = better
    n_success    = 0

    for i, w0 in enumerate(starts):
        res = minimize(
            negative_sharpe,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options=SLSQP_OPTIONS,)
        if res.success:
            n_success += 1
            if res.fun < best_sharpe:
                best_sharpe = res.fun
                best_result = res

    log_message(
        f"Multi-start optimisation: {n_success}/{len(starts)} converged. "
        f"Best Sharpe = {-best_sharpe:.6f}")

    if best_result is None:
        # All restarts failed, attempt a final rescue with relaxed tolerance
        log_message("WARNING: all restarts failed, attempting rescue with relaxed ftol.")
        rescue = minimize(
            negative_sharpe,
            np.full(n_assets, 1.0 / n_assets),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-8, "maxiter": 5_000},)
        if not rescue.success:
            raise Exception(f"Optimisation failed : {rescue.message}")
        return rescue.x

    return best_result.x
