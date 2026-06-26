import numpy as np
from config import RISK_LEVEL_MAPPING
from logger import log_message


def get_asset_volatility(ticker, returns, historical_volatility, risk_level):
    """
    Return annualized asset volatility priorizing defferent methods in case of missing data
      1. calculated volatility from yfinance
      2. historical volatility sourced in excel spreadsheet (fallback Excel)
      3. default volatility from risk level (fallback Excel)
    """
    # Priority 1 : calculate daily return (yfinance)
    try:
        if ticker in returns.columns:
            vol = returns[ticker].std() * np.sqrt(252)
            if not np.isnan(vol) and vol > 0:
                log_message(f"[{ticker}] Volatility computed from yfinance : {vol:.4f}")
                return vol, 'calculated'
    except Exception:
        pass

    # Priority 2 : historical volatility in excel
    try:
        if historical_volatility is not None and not np.isnan(float(historical_volatility)):
            vol = float(historical_volatility)
            log_message(f"[{ticker}] Fallback → Historical volatility from Excel : {vol:.4f}")
            return vol, 'excel_historical'
    except (TypeError, ValueError):
        pass

    # Priority 3 : risk level mapping
    default_vol = RISK_LEVEL_MAPPING.get(int(risk_level), 0.12)
    log_message(f"[{ticker}] Fallback → Default volatility from Risk level {int(risk_level)} : {default_vol:.4f}")
    return default_vol, 'excel_risk_level'
