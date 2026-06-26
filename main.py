import pandas as pd
import numpy as np

from logger import log_message
from data_loader import load_excel, build_price_dataframe, apply_fees
from validators import validate_excel
from black_litterman_engine import compute_black_litterman, compute_portfolio_alpha
from optimizer import optimize_portfolio
from metrics import compute_metrics
from excel_export import export_results
from volatility import get_asset_volatility


def print_fallback_report(failed_tickers, volatility_sources, assets_df):
    sep = "=" * 55
    price_fallbacks      = [t for t in assets_df['Ticker'].str.strip() if t in failed_tickers]
    vol_excel_historical = [t for t, s in volatility_sources.items() if s == 'excel_historical']
    vol_excel_risk       = [t for t, s in volatility_sources.items() if s == 'excel_risk_level']
    vol_calculated       = [t for t, s in volatility_sources.items() if s == 'calculated']
    any_excel_used = price_fallbacks or vol_excel_historical or vol_excel_risk

    print(f"\n{sep}")
    print("  DATA SOURCE REPORT")
    print(sep)

    if not any_excel_used:
        print("  ✓ All data retrieved from yfinance.")
        print("    No fallback to Excel data was necessary.")
    else:
        print("  ⚠  Some data was sourced from the Excel file.\n")
        if price_fallbacks:
            print("  [MISSING PRICES — tickers not found on yfinance]")
            for t in price_fallbacks:
                print(f"    • {t}")
            print()
        if vol_excel_historical:
            print("  [VOLATILITY — 'Historical Volatility' column from Excel used]")
            for t in vol_excel_historical:
                print(f"    • {t}")
            print()
        if vol_excel_risk:
            print("  [VOLATILITY — 'Risk level' column from Excel used (config mapping)]")
            for t in vol_excel_risk:
                print(f"    • {t}")
            print()

    if vol_calculated:
        print("  [VOLATILITY — computed from yfinance prices]")
        for t in vol_calculated:
            print(f"    ✓ {t}")
    print(sep)

    log_message(f"Price fallback : {price_fallbacks or 'aucun'}")
    log_message(f"Historical vol fallback (Excel) : {vol_excel_historical or 'aucun'}")
    log_message(f"Risk-level vol fallback (Excel) : {vol_excel_risk or 'aucun'}")


def main():
    try:
        log_message("=" * 50)
        log_message("Starting Black-Litterman model")

        # 1. Loading excel data
        wb, assets_df, horizon, history, risk_free_rate = load_excel()

        # 2. Validation
        validate_excel(assets_df)

        # 3. Historical prices
        prices, failed_tickers = build_price_dataframe(assets_df, history)
        if prices.empty:
            raise Exception("Aucune donnée de prix récupérée.")

        # 4. Black-Litterman, benchmark index (S&P500) used for delta and
        #    prior, downloaded over the same history window and risk-free rate
        bl_returns, bl_covariance, market_prior, benchmark_prices = compute_black_litterman(
            prices, assets_df, history=history, risk_free_rate=risk_free_rate)

        # 5. subsracting fees
        adjusted_returns = apply_fees(bl_returns, assets_df, horizon)

        # 6. Volatility sources (for fallback reporting)
        daily_returns = prices.pct_change().dropna() if not prices.empty else pd.DataFrame()
        volatility_sources = {}
        for _, row in assets_df.iterrows():
            ticker = str(row['Ticker']).strip()
            _, source = get_asset_volatility(
                ticker, daily_returns,
                row.get('Historical Volatility'),
                row.get('Risk level', 4))
            volatility_sources[ticker] = source

        # 7. Optimization 
        weights = optimize_portfolio(
            adjusted_returns, bl_covariance, assets_df,
            risk_free_rate=risk_free_rate)

        # 8. all metrics
        max_weights = [float(row['Max Weight']) for _, row in assets_df.iterrows()]

        # Portfolio's realised historical returns (weighted, on available
        # tickers) vs the real benchmark index — used to report alpha/beta.
        # This is purely informational (reporting), not an input to the BL
        # optimisation itself.
        portfolio_alpha, portfolio_beta = np.nan, np.nan
        if benchmark_prices is not None and not prices.empty:
            available = [t for t in adjusted_returns.index if t in prices.columns]
            if available:
                w_avail = pd.Series(weights, index=adjusted_returns.index).loc[available]
                w_avail = w_avail / w_avail.sum() if w_avail.sum() > 0 else w_avail
                port_daily_ret = (prices[available].pct_change() * w_avail).sum(axis=1)
                portfolio_alpha, portfolio_beta = compute_portfolio_alpha(
                    port_daily_ret, benchmark_prices, risk_free_rate)
                log_message(
                    f"Portfolio alpha vs benchmark : {portfolio_alpha:.4f}  "
                    f"(beta : {portfolio_beta:.4f})")

        metrics = compute_metrics(
            weights,
            adjusted_returns,
            bl_covariance,
            risk_free_rate=risk_free_rate,
            prices=prices,
            market_prior=market_prior,
            assets_df=assets_df,
            max_weights=max_weights,
            portfolio_alpha=portfolio_alpha,
            portfolio_beta=portfolio_beta,)

        # 9. Excel export 
        export_results(wb, assets_df, weights, metrics)

        log_message("Optimisation completed successfully.")
        log_message("=" * 50)

        # 10. Fallback reporting
        print_fallback_report(failed_tickers, volatility_sources, assets_df)

        input("\nPress Enter to close...")

    except Exception as e:
        log_message(f"CRITICAL ERROR : {e}")
        print("\n>>> ERROR <<<")
        print(str(e))
        import traceback
        traceback.print_exc()
        input("\nPress Enter to close...")


if __name__ == '__main__':
    main()
