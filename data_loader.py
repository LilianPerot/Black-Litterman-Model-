import pandas as pd
import yfinance as yf
import xlwings as xw
from logger import log_message


def load_excel():
    """Read excel data from xlwings"""
    try:
        wb = xw.Book.caller()

        sheet_assets = wb.sheets['Assets']
        sheet_params = wb.sheets['Parameters']

        assets_df = sheet_assets.range('A1').options(
            pd.DataFrame, expand='table', index=False).value

        if assets_df is not None:
            # Columns name normalization 
            assets_df.columns = assets_df.columns.str.strip()
            assets_df = assets_df.dropna(subset=['Ticker'])
            assets_df = assets_df[assets_df['Ticker'].astype(str).str.strip() != '']

        horizon = sheet_params.range('B1').value or 1
        history = sheet_params.range('B2').value or 5
        risk_free_rate = sheet_params.range('B3').value or 0.02
        history = int(float(history))

        log_message("Excel data loaded successfully.")
        return wb, assets_df, horizon, history, risk_free_rate

    except Exception as e:
        raise Exception(f"Error reading Excel file : {e}")


def download_single_ticker(ticker, history):
    """Download close price from Yfinance"""
    try:
        period_str = f"{int(history)}y"
        t_clean = str(ticker).strip()

        data = yf.download(t_clean, period=period_str, auto_adjust=True, progress=False)

        if data.empty:
            log_message(f"No data returned for {t_clean}.")
            return None

        # Multi-index security
        if 'Close' in data.columns:
            series = data['Close']
            if isinstance(series, pd.DataFrame):
                if t_clean in series.columns:
                    return series[t_clean]
                else:
                    return series.iloc[:, 0]
            return series

        return None

    except Exception as e:
        log_message(f"Download error {ticker} : {e}")
        return None


def build_price_dataframe(assets_df, history):
   
    prices = pd.DataFrame()
    failed_tickers = []

    for _, row in assets_df.iterrows():
        ticker = str(row['Ticker']).strip()
        log_message(f"Downloading : {ticker}")
        series = download_single_ticker(ticker, history)
        if series is not None:
            prices[ticker] = series
        else:
            log_message(f"Unable to retrieve prices for {ticker}.")
            failed_tickers.append(ticker)
            prices = prices.dropna()
            
    return prices, failed_tickers


def apply_fees(returns, assets_df, horizon):
    """
    adjust expected return substracting management fees and entry fees 
    """
    adjusted_returns = returns.copy()

    for _, row in assets_df.iterrows():
        ticker = str(row['Ticker']).strip()

        if ticker not in adjusted_returns.index:
            continue

        entry_fees = float(row['Entry Fees%']) if pd.notna(row['Entry Fees%']) else 0.0
        mgmt_fees = float(row['Management Fees%']) if pd.notna(row['Management Fees%']) else 0.0

        # Entry fees repartition
        amortized_entry_fees = entry_fees / horizon if horizon > 0 else entry_fees

        adjusted_returns[ticker] = adjusted_returns[ticker] - mgmt_fees - amortized_entry_fees

    return adjusted_returns
