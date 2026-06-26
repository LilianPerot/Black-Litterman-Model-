import pandas as pd
from logger import log_message

# Names must match columns name on excel spreadsheet 
REQUIRED_COLUMNS = [
    'ID',
    'Asset',
    'Ticker',
    'Max Weight',
    'View%',
    'Trust%',
    'Entry Fees%',
    'Management Fees%',
    'Risk level',
    'Historical Volatility']


def validate_columns(df):
    df.columns = df.columns.str.strip()
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise Exception(
                f"Missing column : '{col}'. Available columns : {list(df.columns)}")

def validate_weights(df):
    total_max = df['Max Weight'].sum()
    if total_max < 1:
        raise Exception(
            f"Sum of max weights ({total_max:.2%}) est inférieure à 100%. ")
        

def validate_confidence(df):
    for val in df['Trust%']:
        if pd.isna(val):
            continue
        if val < 0 or val > 1:
            raise Exception(
                f"Trust% must be between 0 and 1 (decimal). Value found : {val}. ")


def validate_views(df):
    for val in df['View%']:
        if pd.isna(val):
            continue
        if val < -0.50 or val > 0.50:
            raise Exception(
                f"Inconsistent View% (must be between -50% and +50%). Value found : {val}")


def validate_excel(df):
    validate_columns(df)
    validate_weights(df)
    validate_confidence(df)
    validate_views(df)
    log_message("Excel validation passed.")
