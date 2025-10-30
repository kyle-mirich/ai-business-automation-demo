"""
Data loading utilities
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional


def load_sales_data(path: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Load sales data from CSV file

    Args:
        path: Path to CSV file

    Returns:
        Tuple of (DataFrame, error_message)
        If successful, error_message is None
        If failed, DataFrame is None and error_message contains the error
    """
    try:
        # Load CSV
        df = pd.read_csv(path)

        # Parse dates
        df['date'] = pd.to_datetime(df['date'])

        # Validate required columns
        required_cols = ['date', 'product', 'quantity', 'revenue', 'cost', 'category', 'customer_segment']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            return None, f"Missing required columns: {', '.join(missing_cols)}"

        # Validate data types
        if not pd.api.types.is_numeric_dtype(df['quantity']):
            return None, "Column 'quantity' must be numeric"
        if not pd.api.types.is_numeric_dtype(df['revenue']):
            return None, "Column 'revenue' must be numeric"
        if not pd.api.types.is_numeric_dtype(df['cost']):
            return None, "Column 'cost' must be numeric"

        return df, None

    except FileNotFoundError:
        return None, f"File not found: {path}"
    except Exception as e:
        return None, f"Error loading data: {str(e)}"


def validate_data(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validate sales data for quality and integrity

    Args:
        df: Sales DataFrame

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for null values
    if df.isnull().any().any():
        return False, "Data contains null values"

    # Check for negative values
    if (df['quantity'] < 0).any():
        return False, "Negative quantities found"
    if (df['revenue'] < 0).any():
        return False, "Negative revenue found"
    if (df['cost'] < 0).any():
        return False, "Negative cost found"

    # Check if data is empty
    if len(df) == 0:
        return False, "Data is empty"

    return True, None
