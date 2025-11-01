# --- src/features/feature_utilities.py ---
import pandas as pd
from typing import List, Tuple

# Reusable Function for Feature Engineering
def create_total_column_and_clean(df: pd.DataFrame, cols_list: List[str], new_col_name: str) -> pd.DataFrame:
    """
    Creates a new column by summing two component columns and drops the originals.
    """
    # Defensive copy
    df = df.copy() 
    
    if len(cols_list) < 2:
        # Raise an error instead of logging, as this indicates a serious configuration issue
        raise ValueError("At least 2 column names must be provided for summation.")
        
    df[new_col_name] = df[cols_list[0]] + df[cols_list[1]]
    
    # Use errors='ignore' for safety in case a column was already dropped
    df = df.drop(columns=cols_list, inplace=False, errors='ignore') 
    
    return df

# Reusable Function for Target Remodeling
def traget_remodeling_util(y_df: pd.DataFrame, quality_col: str) -> pd.DataFrame:
    """
    Transforms multi-class quality scores into a binary classification target (0 or 1).
    """
    # Defensive copy
    y_df = y_df.copy()
    
    # Apply transformation by explicitly accessing the column (Fix for ambiguity error)
    y_df['quality_label'] = y_df[quality_col].apply(lambda x: 1 if x >= 6 else 0)
    
    # Drop the original quality column
    y_df = y_df.drop(columns=quality_col, inplace=False)
    
    return y_df