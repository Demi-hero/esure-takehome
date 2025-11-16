"""Main pipeline: preprocess data, load XGBoost model, predict risk, and map to levels.

Usage example:
    python main.py --input data.csv --model Models/2025-11-15/xgb_modle.json

This script expects the workspace to have `model_functions.py` with preprocessing helpers.
"""

import argparse
import sys
import datetime as dt
import polars as pl
import numpy as np
import xgboost as xgb
import model_functions as mf

# Debugging paths
input_file = '../data/home_insurance.parquet'
model_path = '../Models/2025-11-15/xgb_modle.json'

def parse_args():
    p = argparse.ArgumentParser(description='Predict risk from input CSV using an XGBoost JSON model')
    p.add_argument('--input', '-i', required=True, help='Input CSV/Parquet file path')
    p.add_argument('--model', '-m', required=True, help='XGBoost model path (JSON)')
    return p.parse_args()

def predict_file(input_file: str,
                 model_path: str) -> pl.DataFrame:
    """Read input CSV, run preprocessing, predict probabilities and assign risk levels.

    Returns a Polars DataFrame with predictions appended.
    """
    # Preprocess features
    df, model_data ,model = mf.preprocessing(file_path=input_file, model_path=model_path)
    # Predict
    dmatrix_features = xgb.DMatrix(model_data.to_numpy(), feature_names=model_data.columns)
    preds = model.predict(dmatrix_features)
    df = df.with_columns(pl.Series('predicted_risk', preds))
    
    return df


def assign_risk_levels(df: pl.DataFrame, pred_col: str, thresholds: dict|None = None) -> pl.DataFrame:
    """Assign risk levels based on predicted probabilities."""
    
    if thresholds is None:
        # Calculate percentile-based thresholds
        probs = df[pred_col].to_numpy()
        thresholds = {
            'high': np.percentile(probs, 95),      # Top 5%
            'medium': np.percentile(probs, 80),    # 80th-95th percentile
            'low': np.percentile(probs, 50),       # 50th-80th percentile
        }

    df = df.with_columns(
    pl.when(pl.col(pred_col) >= thresholds['high'])
      .then(pl.lit('high'))
      .when(pl.col(pred_col) >= thresholds['medium'])
      .then(pl.lit('medium'))
      .when(pl.col(pred_col) >= thresholds['low'])
      .then(pl.lit('low'))
      .otherwise(pl.lit('very low'))
      .alias('risk_level')
)
    return df

def main():
    args = parse_args()
    try:
        data = predict_file(args.input, args.model)
        # debug out
        # data = predict_file(input_file, model_path)
        print("Prediction sample (first 5 rows):")
        print(data.head(5))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise
    
    risk_assigned = assign_risk_levels(data, 'predicted_risk')
    print("Risk level assignment sample (first 5 rows):")
    print(risk_assigned.head(5))

    customers_to_investigate = risk_assigned.filter(pl.col('risk_level').is_in(['high', 'medium']))
    customers_to_investigate.write_parquet(f'../data/{dt.date.today()}/customers_to_investigate.parquet')

if __name__ == '__main__':
    main()

