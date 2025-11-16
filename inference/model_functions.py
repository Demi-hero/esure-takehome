
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (confusion_matrix, roc_auc_score, f1_score, 
                            average_precision_score, recall_score, precision_score)
import xgboost as xgb
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')


def preprocess_data(df: pl.DataFrame, target_col: str) -> pl.DataFrame:
    """
    Preprocess insurance data using Polars expressions.
    
    Parameters:
    -----------
    df : pl.DataFrame - Raw data
    target_col : str - Target column name
    
    Returns:
    --------
    Preprocessed pl.DataFrame
    """

    categorical_cols = [col for col in df.columns 
                       if df[col].dtype == pl.Utf8 and col != target_col]
    
    if categorical_cols:
        print("Encoding categorical columns:")
        expressions = []
        
        for col in categorical_cols:
            n_unique = df[col].n_unique()
            print(f"  - {col}: {n_unique} unique values")
            # Cast to categorical then to integer codes
            expressions.append(
                pl.col(col).cast(pl.Categorical).to_physical().alias(col)
            )
        
        # Apply all transformations at once (efficient!)
        non_cat_cols = [col for col in df.columns if col not in categorical_cols]
        df = df.with_columns(expressions)


        for col in categorical_cols:
            if col != target_col:
                nunique = df[col].n_unique()
                if nunique > 50:
                    print(f"{col}: {nunique} unique values (high cardinality)")

    return df


def prepare_modeling_data(df: pl.DataFrame, 
                         target_col: str = 'lapsed_flag',
                         test_size: float = 0.2,
                         random_state: int = 42):
    """
    Prepare data for modeling - converts Polars → NumPy for sklearn.
    
    Parameters:
    -----------
    df : pl.DataFrame
    target_col : str - Binary target column
    test_size : float - Proportion for test set
    random_state : int - For reproducibility
    
    Returns:
    --------
    X_train, X_test, y_train, y_test, feature_names
     """
    
    # Validate target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found!")
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col != target_col]
    
    # Convert to numpy (this is where Polars → sklearn happens)
    X = df.select(feature_cols).to_numpy()
    y = df[target_col].to_numpy()
    
    print(f"Dataset prepared:")
    print(f"  Total samples: {len(df):,}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Target distribution:")
    
    n_not_lapsed = (y == 0).sum()
    n_lapsed = (y == 1).sum()
    print(f"    Not Lapsed (0): {n_not_lapsed:,} ({n_not_lapsed/len(y)*100:.1f}%)")
    print(f"    Lapsed (1): {n_lapsed:,} ({n_lapsed/len(y)*100:.1f}%)")
    
    imbalance_ratio = n_not_lapsed / n_lapsed if n_lapsed > 0 else 0
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    if imbalance_ratio > 10:
        print("High imbalance - consider class weights")
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    ) # Startifying against more than the target column would be worth exploring in future.

    # Summary 
    print(f"\nTrain set: {len(X_train):,} samples")
    print(f"  Not Lapsed: {(y_train==0).sum():,} ({(y_train==0).mean()*100:.1f}%)")
    print(f"  Lapsed: {(y_train==1).sum():,} ({(y_train==1).mean()*100:.1f}%)")
    
    print(f"\nTest set: {len(X_test):,} samples")
    print(f"  Not Lapsed: {(y_test==0).sum():,} ({(y_test==0).mean()*100:.1f}%)")
    print(f"  Lapsed: {(y_test==1).sum():,} ({(y_test==1).mean()*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test, feature_cols



def calculate_class_weights(y_train):
    """Calculate class weights for imbalanced data."""
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    scale_pos_weight = n_negative / n_positive

    print("CLASS WEIGHT CALCULATION")
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")
    print(f"\nThis makes the model {scale_pos_weight:.1f}x more sensitive to lapse")
    
    return scale_pos_weight


def train_xgboost(X_train, y_train, scale_pos_weight=None, 
                 use_cv=False, random_state=42):
    """Train XGBoost model."""
    print(f"\n{'='*60}")
    print("TRAINING XGBOOST")
    print(f"{'='*60}")
    
    params = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': random_state,
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'verbosity': 0
    }
    
    if scale_pos_weight is not None:
        params['scale_pos_weight'] = scale_pos_weight
        print(f"Using scale_pos_weight={scale_pos_weight:.2f}")
    
    model = xgb.XGBClassifier(**params)
    
    if use_cv:
        print("\nRunning 5-fold CV...")
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
            scoring='roc_auc',
            n_jobs=-1
        )
        print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    model.fit(X_train, y_train, verbose=False)
    print("Training complete")
    
    return model


def train_lightgbm(X_train, y_train, scale_pos_weight=None, 
                  use_cv=False, random_state=42):
    """Train LightGBM model."""
    print(f"\n{'='*60}")
    print("TRAINING LIGHTGBM")
    print(f"{'='*60}")
    
    params = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': random_state,
        'verbose': -1
    }
    
    if scale_pos_weight is not None:
        params['scale_pos_weight'] = scale_pos_weight
        print(f"Using scale_pos_weight={scale_pos_weight:.2f}")
    
    model = LGBMClassifier(**params)
    
    if use_cv:
        print("\nRunning 5-fold CV...")
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
            scoring='roc_auc',
            n_jobs=-1
        )
        print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    model.fit(X_train, y_train)
    print("Training complete")
    
    return model


def evaluate_model(model, X_test, y_test, model_name="Model", threshold=0.5):
    """Comprehensive model evaluation."""
    print(f"\n{'='*60}")
    print(f"EVALUATING {model_name.upper()}")
    print(f"{'='*60}")
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  Average Precision: {avg_precision:.4f}")
    print(f"  Threshold: {threshold:.2f}")
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"                Predicted")
    print(f"                No    Yes")
    print(f"Actual No    {tn:6d}  {fp:6d}")
    print(f"       Yes   {fn:6d}  {tp:6d}")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"  Precision: {precision:.3f} (of flagged, % actually lapsed)")
    print(f"  Recall: {recall:.3f} (% of lapses caught)")
    print(f"  F1-Score: {f1:.3f}")
    print(f"  Specificity: {specificity:.3f}")
    print(f"\n  True Positives: {tp:,} (caught lapses)")
    print(f"  False Positives: {fp:,} (false alarms)")
    print(f"  False Negatives: {fn:,} (missed lapses)")
    print(f"  True Negatives: {tn:,}")
    
    find_optimal_threshold(y_test, y_pred_proba, threshold=threshold)
    
    return {
        'model_name': model_name,
        'auc': auc,
        'avg_precision': avg_precision,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def find_optimal_threshold(y_test, y_pred_proba, metric='f1', threshold=0.5):
    """Find optimal classification threshold."""
    thresholds = np.arange(0.1, 0.9, 0.05)
    scores = []
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_test, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_test, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_test, y_pred, zero_division=0)
        
        scores.append(score)
    
    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_score = scores[optimal_idx]
    
    print(f"  Optimal threshold ({metric}): {optimal_threshold:.2f}")
    print(f"  Score at optimal: {optimal_score:.3f}")
    
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_optimal)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"  At {optimal_threshold:.2f}: catch {tp}/{tp+fn} lapses, {fp} false alarms")

    y_pred_set = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_set)
    tn, fp, fn, tp = cm.ravel()

    print(f"  At {threshold:.2f}: catch {tp}/{tp+fn} lapses, {fp} false alarms")


def analyze_feature_importance(model, feature_names, top_n=20, plot=True):
    """Analyze feature importance - returns Polars DataFrame."""
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE")
    print(f"{'='*60}")
    
    importance = model.feature_importances_
    
    # Create Polars DataFrame (not pandas!)
    importance_df = pl.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort('importance', descending=True)
    
    # Add cumulative importance using Polars expressions
    importance_df = importance_df.with_columns([
        pl.col('importance').cum_sum().alias('cumulative_importance'),
        (pl.col('importance').cum_sum() / pl.col('importance').sum() * 100)
        .alias('cumulative_pct')
    ])
    
    print(f"\nTop {min(top_n, len(importance_df))} Features:")
    print(importance_df.head(top_n))
    
    # Key insights
    features_80 = len(importance_df.filter(pl.col('cumulative_pct') <= 80))
    features_95 = len(importance_df.filter(pl.col('cumulative_pct') <= 95))
    
    print(f"\n💡 Insights:")
    print(f"   - Top {features_80} features → 80% importance")
    print(f"   - Top {features_95} features → 95% importance")
    print(f"   - Bottom {len(feature_names) - features_95} features → 5% importance")
    
    if plot:
        top_df = importance_df.head(top_n)
        features = top_df['feature'].to_list()
        importances = top_df['importance'].to_list()
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        plt.barh(range(len(features)), importances, color=colors)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance', fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    return importance_df


def load_xgb_booster(model_path: str) -> xgb.Booster:
	"""Load an XGBoost model from JSON (or other supported formats).

	Returns an xgboost.Booster instance.
	"""
	booster = xgb.Booster()
	booster.load_model(model_path)
	return booster


def preprocessing(file_path: str, model_path: str) -> tuple[pl.DataFrame, pl.DataFrame, xgb.Booster]:
    """Main preprocessing function to read data and select features."""
    if file_path.lower().endswith('.parquet'):
        df = pl.read_parquet(file_path)
    elif file_path.lower().endswith('.csv'):
        df = pl.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Load model
    booster = xgb.Booster()
    booster.load_model(model_path)
    features = getattr(booster, 'feature_names', None)
    print(features)
    if features is None:   
        raise ValueError("Model does not have feature names stored.")
	
	# Preprocess
	# Strip WoE from feature lis
    raw_features = [x[4:] if  x.startswith('WoE') else x for x in features]
    
    model_data = df.select(raw_features)

    model_data = yes_no_encoding(model_data)
    
    model_data = woe_encoding(model_data, model_path, feature_list=raw_features)
    model_data = model_data.select(features)

    return df, model_data, booster


def yes_no_encoding(df: pl.DataFrame) -> pl.DataFrame:
    """Encode 'Yes'/'No' columns to 1/0 using Polars expressions."""
    yes_no_cols = [col for col in df.columns if df[col].dtype == pl.Utf8]
    
    for col in yes_no_cols:
        unique_vals = df[col].unique().to_list()
        if set(unique_vals).issubset({'Y', 'N', None}):
            df = df.with_columns(
                pl.when(pl.col(col) == 'Y')
                .then(1)
                .when(pl.col(col) == 'N')
                .then(0)
                .otherwise(None)
                .cast(pl.Int8)
                .alias(col)
            )
    
    return df


def woe_encoding(df: pl.DataFrame, model_path: str, feature_list: list) -> pl.DataFrame:
    """Apply WoE encoding to categorical columns using Polars expressions."""
    
    woe_path = "/".join(model_path.split('/')[:-1] + ['woe_values.parquet'])
    woe_values = pl.read_parquet(woe_path)
    for col_name in feature_list:
        woe_map = woe_values.filter(pl.col('variable') == col_name)
        if woe_map.is_empty():
            continue 
        mapping_dict = dict(zip(woe_map['var'].to_list(), woe_map['WoE'].to_list()))
        
        df = df.with_columns(
            pl.col(col_name)
            .replace(mapping_dict, default=None).
            alias(f"WoE_{col_name}")).drop(col_name)
    return df