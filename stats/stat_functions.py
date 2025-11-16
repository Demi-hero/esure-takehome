import polars as pl
import numpy as np
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind
import warnings
warnings.filterwarnings('ignore')


def chi_square_test(df: pl.DataFrame, categorical_col: str, target_col: str ,
                     alpha: float = 0.05, truncated:bool = False) -> dict:
    """
    Chi-square test
    """
    # Polars Crosstab for contingency table (efficient and handles nulls)
    contingency_table = df.pivot(
        index=categorical_col, 
        columns=target_col, 
        values=df.columns[0],
        aggregate_function="count"
    ).fill_null(0).to_pandas().iloc[:, 1:]
    
    # Check for empty groups (required for chi2_contingency)
    if contingency_table.empty or min(contingency_table.shape) <= 1:
        return {
            'variable': categorical_col, 
            'test': 'Chi-Square', 
            'p_value': 1.0, 
            'significant': False, 
            'interpretation': f"NOT SIGNIFICANT - Data insufficient for {categorical_col}"
        }

    # Perform chi-square test (using SciPy/Pandas)
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Calculate Cramér's V (effect size)
    n = contingency_table.values.sum()
    min_dim = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim))
    
    # Calculate percentage breakdown for each category (Polars efficient way)
    percentages = df.group_by(categorical_col).agg(
        total_count=pl.len(),
        risk_count=pl.col(target_col).sum()
    ).with_columns(
        (pl.col('risk_count') / pl.col('total_count') * 100).alias('Lapse_Rate_%')
    ).sort(pl.col('Lapse_Rate_%'), descending=True)

    # Convert contingency table back to Polars for clean output
    contingency_pl = pl.from_pandas(contingency_table)
    contingency_pl = contingency_pl.rename({str(col): f"Group_{col}" for col in contingency_pl.columns})
    
    if truncated:
        result = {'variable': categorical_col,
                  'significant': p_value < alpha,
                  'cramers_v': interpret_cramers_v(cramers_v),}
    else:
        result = {
            'variable': categorical_col,
            'test': 'Chi-Square',
            'statistic': chi2,
            'p_value': p_value,
            'significant': p_value < alpha,
            'cramers_v': interpret_cramers_v(cramers_v),
            'percentages_df': percentages,
            'interpretation': f"{'SIGNIFICANT' if p_value < alpha else 'NOT SIGNIFICANT'} - "
                            f"{categorical_col} {'IS' if p_value < alpha else 'is NOT'} "
                            f"associated with lapse risk (p={p_value:.4f})"
        }
    
    return result


def _split_and_clean_numeric(df: pl.DataFrame, numeric_col: str, target_col: str):
    """Utility to extract clean numpy arrays for SciPy tests."""
    
    # Filter and extract to NumPy arrays for SciPy
    live_group = df.filter(pl.col(target_col) == 0)[numeric_col].drop_nulls().to_numpy()
    risk_group = df.filter(pl.col(target_col) == 1)[numeric_col].drop_nulls().to_numpy()
    
    # Check minimum required size (at least 2 for t-test, usually more recommended)
    if len(live_group) < 2 or len(risk_group) < 2:
        raise ValueError("Not enough non-missing data points in one or both groups for testing.")
        
    return live_group, risk_group


def mann_whitney_test(df: pl.DataFrame, numeric_col: str, target_col: str ,
                       alpha: float = 0.05, truncated:bool = False) -> dict:
    """Mann-Whitney U test (Polars data preparation)."""
    try:
        live_group, risk_group = _split_and_clean_numeric(df, numeric_col, target_col)
    except ValueError as e:
        return {'variable': numeric_col, 'test': 'Mann-Whitney U', 'p_value': 1.0, 'significant': False, 'interpretation': f"NOT SIGNIFICANT - {e}"}
        
    # Perform Mann-Whitney U test
    statistic, p_value = mannwhitneyu(live_group, risk_group, alternative='two-sided')

    # Descriptive statistics using Polars expressions (more efficient than NumPy/Pandas)
    stats_df = df.group_by(target_col).agg(
        mean=pl.col(numeric_col).mean(),
        median=pl.col(numeric_col).median(),
        std=pl.col(numeric_col).std()
    ).sort(target_col)
    
    stats_df = stats_df.transpose(
        include_header=True, 
        header_name='Metric', 
        column_names=['Live', 'Risk'] # Assign the names for the two groups (0 and 1)
    )

    stats_dict_transposed = stats_df.to_dict(as_series=False)
    stats_dict = {}
    metrics = stats_dict_transposed['Metric']
    for group in ['Live', 'Risk']:
        for metric, value in zip(metrics, stats_dict_transposed[group]):
            stats_dict[f'{metric}_{group.lower()}'] = value

    # Add differences (calculated on numeric values)
    stats_dict['median_diff'] = stats_dict['median_risk'] - stats_dict['median_live']
    stats_dict['mean_diff'] = stats_dict['mean_risk'] - stats_dict['mean_live']

    
    n1, n2 = len(live_group), len(risk_group)
    rank_biserial = 1 - (2 * statistic) / (n1 * n2)
    
    stats_dict['median_diff'] = stats_dict['median_risk'] - stats_dict['median_live']
    stats_dict['mean_diff'] = stats_dict['mean_risk'] - stats_dict['mean_live']
    
    if truncated:
        result = {'variable': numeric_col,
                  'significant': p_value < alpha,
                  'effect_size': interpret_rank_biserial(abs(rank_biserial))}
        
        return result
    result = {
        'variable': numeric_col,
        'test': 'Mann-Whitney U',
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < alpha,
        'rank_biserial': abs(rank_biserial),
        'effect_size': interpret_rank_biserial(abs(rank_biserial)),
        'descriptive_stats': stats_dict,
        'interpretation': f"{'SIGNIFICANT' if p_value < alpha else 'NOT SIGNIFICANT'} - "
                         f"At-Risk customers have {'higher' if stats_dict['median_diff'] > 0 else 'lower'} "
                         f"{numeric_col} (median diff: {stats_dict['median_diff']:.2f}, p={p_value:.4f})"
    }
    
    return result


def t_test(df: pl.DataFrame, numeric_col: str, target_col: str = 'target_binary', alpha: float = 0.05) -> dict:
    """Independent samples t-test """
    try:
        live_group, risk_group = _split_and_clean_numeric(df, numeric_col, target_col)
    except ValueError as e:
        return {'variable': numeric_col, 'test': 'Independent t-test', 
                'p_value': 1.0, 'significant': False, 'interpretation': f"NOT SIGNIFICANT - {e}"}

    # Perform t-test (Welch's t-test - doesn't assume equal variances)
    statistic, p_value, degs_freedom = ttest_ind(live_group, risk_group, equal_var=False)
    
    pooled_std = np.sqrt(((len(live_group) - 1) * live_group.std()**2 + 
                          (len(risk_group) - 1) * risk_group.std()**2) / 
                         (len(live_group) + len(risk_group) - 2))
    cohens_d = (risk_group.mean() - live_group.mean()) / pooled_std

    # Descriptive statistics (using Polars)
    live_mean, risk_mean = live_group.mean(), risk_group.mean()
    mean_diff = risk_mean - live_mean
    
    result = {
        'variable': numeric_col,
        'test': 'Independent t-test',
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < alpha,
        'cohens_d': abs(cohens_d),
        'effect_size': interpret_cohens_d(abs(cohens_d)),
        'live_mean': live_group.mean(),
        'risk_mean': risk_group.mean(),
        'mean_diff': risk_group.mean() - live_group.mean(),
        'interpretation': f"{'SIGNIFICANT' if p_value < alpha else 'NOT SIGNIFICANT'} - "
                         f"At-Risk customers have {'higher' if cohens_d > 0 else 'lower'} "
                         f"{numeric_col} on average (diff: {risk_group.mean() - live_group.mean():.2f}, p={p_value:.4f})"
    }

    return result


def interpret_cohens_d(d):
    """Interpret Cohen's d effect size"""
    if d < 0.2:
        return "Negligible"
    elif d < 0.5:
        return "Small"
    elif d < 0.8:
        return "Medium"
    else:
        return "Large"
    

def interpret_rank_biserial(r):
    """Interpret rank-biserial correlation effect size"""
    if r < 0.1:
        return "Negligible"
    elif r < 0.3:
        return "Small"
    elif r < 0.5:
        return "Medium"
    else:
        return "Large"
    

def interpret_cramers_v(v):
    """Interpret Cramér's V effect size"""
    if v < 0.1:
        return "Negligible"
    elif v < 0.3:
        return "Small"
    elif v < 0.5:
        return "Medium"
    else:
        return "Large"