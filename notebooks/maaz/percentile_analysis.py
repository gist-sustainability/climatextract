"""
Percentile Analysis for Information Extraction Confidence Scores

This script analyzes confidence score distributions across different thresholds
to evaluate extraction performance at various probability cutoffs.

Usage:
    python percentile_analysis.py <run_id>

Example:
    python percentile_analysis.py 5de297ff670844d68486f9b5eed8bf92
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# Optional scope filter (set to values like ['1', '3', '2lb', '2mb'] or leave empty for all)
SCOPES_FILTER: List[str] = ['1']

# Recent years filtering configuration
RECENT_YEARS_ONLY: int = 3


def load_run_data(run_id: str, base_path: str = "../../output") -> pd.DataFrame:
    """Load error analysis data for a specific run."""
    file_path = Path(base_path) / run_id / "error_analysis_per_row.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")

    df = pd.read_csv(file_path)

    if 'value_probability' not in df.columns:
        raise ValueError(f"Run {run_id} does not have 'value_probability' column")

    return df


def extract_publication_year(report_name: str) -> int:
    """Extract publication year from report filename."""
    import re

    year_match = re.search(r'_(\d{4})_', report_name)
    if year_match:
        return int(year_match.group(1))

    year_match = re.search(r'(20\d{2})', report_name)
    if year_match:
        return int(year_match.group(1))

    return 2023


def filter_recent_years_only(df: pd.DataFrame, years_back: int = 3) -> pd.DataFrame:
    """Filter dataframe to keep only entries within recent years."""
    if 'ReportName' not in df.columns or 'year_man' not in df.columns:
        return df

    filtered_rows = []

    for report_name in df['ReportName'].unique():
        pub_year = extract_publication_year(report_name)
        report_data = df[df['ReportName'] == report_name]

        target_years = list(range(pub_year - years_back + 1, pub_year + 1))
        recent_data = report_data[report_data['year_man'].isin(target_years)]

        if len(recent_data) > 0:
            filtered_rows.append(recent_data)

    if filtered_rows:
        return pd.concat(filtered_rows, ignore_index=True)
    else:
        return pd.DataFrame()


def calculate_threshold_metrics(df: pd.DataFrame) -> Dict:
    """Calculate metrics at different probability thresholds."""
    thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999, 0.999999]

    baseline_counts = df['error_value'].value_counts()
    baseline_tn = baseline_counts.get('true_negative_value', 0)
    baseline_fn = baseline_counts.get('false_negative_value', 0)

    results = {}

    for threshold in thresholds:
        # Get data with probability scores
        prob_rows = df[df['value_probability'].notna()]
        above_threshold = prob_rows[prob_rows['value_probability'] >= threshold]
        below_threshold = prob_rows[prob_rows['value_probability'] < threshold]

        # Count classifications above threshold
        tp_count = len(above_threshold[above_threshold['error_value'] == 'true_positive_value'])
        fp_count = len(above_threshold[above_threshold['error_value'] == 'false_positive_value'])

        # Reclassify below threshold as "no extraction"
        tn_from_threshold = len(below_threshold[below_threshold['value_man'].isna()])
        fn_from_threshold = len(below_threshold[below_threshold['value_man'].notna()])

        tn_count = baseline_tn + tn_from_threshold
        fn_count = baseline_fn + fn_from_threshold

        # Calculate metrics
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Calculate conversion metrics
        tp_below_df = below_threshold[below_threshold['error_value'] == 'true_positive_value']
        fp_below_df = below_threshold[below_threshold['error_value'] == 'false_positive_value']
        tp_to_fn = len(tp_below_df)
        fp_to_fn = len(fp_below_df[fp_below_df['value_man'].notna()])

        results[threshold] = {
            'tp_count': tp_count,
            'fp_count': fp_count,
            'tn_count': tn_count,
            'fn_count': fn_count,
            'tp_to_fn': tp_to_fn,
            'fp_to_fn': fp_to_fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

    return results


def filter_tp_fp_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter data to get only True Positives and False Positives with valid probability scores.

    Args:
        df: DataFrame with error analysis results

    Returns:
        Tuple of (true_positives_df, false_positives_df)
    """
    # Filter for TP and FP cases with valid probability scores
    tp_data = df[
        (df['error_value'] == 'true_positive_value') &
        (df['value_probability'].notna())
    ].copy()

    fp_data = df[
        (df['error_value'] == 'false_positive_value') &
        (df['value_probability'].notna())
    ].copy()

    print(f"True Positives with probability scores: {len(tp_data)}")
    print(f"False Positives with probability scores: {len(fp_data)}")

    return tp_data, fp_data


def calculate_basic_stats(tp_data: pd.DataFrame, fp_data: pd.DataFrame) -> Dict:
    """
    Calculate basic statistics for True Positives and False Positives.

    Args:
        tp_data: True positives data
        fp_data: False positives data

    Returns:
        Dictionary with basic statistics
    """
    tp_probs = tp_data['value_probability'].values
    fp_probs = fp_data['value_probability'].values

    return {
        'tp_min': np.min(tp_probs),
        'tp_max': np.max(tp_probs),
        'tp_mean': np.mean(tp_probs),
        'fp_min': np.min(fp_probs),
        'fp_max': np.max(fp_probs),
        'fp_mean': np.mean(fp_probs)
    }


def calculate_percentiles(tp_data: pd.DataFrame, fp_data: pd.DataFrame) -> Dict:
    """
    Calculate percentiles for True Positives and False Positives.

    Args:
        tp_data: True positives data
        fp_data: False positives data

    Returns:
        Dictionary with percentile analysis results
    """
    tp_probs = tp_data['value_probability'].values
    fp_probs = fp_data['value_probability'].values

    # Define percentiles to calculate
    percentiles = [25, 50, 75, 90, 95, 99]  # 25th, median, 75th, 90th, 95th, 99th
    percentile_values = [p/100.0 for p in percentiles]

    results = {
        'tp_count': len(tp_probs),
        'fp_count': len(fp_probs),
        'percentiles': {}
    }

    # Calculate percentiles for both TP and FP
    for i, p in enumerate(percentiles):
        tp_percentile = np.percentile(tp_probs, p)
        fp_percentile = np.percentile(fp_probs, p)

        results['percentiles'][p] = {
            'tp': tp_percentile,
            'fp': fp_percentile
        }

    return results


def format_threshold(t: float) -> str:
    """Format threshold value for display."""
    if t == 0:
        return "Baseline"
    t_str = str(t)
    if '.' in t_str:
        decimal_places = len(t_str.split('.')[1])
        decimal_places = max(1, min(6, decimal_places))
        return f"{t:.{decimal_places}f}"
    else:
        return f"{t:.1f}"


def print_percentiles_summary(percentiles_results: Dict, threshold_results: Dict, basic_stats: Dict):
    """
    Print a summary of percentiles and threshold analysis.

    Args:
        percentiles_results: Results from percentile calculation
        threshold_results: Results from threshold analysis
        basic_stats: Basic statistics (min, max, mean)
    """
    print("\n" + "="*70)
    print("PERCENTILES ANALYSIS RESULTS")
    print("="*70)

    # Sample sizes
    print(f"\nSample Sizes (value_probability scores):")
    print(f"  True Positives:  {percentiles_results['tp_count']:,}")
    print(f"  False Positives: {percentiles_results['fp_count']:,}")

    # Basic statistics
    print(f"\nBasic Statistics (value_probability confidence scores):")
    print(f"  {'Statistic':<12} {'True Positives':<15} {'False Positives':<15}")
    print(f"  {'-'*12} {'-'*15} {'-'*15}")
    print(f"  {'Minimum':<12} {basic_stats['tp_min']:<15.6f} {basic_stats['fp_min']:<15.6f}")
    print(f"  {'Maximum':<12} {basic_stats['tp_max']:<15.6f} {basic_stats['fp_max']:<15.6f}")
    print(f"  {'Mean':<12} {basic_stats['tp_mean']:<15.6f} {basic_stats['fp_mean']:<15.6f}")

    # Percentiles table
    print(f"\nPercentiles Analysis (value_probability confidence scores):")
    print(f"  {'Percentile':<12} {'True Positives':<15} {'False Positives':<15}")
    print(f"  {'-'*12} {'-'*15} {'-'*15}")

    for percentile in [25, 50, 75, 90, 95, 99]:
        data = percentiles_results['percentiles'][percentile]
        print(f"  {percentile}th{'':<8} {data['tp']:<15.6f} {data['fp']:<15.6f}")

    # Enhanced threshold analysis with precision/recall and conversion insights
    print(f"\nProbability Threshold Analysis:")
    print(f"{'Threshold':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'TP':<6} {'FP':<6} {'FN':<6} {'TN':<6} {'TP→FN':<8} {'FP→FN':<8}")
    print("-" * 88)

    # Smart formatting function
    def format_threshold(t):
        if t == 0:
            return "Baseline"
        # Convert to string to count decimal places
        t_str = str(t)
        if '.' in t_str:
            decimal_places = len(t_str.split('.')[1])
            # Use at least 1 decimal place, maximum 6
            decimal_places = max(1, min(6, decimal_places))
            return f"{t:.{decimal_places}f}"
        else:
            return f"{t:.1f}"

    # Print baseline first
    baseline_data = threshold_results[0]
    f1_baseline = 2 * (baseline_data['precision'] * baseline_data['recall']) / (baseline_data['precision'] + baseline_data['recall']) if (baseline_data['precision'] + baseline_data['recall']) > 0 else 0.0
    print(f"{'Baseline':<12} {baseline_data['precision']:<10.4f} {baseline_data['recall']:<10.4f} {f1_baseline:<10.4f} {baseline_data['tp_count']:<6} {baseline_data['fp_count']:<6} {baseline_data['fn_count']:<6} {baseline_data['tn_count']:<6} {baseline_data['tp_to_fn']:<8} {baseline_data['fp_to_fn']:<8}")

    # Print all other thresholds
    thresholds_to_show = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999, 0.999999]

    for threshold in thresholds_to_show:
        if threshold in threshold_results:
            data = threshold_results[threshold]
            threshold_str = format_threshold(threshold)
            # Calculate F1 score
            f1 = 2 * (data['precision'] * data['recall']) / (data['precision'] + data['recall']) if (data['precision'] + data['recall']) > 0 else 0.0

            print(f"{threshold_str:<12} {data['precision']:<10.4f} {data['recall']:<10.4f} {f1:<10.4f} {data['tp_count']:<6} {data['fp_count']:<6} {data['fn_count']:<6} {data['tn_count']:<6} {data['tp_to_fn']:<8} {data['fp_to_fn']:<8}")


def analyze_run(run_id: str, base_path: str = "../../output") -> Dict:
    """
    Complete percentiles analysis for a single run.

    Args:
        run_id: MLflow run ID
        base_path: Base path to output directory

    Returns:
        Dictionary with analysis results
    """
    print(f"Starting percentiles analysis for run: {run_id}")
    print("=" * 70)

    # Step 1: Load data
    print("Step 1: Loading data...")
    try:
        df = load_run_data(run_id, base_path)
        print(f"✓ Successfully loaded {len(df):,} rows from error analysis file")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return {}

    # Step 2: Apply year filtering if enabled
    if RECENT_YEARS_ONLY > 0:
        print(f"\nStep 2: Applying recent years filter ({RECENT_YEARS_ONLY} years)...")
        df = filter_recent_years_only(df, years_back=RECENT_YEARS_ONLY)
        if df.empty:
            print("✗ No data remaining after year filtering")
            return {}
        print(f"✓ Year filtering completed")
    else:
        print(f"\nStep 2: Year filtering disabled (RECENT_YEARS_ONLY = {RECENT_YEARS_ONLY})")

    # Step 3: Filter by scopes if configured
    if SCOPES_FILTER:
        if 'scope_man' in df.columns:
            print(f"\nStep 3: Applying scope filter...")
            before = len(df)
            df = df[df['scope_man'].isin(SCOPES_FILTER)].copy()
            after = len(df)
            print(f"Applied scope filter {SCOPES_FILTER}: kept {after:,}/{before:,} rows")
        else:
            print(f"\nStep 3: Scope filtering skipped (no scope_man column)")
    else:
        print(f"\nStep 3: Scope filtering disabled")

    # Step 4: Show data overview
    print(f"\nStep 4: Data Overview")
    print(f"Total rows in dataset: {len(df):,}")
    print(f"Error value distribution:")
    error_counts = df['error_value'].value_counts()
    for error_type, count in error_counts.items():
        print(f"  {error_type}: {count:,} ({count/len(df)*100:.1f}%)")

    print(f"\nValue probability column statistics:")
    prob_stats = df['value_probability'].describe()
    for stat, value in prob_stats.items():
        if pd.notna(value):
            print(f"  {stat}: {value:.6f}")

    non_null_probs = df['value_probability'].notna().sum()
    print(f"  Non-null probability scores: {non_null_probs:,} ({non_null_probs/len(df)*100:.1f}%)")

    # Step 5: Filter for TP and FP
    print(f"\nStep 5: Filtering for True Positives and False Positives...")
    tp_data, fp_data = filter_tp_fp_data(df)

    if len(tp_data) == 0:
        print("✗ No True Positives with probability scores found")
        return {}
    if len(fp_data) == 0:
        print("✗ No False Positives with probability scores found")
        return {}

    print(f"✓ Found {len(tp_data):,} True Positives and {len(fp_data):,} False Positives with probability scores")

    # Step 6: Calculate statistics
    basic_stats = calculate_basic_stats(tp_data, fp_data)
    percentiles_results = calculate_percentiles(tp_data, fp_data)
    threshold_results = calculate_threshold_metrics(df)
    print_percentiles_summary(percentiles_results, threshold_results, basic_stats)

    return {
        'basic_stats': basic_stats,
        'percentiles': percentiles_results,
        'thresholds': threshold_results
    }


def get_available_runs(base_path: str = "../../output") -> List[str]:
    """Get list of available run IDs with probability data."""
    output_dir = Path(base_path)
    available_runs = []

    for run_dir in output_dir.iterdir():
        if run_dir.is_dir() and run_dir.name != "comparison":
            analysis_file = run_dir / "error_analysis_per_row.csv"
            if analysis_file.exists():
                try:
                    sample_df = pd.read_csv(analysis_file, nrows=1)
                    if 'value_probability' in sample_df.columns:
                        available_runs.append(run_dir.name)
                except Exception:
                    continue

    return available_runs


def list_available_runs_with_probability(base_path: str = "../../output"):
    """
    List all runs that have probability data available.

    Args:
        base_path: Base path to output directory
    """
    print("AVAILABLE RUNS WITH PROBABILITY DATA")
    print("=" * 50)

    output_dir = Path(base_path)
    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        return

    runs_with_prob = []
    runs_without_prob = []

    for run_dir in output_dir.iterdir():
        if run_dir.is_dir() and run_dir.name != "comparison":
            analysis_file = run_dir / "error_analysis_per_row.csv"
            if analysis_file.exists():
                try:
                    # Check if value_probability column exists
                    sample_df = pd.read_csv(analysis_file, nrows=1)
                    if 'value_probability' in sample_df.columns:
                        runs_with_prob.append(run_dir.name)
                    else:
                        runs_without_prob.append(run_dir.name)
                except Exception:
                    runs_without_prob.append(run_dir.name)

    print(f"Runs with probability data ({len(runs_with_prob)}):")
    if runs_with_prob:
        for i, run_id in enumerate(sorted(runs_with_prob), 1):
            print(f"  {i:3d}. {run_id}")
    else:
        print("  None found")

    if runs_with_prob:
        print(f"\nTo analyze a run, use:")
        print(f"  python percentile_analysis.py {runs_with_prob[0]}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze confidence score thresholds for extraction performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python percentile_analysis.py <run_id>
  python percentile_analysis.py --list-runs
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('run_id', nargs='?', help='MLflow run ID to analyze')
    group.add_argument('--run-id', dest='run_id_flag', help='MLflow run ID to analyze')
    group.add_argument('--list-runs', action='store_true', help='List available runs')

    parser.add_argument('--base-path', default='../../output', help='Base path to output directory')

    return parser.parse_args()


def main():
    """
    Main function to run the percentiles analysis.
    """
    args = parse_arguments()

    # Handle list runs option
    if args.list_runs:
        list_available_runs_with_probability(args.base_path)
        return

    # Get run ID from arguments
    run_id = args.run_id or args.run_id_flag

    print("PERCENTILES ANALYSIS")
    print("=" * 70)
    print("Analyzing percentile distribution of confidence scores for")
    print("True Positives vs False Positives")
    print()
    print(f"Run ID: {run_id}")
    print(f"Base path: {args.base_path}")
    print()

    try:
        results = analyze_run(run_id, args.base_path)

        if not results:
            print("\n" + "=" * 70)
            print("ANALYSIS FAILED")
            print("=" * 70)
            print("The analysis could not be completed due to insufficient data.")
            print("This run may not have both True Positives and False Positives")
            print("with probability scores, or the data file may be corrupted.")

    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR DURING ANALYSIS")
        print("=" * 70)
        print(f"Error: {e}")
        print("\nPossible causes:")
        print("1. Run ID does not exist")
        print("2. Run does not have 'value_probability' column")
        print("3. Error analysis file is missing or corrupted")
        print("4. Insufficient permissions to read the file")
        print(f"\nUse '--list-runs' to see available runs with probability data.")


if __name__ == "__main__":
    main()