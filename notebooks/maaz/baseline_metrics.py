"""
Baseline Metrics Analysis for Information Extraction

This script calculates basic metrics (True Positives, False Positives, True Negatives, False Negatives)
from error analysis data without applying probability thresholds.

Usage:
    python baseline_metrics.py <run_id>

Example:
    python baseline_metrics.py 5de297ff670844d68486f9b5eed8bf92

The script will:
1. Load the error analysis data for the specified run
2. Apply optional scope and year filtering
3. Calculate baseline metrics without thresholding
4. Display basic statistics and metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys
import argparse

# Optional scope filter (set to values like ['1', '3', '2lb', '2mb'] or leave empty for all)
SCOPES_FILTER: List[str] = ['1']

# Recent years filtering configuration
# 0 = disabled (use all years), 1 = most recent year only, 3 = recent 3 years, etc.
RECENT_YEARS_ONLY: int = 3


def load_run_data(run_id: str, base_path: str = "../../output") -> pd.DataFrame:
    """
    Load the error analysis data for a specific run.

    Args:
        run_id: MLflow run ID
        base_path: Base path to output directory

    Returns:
        DataFrame with the error analysis results
    """
    file_path = Path(base_path) / run_id / "error_analysis_per_row.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")

    df = pd.read_csv(file_path)
    return df


def extract_publication_year(report_name: str) -> int:
    """
    Extract publication year from report filename.

    Args:
        report_name: Report filename (e.g., "company_2022_report.pdf")

    Returns:
        Publication year as integer
    """
    import re

    # Try pattern like "company_2022_report.pdf"
    year_match = re.search(r'_(\d{4})_', report_name)
    if year_match:
        return int(year_match.group(1))

    # Fallback: try to find any 4-digit number that looks like a year (2000-2099)
    year_match = re.search(r'(20\d{2})', report_name)
    if year_match:
        return int(year_match.group(1))

    # Default fallback if no year found
    return 2023


def filter_recent_years_only(df: pd.DataFrame, years_back: int = 3) -> pd.DataFrame:
    """
    Filter dataframe to keep only entries within recent years of report publication.

    Args:
        df: DataFrame with error analysis results
        years_back: Number of years back from publication to keep

    Returns:
        Filtered DataFrame containing only recent years data
    """
    if 'ReportName' not in df.columns or 'year_man' not in df.columns:
        return df

    filtered_rows = []
    total_before = len(df)

    for report_name in df['ReportName'].unique():
        pub_year = extract_publication_year(report_name)
        report_data = df[df['ReportName'] == report_name]

        # Define target years: from (pub_year - years_back + 1) to pub_year
        target_years = list(range(pub_year - years_back + 1, pub_year + 1))

        # Keep only entries for target years
        recent_data = report_data[report_data['year_man'].isin(target_years)]

        if len(recent_data) > 0:
            filtered_rows.append(recent_data)

    if filtered_rows:
        filtered_df = pd.concat(filtered_rows, ignore_index=True)
        return filtered_df
    else:
        return pd.DataFrame()


def calculate_baseline_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate baseline metrics without applying probability thresholds.

    Args:
        df: DataFrame with error analysis results

    Returns:
        Dictionary with baseline metrics
    """
    # Count error classifications
    error_counts = df['error_value'].value_counts()

    tp_count = error_counts.get('true_positive_value', 0)
    fp_count = error_counts.get('false_positive_value', 0)
    tn_count = error_counts.get('true_negative_value', 0)
    fn_count = error_counts.get('false_negative_value', 0)

    total = tp_count + fp_count + tn_count + fn_count

    # Calculate metrics
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp_count + tn_count) / total if total > 0 else 0.0

    return {
        'tp_count': tp_count,
        'fp_count': fp_count,
        'tn_count': tn_count,
        'fn_count': fn_count,
        'total': total,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'error_counts': error_counts.to_dict()
    }


def print_baseline_summary(metrics: Dict):
    """
    Print baseline metrics summary.

    Args:
        metrics: Dictionary with baseline metrics
    """
    print("\n" + "="*60)
    print("BASELINE METRICS ANALYSIS")
    print("="*60)

    print(f"\nSample Size: {metrics['total']:,}")

    print(f"\nError Classification Counts:")
    print(f"  True Positives:  {metrics['tp_count']:,}")
    print(f"  False Positives: {metrics['fp_count']:,}")
    print(f"  True Negatives:  {metrics['tn_count']:,}")
    print(f"  False Negatives: {metrics['fn_count']:,}")

    print(f"\nPerformance Metrics:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")


def analyze_run(run_id: str, base_path: str = "../../output") -> Dict:
    """
    Complete baseline analysis for a single run.

    Args:
        run_id: MLflow run ID
        base_path: Base path to output directory

    Returns:
        Dictionary with analysis results
    """
    print(f"Analyzing run: {run_id}")

    # Load data
    try:
        df = load_run_data(run_id, base_path)
        print(f"Loaded {len(df):,} rows from error analysis file")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return {}

    # Apply year filtering if enabled
    if RECENT_YEARS_ONLY > 0:
        df = filter_recent_years_only(df, years_back=RECENT_YEARS_ONLY)
        if df.empty:
            print("No data remaining after year filtering")
            return {}
        print(f"Applied recent years filter ({RECENT_YEARS_ONLY} years): {len(df):,} rows remaining")

    # Apply scope filtering if configured
    if SCOPES_FILTER:
        if 'scope_man' in df.columns:
            before = len(df)
            df = df[df['scope_man'].isin(SCOPES_FILTER)].copy()
            after = len(df)
            print(f"Applied scope filter {SCOPES_FILTER}: {after:,}/{before:,} rows kept")

    # Calculate baseline metrics
    metrics = calculate_baseline_metrics(df)
    print_baseline_summary(metrics)

    return metrics


def get_available_runs(base_path: str = "../../output") -> list:
    """
    Get list of available run IDs.

    Args:
        base_path: Base path to output directory

    Returns:
        List of run IDs
    """
    output_dir = Path(base_path)
    available_runs = []

    for run_dir in output_dir.iterdir():
        if run_dir.is_dir() and run_dir.name != "comparison":
            analysis_file = run_dir / "error_analysis_per_row.csv"
            if analysis_file.exists():
                available_runs.append(run_dir.name)

    return available_runs


def list_available_runs(base_path: str = "../../output"):
    """
    List all available runs.

    Args:
        base_path: Base path to output directory
    """
    print("AVAILABLE RUNS")
    print("=" * 50)

    runs = get_available_runs(base_path)

    if runs:
        for i, run_id in enumerate(sorted(runs), 1):
            print(f"  {i:3d}. {run_id}")

        print(f"\nTo analyze a run, use:")
        print(f"  python baseline_metrics.py {runs[0]}")
    else:
        print("  No runs found")


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Analyze baseline metrics without probability thresholds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python baseline_metrics.py 5de297ff670844d68486f9b5eed8bf92
  python baseline_metrics.py --run-id 5de297ff670844d68486f9b5eed8bf92
  python baseline_metrics.py --list-runs

This script calculates basic classification metrics from error analysis data
without applying confidence score thresholds.
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        'run_id',
        nargs='?',
        help='MLflow run ID to analyze'
    )
    group.add_argument(
        '--run-id',
        dest='run_id_flag',
        help='MLflow run ID to analyze (alternative syntax)'
    )
    group.add_argument(
        '--list-runs',
        action='store_true',
        help='List all available runs'
    )

    parser.add_argument(
        '--base-path',
        default='../../output',
        help='Base path to output directory (default: ../output)'
    )

    return parser.parse_args()


def main():
    """
    Main function to run the baseline analysis.
    """
    args = parse_arguments()

    if args.list_runs:
        list_available_runs(args.base_path)
        return

    run_id = args.run_id or args.run_id_flag

    print("BASELINE METRICS ANALYSIS")
    print("=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Base path: {args.base_path}")

    analyze_run(run_id, args.base_path)


if __name__ == "__main__":
    main()