"""
Kalyan Matka Historical Exploratory Data Analysis (EDA) Module.
Performs statistical frequency counts, calculates top/least frequent Jodi numbers,
and generates digit distributions for Open and Close positions (0-9).
"""

import sys
import pandas as pd
from tabulate import tabulate

# Ensure stdout handles UTF-8 safely on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def make_bar(percent: float, max_len: int = 15) -> str:
    """Helper to generate a clean ASCII bar chart using standard ASCII characters."""
    filled = int(round((percent / 100.0) * max_len * 4)) # scale factor for visibility
    filled = min(filled, max_len)
    return "#" * filled + "-" * (max_len - filled)


def analyze_frequencies(df: pd.DataFrame) -> dict:
    """
    Run complete frequency analysis suite on sanitized Kalyan dataset.
    Excludes non-numeric and placeholder draws.
    """
    valid_df = df[df["Is_Valid"] == True].copy()
    total_valid = len(valid_df)

    if total_valid == 0:
        raise ValueError("No valid records available for EDA analysis.")

    # 1. Jodi Frequency
    jodi_counts = valid_df["Jodi"].astype(str).str.zfill(2).value_counts()
    
    # Top 10
    top_10 = jodi_counts.head(10).reset_index()
    top_10.columns = ["Jodi", "Frequency"]
    top_10["Rank"] = range(1, len(top_10) + 1)
    top_10["Percentage"] = (top_10["Frequency"] / total_valid * 100).round(2)
    top_10["Distribution"] = top_10["Percentage"].apply(lambda p: make_bar(p, 12))
    top_10_table = top_10[["Rank", "Jodi", "Frequency", "Percentage", "Distribution"]]

    # Bottom 10
    bottom_10 = jodi_counts.tail(10).reset_index()
    bottom_10.columns = ["Jodi", "Frequency"]
    bottom_10["Rank"] = range(len(jodi_counts) - len(bottom_10) + 1, len(jodi_counts) + 1)
    bottom_10["Percentage"] = (bottom_10["Frequency"] / total_valid * 100).round(2)
    bottom_10["Distribution"] = bottom_10["Percentage"].apply(lambda p: make_bar(p, 12))
    bottom_10_table = bottom_10[["Rank", "Jodi", "Frequency", "Percentage", "Distribution"]]

    # Check for any undrawn Jodis (00-99)
    all_possible_jodis = {f"{i:02d}" for i in range(100)}
    drawn_jodis = set(jodi_counts.index)
    undrawn_jodis = sorted(list(all_possible_jodis - drawn_jodis))

    # 2. Open Digit Frequency (0-9)
    open_counts = valid_df["Open_Digit"].astype(int).value_counts().reindex(range(10), fill_value=0).reset_index()
    open_counts.columns = ["Digit", "Frequency"]
    open_counts["Percentage"] = (open_counts["Frequency"] / total_valid * 100).round(2)
    open_counts["Distribution"] = open_counts["Percentage"].apply(lambda p: make_bar(p, 12))

    # 3. Close Digit Frequency (0-9)
    close_counts = valid_df["Close_Digit"].astype(int).value_counts().reindex(range(10), fill_value=0).reset_index()
    close_counts.columns = ["Digit", "Frequency"]
    close_counts["Percentage"] = (close_counts["Frequency"] / total_valid * 100).round(2)
    close_counts["Distribution"] = close_counts["Percentage"].apply(lambda p: make_bar(p, 12))

    # 4. Overall Single Digit Frequency (Open + Close combined)
    all_digits = pd.concat([valid_df["Open_Digit"].astype(int), valid_df["Close_Digit"].astype(int)])
    total_digits = len(all_digits)
    overall_counts = all_digits.value_counts().reindex(range(10), fill_value=0).reset_index()
    overall_counts.columns = ["Digit", "Frequency"]
    overall_counts["Percentage"] = (overall_counts["Frequency"] / total_digits * 100).round(2)
    overall_counts["Distribution"] = overall_counts["Percentage"].apply(lambda p: make_bar(p, 12))

    return {
        "total_draws": len(df),
        "valid_draws": total_valid,
        "invalid_draws": len(df) - total_valid,
        "date_range": (df["Date"].min(), df["Date"].max()),
        "top_10": top_10_table,
        "bottom_10": bottom_10_table,
        "undrawn_jodis": undrawn_jodis,
        "open_counts": open_counts,
        "close_counts": close_counts,
        "overall_counts": overall_counts,
    }


def print_eda_report(results: dict):
    """Print clean ASCII formatted CLI report."""
    print("\n" + "=" * 75)
    print("        KALYAN HISTORICAL RECORD EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 75)
    print(f"Time Horizon:            {results['date_range'][0]} to {results['date_range'][1]}")
    print(f"Total Calendar Draws:    {results['total_draws']}")
    print(f"Valid Jodi Draws:        {results['valid_draws']} ({results['valid_draws']/results['total_draws']*100:.1f}%)")
    print(f"Holidays / Excluded:     {results['invalid_draws']} ({results['invalid_draws']/results['total_draws']*100:.1f}%)")
    print(f"Undrawn Numbers (00-99): {results['undrawn_jodis'] if results['undrawn_jodis'] else 'None (All 100 drawn at least once)'}")
    print("-" * 75)

    print("\n[+] TOP 10 MOST FREQUENT JODI NUMBERS:")
    print(tabulate(results["top_10"], headers="keys", tablefmt="github", showindex=False))

    print("\n[-] LEAST FREQUENT JODI NUMBERS (BOTTOM 10):")
    print(tabulate(results["bottom_10"], headers="keys", tablefmt="github", showindex=False))

    print("\n[*] SINGLE OPEN DIGIT FREQUENCY DISTRIBUTION (0-9):")
    print(tabulate(results["open_counts"], headers="keys", tablefmt="github", showindex=False))

    print("\n[*] SINGLE CLOSE DIGIT FREQUENCY DISTRIBUTION (0-9):")
    print(tabulate(results["close_counts"], headers="keys", tablefmt="github", showindex=False))

    print("\n[*] OVERALL SINGLE DIGIT COMBINED FREQUENCY (0-9):")
    print(tabulate(results["overall_counts"], headers="keys", tablefmt="github", showindex=False))
    print("=" * 75 + "\n")


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "kalyan_historical_data.csv"
    try:
        df = pd.read_csv(csv_file, dtype={"Jodi": str, "Raw_Entry": str})
        results = analyze_frequencies(df)
        print_eda_report(results)
    except FileNotFoundError:
        print(f"[ERROR] File '{csv_file}' not found. Run scraper first.")
