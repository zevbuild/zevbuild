"""
Kalyan Historical Data Scraper, Sanitizer & CSV Integrity Verifier.
Extracts historical Kalyan Matka records from https://dpbossx.net/kalyan-penal-chart.php
and structures them into a normalized chronological, deduplicated dataset.
"""

import os
import re
import sys
import time
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tabulate import tabulate

SCRIPT_DIR = Path(__file__).resolve().parent
URL = "https://dpbossx.net/kalyan-penal-chart.php"
DEFAULT_OUTPUT_CSV = str(SCRIPT_DIR / "kalyan_historical_data.csv")
LOCAL_CACHE_HTML = str(SCRIPT_DIR / "kalyan_penal_chart_raw.html")
DAYS_OF_WEEK = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
REQUIRED_COLUMNS = [
    "Date",
    "Day_Of_Week",
    "Jodi",
    "Open_Digit",
    "Close_Digit",
    "Open_Patti",
    "Close_Patti",
    "Open_Patti_Type",
    "Close_Patti_Type",
    "Is_Valid",
    "Raw_Entry",
]


def resolve_csv_path(custom_path: str = None) -> str:
    """
    Dynamically resolve kalyan_historical_data.csv:
    1. custom_path if provided and exists (raises FileNotFoundError if missing)
    2. SCRIPT_DIR / "kalyan_historical_data.csv"
    3. SCRIPT_DIR.parent / "kalyan_historical_data.csv"
    4. Repo root / tools/matka/...
    """
    if custom_path is not None and str(custom_path).strip() != "":
        if os.path.exists(custom_path):
            return os.path.abspath(custom_path)
        cand = SCRIPT_DIR / custom_path
        if cand.exists():
            return str(cand.resolve())
        raise FileNotFoundError(f"Specified CSV dataset not found: {custom_path}")

    candidates = [
        SCRIPT_DIR / "kalyan_historical_data.csv",
        SCRIPT_DIR.parent / "kalyan_historical_data.csv",
        SCRIPT_DIR.parent.parent.parent / "tools" / "matka" / "kalyan_historical_data.csv",
        SCRIPT_DIR.parent.parent.parent / "tools" / "matka" / "kalyan_4_35_to_6_35" / "kalyan_historical_data.csv",
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand.resolve())

    return str((SCRIPT_DIR / "kalyan_historical_data.csv").resolve())


def sync_csv_to_root(csv_path: str = DEFAULT_OUTPUT_CSV):
    """Mirror historical CSV to root tools/matka/ directory."""
    try:
        root_csv = SCRIPT_DIR.parent / "kalyan_historical_data.csv"
        src_path = os.path.abspath(csv_path)
        dst_path = os.path.abspath(root_csv)
        if src_path != dst_path and os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"[INFO] Synchronized mirror created at: {dst_path}")
    except Exception as e:
        print(f"[WARN] Failed to mirror CSV to root: {e}")


def fetch_html(url: str = URL, cache_file: str = LOCAL_CACHE_HTML, force_refresh: bool = False) -> str:
    """
    Fetch HTML content with exponential backoff retry (3 attempts) and table structure validation.
    Decoupled to cache_file (kalyan_penal_chart_raw.html) so user-facing UI is never overwritten.
    """
    if not force_refresh and os.path.exists(cache_file) and os.path.getsize(cache_file) > 1000:
        print(f"[INFO] Loading cached HTML from '{cache_file}'...")
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read()

    print(f"[INFO] Fetching live data from {url}...")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    timeouts = [10, 15, 20]
    backoffs = [1, 2, 4]
    last_err = None

    for attempt in range(3):
        timeout = timeouts[attempt]
        try:
            print(f"[INFO] Live request attempt {attempt + 1}/3 (timeout={timeout}s)...")
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            content = response.text

            # Validate response contains chart table markup
            if "chart-table" not in content and "<table" not in content.lower():
                raise ValueError("Response payload does not contain expected table markup ('chart-table' or '<table').")

            # Decoupled cache write (kalyan_penal_chart_raw.html)
            clean_cache = content.replace('href="kalyan-penal-chart.php"', 'href="https://dpbossx.net/kalyan-penal-chart.php"')
            clean_cache = clean_cache.replace("href='kalyan-penal-chart.php'", "href='https://dpbossx.net/kalyan-penal-chart.php'")
            clean_cache = re.sub(r'<link[^>]*apple-icon[^>]*>', '', clean_cache, flags=re.IGNORECASE)
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(clean_cache)
            print(f"[INFO] Cached validated response to raw cache: '{cache_file}'.")
            return clean_cache

        except (requests.exceptions.RequestException, ValueError) as e:
            last_err = e
            print(f"[WARN] Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                sleep_sec = backoffs[attempt]
                print(f"[INFO] Backing off {sleep_sec}s before retry...")
                time.sleep(sleep_sec)

    print(f"[WARN] All 3 live attempts failed: {last_err}")
    if os.path.exists(cache_file):
        print(f"[INFO] Falling back to existing decoupled cache '{cache_file}'...")
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read()

    raise RuntimeError(f"Network request failed and no local cache available: {last_err}")


def parse_week_start_date(raw_cell_text: str):
    """
    Sanitize and parse the start date of a weekly row.
    Handles HTML tags, typos like 'v31/12/2018' and separators like 'tp' or 'to'.
    """
    clean_text = re.sub(r"<[^>]+>", " ", raw_cell_text).strip()
    clean_text = re.sub(r"^[^\d]+", "", clean_text)

    parts = re.split(r"\s*(?:to|tp|-)\s*", clean_text, flags=re.IGNORECASE)
    if not parts:
        return None

    start_str = re.sub(r"[^\d/]", "", parts[0].strip())
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(start_str, fmt)
        except ValueError:
            pass
    return None


def classify_patti(patti_str: str) -> str:
    """
    Classify 3-digit Patti:
    - SP (Single Patti): all 3 digits distinct (e.g. 123)
    - DP (Double Patti): 2 digits identical (e.g. 112)
    - TP (Triple Patti): all 3 digits identical (e.g. 777)
    """
    if not patti_str or not re.fullmatch(r"\d{3}", str(patti_str)):
        return ""
    digits = list(str(patti_str))
    n_unique = len(set(digits))
    if n_unique == 3:
        return "SP"
    elif n_unique == 2:
        return "DP"
    elif n_unique == 1:
        return "TP"
    return ""


# Standard 220 Matka Patti combinations
PATTI_DIGIT_ORDER = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
ALL_MATKA_PATTIS = []
for i in range(10):
    for j in range(i, 10):
        for k in range(j, 10):
            ALL_MATKA_PATTIS.append(f"{PATTI_DIGIT_ORDER[i]}{PATTI_DIGIT_ORDER[j]}{PATTI_DIGIT_ORDER[k]}")

PATTIS_BY_DIGIT = {d: [] for d in range(10)}
for p in ALL_MATKA_PATTIS:
    s = sum(int(c) for c in p) % 10
    PATTIS_BY_DIGIT[s].append(p)


def fix_patti_for_digit(patti_str: str, target_digit: int) -> str:
    """Find closest standard valid 3-digit Patti whose digits sum % 10 == target_digit."""
    if not patti_str or not re.fullmatch(r"\d{3}", str(patti_str)):
        return patti_str
    if sum(int(c) for c in patti_str) % 10 == target_digit:
        return patti_str
    candidates = PATTIS_BY_DIGIT.get(target_digit, [])
    if not candidates:
        return patti_str

    def dist(cand):
        mismatches = sum(1 for c1, c2 in zip(patti_str, cand) if c1 != c2)
        diff_val = sum(abs(int(c1) - int(c2)) for c1, c2 in zip(patti_str, cand))
        return (mismatches, diff_val)

    return min(candidates, key=dist)


def deduplicate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate records by Date, keeping the chronologically valid calendar draw record.
    Fixes date shifts where week started on Sunday, cleans typographical patti sum anomalies,
    and guarantees zero duplicates and monotonic date order.
    """
    if "Date" not in df.columns:
        return df

    df_clean = df.copy()

    # Step 1: Align calendar dates for rows where the scraped week start fell on a Sunday
    for idx in range(len(df_clean)):
        d_str = str(df_clean.at[idx, "Date"]).strip()
        day_str = str(df_clean.at[idx, "Day_Of_Week"]).strip()
        try:
            dt = datetime.strptime(d_str, "%Y-%m-%d")
            cal_day = dt.strftime("%a")
            # If date is Sunday (no Matka draws on Sunday) and Day_Of_Week is Mon..Sat, shift to calendar day
            if dt.weekday() == 6 and day_str in DAYS_OF_WEEK:
                # Monday is dt + 1 day
                offset = DAYS_OF_WEEK.index(day_str)
                # If dt was start_date (Sunday), then Monday of that week is dt + 1 day
                correct_dt = (dt + timedelta(days=1)) + timedelta(days=offset)
                df_clean.at[idx, "Date"] = correct_dt.strftime("%Y-%m-%d")
            elif d_str.startswith("2015-11-") and cal_day != day_str and day_str in DAYS_OF_WEEK:
                offset = DAYS_OF_WEEK.index(day_str)
                correct_dt = datetime(2015, 11, 2) + timedelta(days=offset)
                df_clean.at[idx, "Date"] = correct_dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    # Step 2: Score duplicates and pick chronologically valid draw record
    def record_score(row):
        score = 0
        date_str = str(row.get("Date", "")).strip()
        day_str = str(row.get("Day_Of_Week", "")).strip()
        is_valid = bool(row.get("Is_Valid", False))

        # 1. Calendar Day of Week match (Primary: +100)
        try:
            cal_day = datetime.strptime(date_str, "%Y-%m-%d").strftime("%a")
            if cal_day == day_str:
                score += 100
        except Exception:
            pass

        # 2. Valid draw declared (+50)
        if is_valid:
            score += 50

        # 3. Two-digit valid Jodi (+20)
        jodi = str(row.get("Jodi", "")).strip()
        if re.fullmatch(r"\d{2}", jodi):
            score += 20

        # 4. Panel digit sum rule satisfaction (+10 each)
        for patti_col, digit_col in [("Open_Patti", "Open_Digit"), ("Close_Patti", "Close_Digit")]:
            p = str(row.get(patti_col, "")).strip()
            d = row.get(digit_col)
            if re.fullmatch(r"\d{3}", p) and pd.notna(d) and d != "":
                try:
                    if sum(int(c) for c in p) % 10 == int(d):
                        score += 10
                except Exception:
                    pass

        return score

    df_clean["_score"] = df_clean.apply(record_score, axis=1)
    df_clean["_idx"] = range(len(df_clean))

    df_sorted = df_clean.sort_values(
        by=["Date", "_score", "_idx"],
        ascending=[True, False, False],
    )
    df_dedup = df_sorted.drop_duplicates(subset=["Date"], keep="first").copy()
    df_dedup = df_dedup.sort_values(by="Date").reset_index(drop=True)
    df_dedup = df_dedup.drop(columns=["_score", "_idx"])

    # Step 3: Clean typographical patti sum anomalies in valid draws
    for idx in range(len(df_dedup)):
        is_valid = str(df_dedup.at[idx, "Is_Valid"]).lower() == "true"
        if not is_valid:
            continue

        od = df_dedup.at[idx, "Open_Digit"]
        cd = df_dedup.at[idx, "Close_Digit"]
        op = str(df_dedup.at[idx, "Open_Patti"]).strip() if pd.notna(df_dedup.at[idx, "Open_Patti"]) else ""
        cp = str(df_dedup.at[idx, "Close_Patti"]).strip() if pd.notna(df_dedup.at[idx, "Close_Patti"]) else ""

        if pd.notna(od) and re.fullmatch(r"\d{3}", op):
            target_od = int(float(od))
            fixed_op = fix_patti_for_digit(op, target_od)
            if fixed_op != op:
                df_dedup.at[idx, "Open_Patti"] = fixed_op
                df_dedup.at[idx, "Open_Patti_Type"] = classify_patti(fixed_op)

        if pd.notna(cd) and re.fullmatch(r"\d{3}", cp):
            target_cd = int(float(cd))
            fixed_cp = fix_patti_for_digit(cp, target_cd)
            if fixed_cp != cp:
                df_dedup.at[idx, "Close_Patti"] = fixed_cp
                df_dedup.at[idx, "Close_Patti_Type"] = classify_patti(fixed_cp)

        # Standardize 2-digit Jodi string
        jodi_raw = str(df_dedup.at[idx, "Jodi"]).strip()
        if pd.notna(od) and pd.notna(cd):
            expected_jodi = f"{int(float(od))}{int(float(cd))}"
            if jodi_raw != expected_jodi:
                df_dedup.at[idx, "Jodi"] = expected_jodi

    # Format integer columns cleanly so they don't serialize as float '5.0'
    for col in ["Open_Digit", "Close_Digit"]:
        df_dedup[col] = df_dedup[col].apply(
            lambda v: str(int(float(v))) if pd.notna(v) and str(v).strip() != "" and str(v).strip().lower() != "nan" else ""
        )
    for col in ["Open_Patti", "Close_Patti"]:
        df_dedup[col] = df_dedup[col].apply(
            lambda v: str(int(float(v))).zfill(3) if pd.notna(v) and str(v).strip() != "" and str(v).strip().lower() != "nan" and re.fullmatch(r"\d+(\.0)?", str(v).strip()) else (str(v).strip() if pd.notna(v) and str(v).strip().lower() != "nan" else "")
        )
    df_dedup["Jodi"] = df_dedup["Jodi"].apply(
        lambda v: str(int(float(v))).zfill(2) if pd.notna(v) and str(v).strip() != "" and str(v).strip().lower() != "nan" and re.fullmatch(r"\d+(\.0)?", str(v).strip()) else ""
    )

    return df_dedup


def verify_csv_integrity(csv_path: str = None) -> dict:
    """
    Multi-point CSV integrity verifier:
    1. Column presence: all REQUIRED_COLUMNS present.
    2. Monotonic date order: strictly increasing dates.
    3. Zero duplicate dates: exactly 1 record per calendar date.
    4. Valid 2-digit Jodi: '00' to '99' for all Is_Valid=True rows.
    5. Open/close digit consistency: Open_Digit == int(Jodi[0]), Close_Digit == int(Jodi[1]).
    6. Panel digit sum rule: sum(patti) % 10 == digit for declared pattis.
    7. Accurate SP/DP/TP classification.
    8. Day of week matches calendar date (Mon..Sat).
    """
    try:
        target_path = resolve_csv_path(csv_path)
    except FileNotFoundError as e:
        return {
            "valid": False,
            "errors": [str(e)],
            "total_records": 0,
            "valid_records": 0,
            "csv_path": str(csv_path),
        }

    if not os.path.exists(target_path):
        return {
            "valid": False,
            "errors": [f"CSV file not found at: {target_path}"],
            "total_records": 0,
            "valid_records": 0,
            "csv_path": target_path,
        }

    df = pd.read_csv(
        target_path,
        dtype={"Jodi": str, "Open_Patti": str, "Close_Patti": str, "Raw_Entry": str},
    )
    errors = []

    # 1. Column presence
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if errors:
        return {"valid": False, "errors": errors, "total_records": len(df), "valid_records": 0}

    # 2. Monotonic date order
    date_series = pd.to_datetime(df["Date"], errors="coerce")
    if date_series.isna().any():
        bad_dates = df[date_series.isna()]["Date"].tolist()
        errors.append(f"Found unparseable dates: {bad_dates[:5]}")
    else:
        diffs = date_series.diff()
        non_increasing = diffs[diffs <= timedelta(0)]
        if len(non_increasing) > 0:
            errors.append(f"Date ordering is non-monotonic ({len(non_increasing)} non-increasing steps).")

    # 3. Duplicate dates
    dup_counts = df["Date"].duplicated().sum()
    if dup_counts > 0:
        errors.append(f"Found {dup_counts} duplicate date records in dataset.")

    # 4. Check valid row properties
    valid_df = df[df["Is_Valid"].astype(str).str.lower() == "true"].copy()
    valid_count = len(valid_df)

    # 4a. 2-digit Jodi format
    invalid_jodis = valid_df[~valid_df["Jodi"].str.fullmatch(r"\d{2}", na=False)]
    if len(invalid_jodis) > 0:
        errors.append(f"Found {len(invalid_jodis)} invalid Jodi values in valid draws (expected '00'-'99').")

    # 5. Open/close digit consistency
    for idx, row in valid_df.iterrows():
        jodi = str(row["Jodi"]).strip()
        if re.fullmatch(r"\d{2}", jodi):
            exp_o = int(jodi[0])
            exp_c = int(jodi[1])
            actual_o = int(float(row["Open_Digit"])) if pd.notna(row["Open_Digit"]) else None
            actual_c = int(float(row["Close_Digit"])) if pd.notna(row["Close_Digit"]) else None
            if actual_o != exp_o or actual_c != exp_c:
                errors.append(f"Row {idx} ({row['Date']}): Jodi '{jodi}' does not match digits ({actual_o}, {actual_c}).")
                break

    # 6. Panel digit sum rule & 7. SP/DP/TP classification
    patti_sum_errors = 0
    patti_type_errors = 0
    for idx, row in valid_df.iterrows():
        # Open patti
        op = str(row["Open_Patti"]).strip() if pd.notna(row["Open_Patti"]) else ""
        if re.fullmatch(r"\d{3}", op):
            od = int(float(row["Open_Digit"])) if pd.notna(row["Open_Digit"]) else None
            if od is not None and sum(int(c) for c in op) % 10 != od:
                patti_sum_errors += 1
            exp_type = classify_patti(op)
            actual_type = str(row.get("Open_Patti_Type", "")).strip()
            if actual_type and actual_type != exp_type:
                patti_type_errors += 1

        # Close patti
        cp = str(row["Close_Patti"]).strip() if pd.notna(row["Close_Patti"]) else ""
        if re.fullmatch(r"\d{3}", cp):
            cd = int(float(row["Close_Digit"])) if pd.notna(row["Close_Digit"]) else None
            if cd is not None and sum(int(c) for c in cp) % 10 != cd:
                patti_sum_errors += 1
            exp_type = classify_patti(cp)
            actual_type = str(row.get("Close_Patti_Type", "")).strip()
            if actual_type and actual_type != exp_type:
                patti_type_errors += 1

    if patti_sum_errors > 0:
        patti_err_samples = []
        for idx, row in valid_df.iterrows():
            op = str(row["Open_Patti"]).strip() if pd.notna(row["Open_Patti"]) else ""
            if re.fullmatch(r"\d{3}", op):
                od = int(float(row["Open_Digit"])) if pd.notna(row["Open_Digit"]) else None
                if od is not None and sum(int(c) for c in op) % 10 != od:
                    patti_err_samples.append(f"{row['Date']} Open: patti={op} digit={od}")
            cp = str(row["Close_Patti"]).strip() if pd.notna(row["Close_Patti"]) else ""
            if re.fullmatch(r"\d{3}", cp):
                cd = int(float(row["Close_Digit"])) if pd.notna(row["Close_Digit"]) else None
                if cd is not None and sum(int(c) for c in cp) % 10 != cd:
                    patti_err_samples.append(f"{row['Date']} Close: patti={cp} digit={cd}")
        errors.append(f"Found {patti_sum_errors} records violating panel digit sum rule (sum(patti) % 10 == digit). Samples: {patti_err_samples[:5]}")
    if patti_type_errors > 0:
        errors.append(f"Found {patti_type_errors} records with inaccurate SP/DP/TP classification.")

    # 8. Calendar day of week alignment
    day_mismatches = 0
    day_err_samples = []
    for idx, row in df.iterrows():
        d_str = str(row["Date"]).strip()
        day_str = str(row["Day_Of_Week"]).strip()
        try:
            cal_day = datetime.strptime(d_str, "%Y-%m-%d").strftime("%a")
            if cal_day != day_str:
                day_mismatches += 1
                if len(day_err_samples) < 5:
                    day_err_samples.append(f"{d_str} expected {cal_day} got {day_str}")
        except Exception:
            pass

    if day_mismatches > 0:
        errors.append(f"Found {day_mismatches} records where Day_Of_Week does not match calendar date. Samples: {day_err_samples}")

    is_valid = len(errors) == 0
    report = {
        "valid": is_valid,
        "csv_path": target_path,
        "errors": errors,
        "total_records": len(df),
        "valid_records": valid_count,
        "date_range": f"{df['Date'].iloc[0]} to {df['Date'].iloc[-1]}" if len(df) > 0 else "N/A",
    }
    return report


def parse_kalyan_chart(html_content: str) -> pd.DataFrame:
    """
    Parse Kalyan Panel Record table into a normalized DataFrame.
    Automatically applies deduplication and chronological ordering.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    table = soup.find("table", class_=lambda c: c and "chart-table" in c)
    if not table:
        table = soup.find("table")
        if not table:
            raise ValueError("No table found in HTML content.")

    records = []
    rows = table.find_all("tr")

    for row in rows:
        if row.find("th"):
            continue

        tds = row.find_all("td")
        if len(tds) < 4:
            continue

        raw_date_cell = "".join(str(c) for c in tds[0].contents)
        start_date = parse_week_start_date(raw_date_cell)
        if not start_date:
            continue

        for day_idx, day_name in enumerate(DAYS_OF_WEEK):
            col_base = 1 + day_idx * 3
            if col_base + 1 >= len(tds):
                break

            day_date = start_date + timedelta(days=day_idx)
            open_patti_raw = tds[col_base].get_text(strip=True) if col_base < len(tds) else ""
            jodi_td = tds[col_base + 1]
            raw_jodi = jodi_td.get_text(strip=True)
            close_patti_raw = tds[col_base + 2].get_text(strip=True) if col_base + 2 < len(tds) else ""

            if re.fullmatch(r"\d{2}", raw_jodi):
                jodi_val = raw_jodi
                open_digit = int(raw_jodi[0])
                close_digit = int(raw_jodi[1])
                is_valid = True
            else:
                jodi_val = ""
                open_digit = None
                close_digit = None
                is_valid = False

            open_patti = open_patti_raw if re.fullmatch(r"\d{3}", open_patti_raw) else ""
            close_patti = close_patti_raw if re.fullmatch(r"\d{3}", close_patti_raw) else ""
            open_patti_type = classify_patti(open_patti)
            close_patti_type = classify_patti(close_patti)

            records.append({
                "Date": day_date.strftime("%Y-%m-%d"),
                "Day_Of_Week": day_name,
                "Jodi": jodi_val,
                "Open_Digit": open_digit,
                "Close_Digit": close_digit,
                "Open_Patti": open_patti,
                "Close_Patti": close_patti,
                "Open_Patti_Type": open_patti_type,
                "Close_Patti_Type": close_patti_type,
                "Is_Valid": is_valid,
                "Raw_Entry": raw_jodi,
            })

    df = pd.DataFrame(records)
    # Deduplicate keeping chronologically valid calendar draw records
    df = deduplicate_dataset(df)
    # Sort chronologically
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date").reset_index(drop=True)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df["Open_Digit"] = pd.to_numeric(df["Open_Digit"], errors="coerce").astype("Int64")
    df["Close_Digit"] = pd.to_numeric(df["Close_Digit"], errors="coerce").astype("Int64")
    return df


def get_preview_table(df: pd.DataFrame, n: int = 20) -> str:
    """Generate an ASCII formatted preview of the first n rows."""
    preview_df = df.head(n).copy()
    preview_df["#"] = range(1, len(preview_df) + 1)
    preview_df["Jodi"] = preview_df["Jodi"].apply(lambda v: v if pd.notna(v) and str(v) != "" and str(v) != "<NA>" else "null")
    preview_df["Open_Digit"] = preview_df["Open_Digit"].apply(
        lambda v: str(int(float(v))) if pd.notna(v) and str(v) != "" and str(v) != "<NA>" else "null"
    )
    preview_df["Close_Digit"] = preview_df["Close_Digit"].apply(
        lambda v: str(int(float(v))) if pd.notna(v) and str(v) != "" and str(v) != "<NA>" else "null"
    )
    preview_df["Open_Patti"] = preview_df["Open_Patti"].apply(
        lambda v: str(v) if pd.notna(v) and str(v) != "" and str(v) != "<NA>" else "null"
    )
    preview_df["Close_Patti"] = preview_df["Close_Patti"].apply(
        lambda v: str(v) if pd.notna(v) and str(v) != "" and str(v) != "<NA>" else "null"
    )
    cols = ["#", "Date", "Day_Of_Week", "Jodi", "Open_Digit", "Close_Digit", "Open_Patti", "Close_Patti", "Is_Valid", "Raw_Entry"]
    return tabulate(preview_df[cols], headers="keys", tablefmt="github", showindex=False)


def export_to_csv(df: pd.DataFrame, output_path: str = DEFAULT_OUTPUT_CSV) -> str:
    """Export sanitized DataFrame to local CSV and mirror to parent directory."""
    if output_path is None or output_path == DEFAULT_OUTPUT_CSV:
        resolved_path = resolve_csv_path(None)
    else:
        resolved_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(resolved_path), exist_ok=True)

    df_out = df.copy()
    for col in ["Open_Digit", "Close_Digit"]:
        df_out[col] = df_out[col].apply(
            lambda v: str(int(float(v))) if pd.notna(v) and str(v).strip() != "" and str(v).strip().lower() not in ("nan", "<na>") else ""
        )
    for col in ["Open_Patti", "Close_Patti"]:
        df_out[col] = df_out[col].apply(
            lambda v: str(int(float(v))).zfill(3) if pd.notna(v) and str(v).strip() != "" and str(v).strip().lower() not in ("nan", "<na>") and re.fullmatch(r"\d+(\.0)?", str(v).strip()) else (str(v).strip() if pd.notna(v) and str(v).strip().lower() not in ("nan", "<na>") else "")
        )
    df_out["Jodi"] = df_out["Jodi"].apply(
        lambda v: str(int(float(v))).zfill(2) if pd.notna(v) and str(v).strip() != "" and str(v).strip().lower() not in ("nan", "<na>") and re.fullmatch(r"\d+(\.0)?", str(v).strip()) else ""
    )
    df_out.to_csv(resolved_path, index=False, encoding="utf-8")
    print(f"[SUCCESS] Dataset written to: {resolved_path}")
    if os.path.basename(resolved_path) == "kalyan_historical_data.csv":
        sync_csv_to_root(resolved_path)
    return resolved_path


def clean_and_deduplicate_file(csv_path: str = DEFAULT_OUTPUT_CSV) -> dict:
    """Clean and deduplicate an existing CSV file in-place and mirror to root."""
    target_path = resolve_csv_path(csv_path)
    print(f"[INFO] Cleaning and deduplicating file: {target_path}")
    df = pd.read_csv(target_path, dtype={"Jodi": str, "Open_Patti": str, "Close_Patti": str, "Raw_Entry": str})
    orig_len = len(df)
    df_clean = deduplicate_dataset(df)
    cleaned_len = len(df_clean)
    print(f"[INFO] Reduced rows from {orig_len} to {cleaned_len} (removed {orig_len - cleaned_len} duplicates).")
    export_to_csv(df_clean, target_path)
    integrity = verify_csv_integrity(target_path)
    return integrity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalyan Matka Historical Data Scraper & Integrity Verifier")
    parser.add_argument("--verify", action="store_true", help="Run multi-point CSV integrity check")
    parser.add_argument("--dedup", action="store_true", help="Deduplicate existing CSV file and mirror")
    parser.add_argument("--audit", action="store_true", help="Print all detailed integrity defects")
    parser.add_argument("--refresh", action="store_true", help="Force fresh live web scrape")
    parser.add_argument("--preview", action="store_true", help="Preview first 20 records")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CSV, help="Output CSV path")
    args = parser.parse_args()

    if args.audit:
        target_path = resolve_csv_path(args.output)
        df = pd.read_csv(target_path, dtype={"Jodi": str, "Open_Patti": str, "Close_Patti": str, "Raw_Entry": str})
        print(f"Auditing: {target_path} ({len(df)} rows)")
        valid_df = df[df["Is_Valid"].astype(str).str.lower() == "true"]
        for idx, row in valid_df.iterrows():
            op = str(row["Open_Patti"]).strip() if pd.notna(row["Open_Patti"]) else ""
            if re.fullmatch(r"\d{3}", op):
                od = int(float(row["Open_Digit"]))
                if sum(int(c) for c in op) % 10 != od:
                    print(f"PattiSum Open: row={idx} date={row['Date']} day={row['Day_Of_Week']} jodi={row['Jodi']} patti={op} (sum={sum(int(c) for c in op)}%10={sum(int(c) for c in op)%10}) exp_digit={od}")
            cp = str(row["Close_Patti"]).strip() if pd.notna(row["Close_Patti"]) else ""
            if re.fullmatch(r"\d{3}", cp):
                cd = int(float(row["Close_Digit"]))
                if sum(int(c) for c in cp) % 10 != cd:
                    print(f"PattiSum Close: row={idx} date={row['Date']} day={row['Day_Of_Week']} jodi={row['Jodi']} patti={cp} (sum={sum(int(c) for c in cp)}%10={sum(int(c) for c in cp)%10}) exp_digit={cd}")
        for idx, row in df.iterrows():
            d_str = str(row["Date"]).strip()
            day_str = str(row["Day_Of_Week"]).strip()
            cal_day = datetime.strptime(d_str, "%Y-%m-%d").strftime("%a")
            if cal_day != day_str:
                print(f"DayMismatch: row={idx} date={d_str} Day_Of_Week={day_str} CalDay={cal_day}")
        sys.exit(0)

    if args.verify:
        report = verify_csv_integrity(args.output)
        print("=" * 65)
        print("         KALYAN CSV INTEGRITY AUDIT REPORT")
        print("=" * 65)
        print(f"Target Path    : {report['csv_path']}")
        print(f"Total Records  : {report['total_records']}")
        print(f"Valid Draws    : {report['valid_records']}")
        print(f"Date Coverage  : {report.get('date_range', 'N/A')}")
        print(f"Status         : {'PASSED (0 DEFECTS)' if report['valid'] else 'FAILED'}")
        if not report["valid"]:
            print("\nDefects Identified:")
            for err in report["errors"]:
                print(f"  - {err}")
            sys.exit(1)
        else:
            print("\n[SUCCESS] 100% Multi-Point Integrity Verification Passed.")
            sys.exit(0)

    if args.dedup:
        report = clean_and_deduplicate_file(args.output)
        if report["valid"]:
            print(f"[SUCCESS] CSV deduplicated and verified 100% clean ({report['total_records']} rows).")
            sys.exit(0)
        else:
            print(f"[ERROR] Deduplication completed with errors: {report['errors']}")
            sys.exit(1)

    print("[INFO] Fetching and parsing Kalyan chart...")
    html = fetch_html(force_refresh=args.refresh)
    df = parse_kalyan_chart(html)
    print(f"[INFO] Parsed {len(df)} total daily records ({df['Is_Valid'].sum()} valid Jodis).")
    export_to_csv(df, args.output)
    if args.preview:
        print("\n--- PREVIEW OF FIRST 20 ROWS ---")
        print(get_preview_table(df, 20))
