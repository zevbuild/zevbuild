"""
Kalyan Historical Data Scraper and Sanitizer.
Extracts historical Kalyan Matka records from https://dpbossx.net/kalyan-penal-chart.php
and structures them into a normalized chronological dataset.
"""

import os
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
from tabulate import tabulate

URL = "https://dpbossx.net/kalyan-penal-chart.php"
DEFAULT_OUTPUT_CSV = "kalyan_historical_data.csv"
LOCAL_CACHE_HTML = "kalyan_penal_chart.html"
DAYS_OF_WEEK = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]


def fetch_html(url: str = URL, cache_file: str = LOCAL_CACHE_HTML) -> str:
    """
    Fetch HTML content. Prefers local cache if present, otherwise fetches live
    and caches the response locally.
    """
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 1000:
        print(f"[INFO] Loading cached HTML from '{cache_file}'...")
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read()

    print(f"[INFO] Fetching live data from {url}...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        content = response.text
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[INFO] Cached response to '{cache_file}'.")
        return content
    except Exception as e:
        print(f"[WARN] Live request failed: {e}")
        if os.path.exists(cache_file):
            print(f"[INFO] Falling back to existing cache '{cache_file}'...")
            with open(cache_file, "r", encoding="utf-8") as f:
                return f.read()
        raise


def parse_week_start_date(raw_cell_text: str) -> datetime:
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


def parse_kalyan_chart(html_content: str) -> pd.DataFrame:
    """
    Parse the Kalyan Panel Record table into a normalized DataFrame.
    Each week row contains 1 Date column and 6 days x 3 columns = 19 columns.
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
        # Skip header rows
        if row.find("th"):
            continue
        
        tds = row.find_all("td")
        if len(tds) < 19:
            continue

        raw_date_cell = "".join(str(c) for c in tds[0].contents)
        start_date = parse_week_start_date(raw_date_cell)
        if not start_date:
            continue

        # Process each day: Mon to Sat
        for day_idx, day_name in enumerate(DAYS_OF_WEEK):
            day_date = start_date + timedelta(days=day_idx)
            col_base = 1 + day_idx * 3
            
            # Jodi is the center column of the 3-column day block
            jodi_td = tds[col_base + 1]
            raw_jodi = jodi_td.get_text(strip=True)

            # Sanitize raw Jodi
            # Standardize valid two-digit numeric strings (00-99)
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

            records.append({
                "Date": day_date.strftime("%Y-%m-%d"),
                "Day_Of_Week": day_name,
                "Jodi": jodi_val,
                "Open_Digit": open_digit,
                "Close_Digit": close_digit,
                "Is_Valid": is_valid,
                "Raw_Entry": raw_jodi,
            })

    df = pd.DataFrame(records)
    # Sort chronologically
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date").reset_index(drop=True)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    # Cast digit columns to nullable Int64 for clean integer formatting in CSV
    df["Open_Digit"] = df["Open_Digit"].astype("Int64")
    df["Close_Digit"] = df["Close_Digit"].astype("Int64")
    return df


def get_preview_table(df: pd.DataFrame, n: int = 20) -> str:
    """Generate an ASCII formatted preview of the first n rows."""
    preview_df = df.head(n).copy()
    preview_df["#"] = range(1, len(preview_df) + 1)
    preview_df["Jodi"] = preview_df["Jodi"].apply(lambda v: v if v != "" else "null")
    preview_df["Open_Digit"] = preview_df["Open_Digit"].apply(lambda v: str(int(v)) if pd.notna(v) and v is not None else "null")
    preview_df["Close_Digit"] = preview_df["Close_Digit"].apply(lambda v: str(int(v)) if pd.notna(v) and v is not None else "null")
    cols = ["#", "Date", "Day_Of_Week", "Jodi", "Open_Digit", "Close_Digit", "Is_Valid", "Raw_Entry"]
    return tabulate(preview_df[cols], headers="keys", tablefmt="github", showindex=False)


def export_to_csv(df: pd.DataFrame, output_path: str = DEFAULT_OUTPUT_CSV) -> str:
    """Export the sanitized DataFrame to local CSV file."""
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[SUCCESS] Dataset successfully written to: {os.path.abspath(output_path)}")
    return output_path


if __name__ == "__main__":
    print("[INFO] Fetching and parsing Kalyan chart...")
    html = fetch_html()
    df = parse_kalyan_chart(html)
    print(f"[INFO] Parsed {len(df)} total daily records ({df['Is_Valid'].sum()} valid Jodis).")
    print("\n--- PREVIEW OF FIRST 20 ROWS ---")
    print(get_preview_table(df, 20))
