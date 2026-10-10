"""
Test fixtures and mock datasets for Kalyan Matka E2E tests.
"""
import os
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent
SAMPLE_DRAWS_CSV = FIXTURES_DIR / "sample_draws.csv"
CORRUPTED_DRAWS_CSV = FIXTURES_DIR / "corrupted_draws.csv"
MOCK_PENAL_CHART_HTML = FIXTURES_DIR / "mock_penal_chart.html"
