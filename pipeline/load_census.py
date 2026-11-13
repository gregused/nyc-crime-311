"""
pipeline/ingest/load_census.py

Pulls ACS 5-year demographic data for NYC census tracts (2010-2025)
from the US Census Bureau API.


Variables pulled per year :
    - Total population
    - Median household income
    - Median gross rent
    - Educational attainment (bachelor's, master's, PhD)
    - Employment by occupation
    - Race/ethnicity (Black, White, Latino, Asian)
    - Age 18-24
    - Single mother households
    - Poverty
    - Unemployment rate (computed from labor force variables)
"""

import os
from pathlib import Path

import pandas as pd
from census import Census

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INTERMEDIATE_DIR = Path("data/intermediate")
OUTPUT_DIR = Path("data/final")

# Load API key from environment variable
CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY")

STATE = "36"  # NY FIPS code
COUNTIES = [
    "061",  # New York (Manhattan)
    "047",  # Kings (Brooklyn)
    "005",  # Bronx
    "081",  # Queens
    "085",  # Richmond (Staten Island)
]

YEARS = range(2010, 2026)

# ACS variables — same across all years 2010-2025
ACS_VARS = {
    "B01003_001E": "tot_pop",
    "B19013_001E": "median_income",
    "B25064_001E": "median_rent",
    "B15002_001E": "edu_total",
    "B15002_015E": "male_bachelor",
    "B15002_032E": "female_bachelor",
    "B15002_016E": "male_masters",
    "B15002_033E": "female_masters",
    "B15002_018E": "male_phd",
    "B15002_035E": "female_phd",
    "C24030_001E": "employ_tot",
    "C24030_017E": "male_emp_sci",
    "C24030_044E": "female_emp_sci",
    "C24030_021E": "male_edu_serv",
    "C24030_048E": "female_edu_serv",
    "C24010_003E": "male_manage_prof",
    "C24010_039E": "female_manage_prof",
    "B03002_003E": "white",
    "B03002_004E": "black",
    "B03002_012E": "latino",
    "B03002_006E": "asian",
    "B01003_001E": "tot_pop",
    "B06001_001E": "tot_age",
    "B06001_004E": "age18_24",
    "B09005_005E": "single_mom",
    "B06012_002E": "poverty",
}

# Labor force variables for unemployment rate calculation
# Males in labor force: B23001_004, 011, 018, ... (every 7th starting at 4)
LABOR_FORCE_MALE = [f"B23001_{str(i).zfill(3)}E" for i in range(4, 68, 7)]
LABOR_FORCE_FEMALE = [f"B23001_{str(i).zfill(3)}E" for i in range(90, 154, 7)]

# Unemployed variables
UNEMPLOYED_MALE = [f"B23001_{str(i).zfill(3)}E" for i in range(8, 72, 7)]
UNEMPLOYED_FEMALE = [f"B23001_{str(i).zfill(3)}E" for i in range(94, 158, 7)]


# ---------------------------------------------------------------------------
# Census API helpers
# ---------------------------------------------------------------------------


def get_census_client() -> Census:
    """Initializes the Census API client."""
    if CENSUS_API_KEY == "YOUR_KEY_HERE":
        raise ValueError(
            "Census API key not set. Add CENSUS_API_KEY to your .env file "
            "or set it directly in this script. "
            "Get a free key at: https://api.census.gov/data/key_signup.html"
        )
    return Census(CENSUS_API_KEY)


def fetch_tracts(c: Census, variables: list, year: int) -> pd.DataFrame:
    """
    Fetches ACS 5-year data for all NYC census tracts.

    Note: ACS 5-year estimates lag by ~1 year.
          e.g. year=2022 returns 2018-2022 5-year estimates.
          Latest available is typically current_year - 2.
    """
    frames = []
    for county in COUNTIES:
        data = c.acs5.state_county_tract(
            fields=["NAME"] + variables,
            state_fips=STATE,
            county_fips=county,
            tract="*",
            year=year,
        )
        frames.append(pd.DataFrame(data))

    df = pd.concat(frames, ignore_index=True)

    # Build GEOID from state + county + tract
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    df = df.drop(columns=["state", "county", "tract", "NAME"], errors="ignore")

    return df


# ---------------------------------------------------------------------------
# Unemployment rate
# ---------------------------------------------------------------------------


def compute_unemployment_rate(c: Census, year: int) -> pd.DataFrame:
    """
    Computes unemployment rate per tract for a given year.

    """
    all_labor_vars = LABOR_FORCE_MALE + LABOR_FORCE_FEMALE
    all_unemp_vars = UNEMPLOYED_MALE + UNEMPLOYED_FEMALE

    # fetch labor force and unemployed counts
    labor_raw = fetch_tracts(c, all_labor_vars, year)
    unemp_raw = fetch_tracts(c, all_unemp_vars, year)

    # sum across all age/sex categories per tract
    labor_cols = [col for col in labor_raw.columns if col.startswith("B23001")]
    unemp_cols = [col for col in unemp_raw.columns if col.startswith("B23001")]

    labor_raw[labor_cols] = labor_raw[labor_cols].apply(pd.to_numeric, errors="coerce")
    unemp_raw[unemp_cols] = unemp_raw[unemp_cols].apply(pd.to_numeric, errors="coerce")

    labor_df = labor_raw.copy()
    labor_df[f"labor_est{year}"] = labor_raw[labor_cols].sum(axis=1)
    labor_df = labor_df[["GEOID", f"labor_est{year}"]]

    unemp_df = unemp_raw.copy()
    unemp_df[f"unemp_est{year}"] = unemp_raw[unemp_cols].sum(axis=1)
    unemp_df = unemp_df[["GEOID", f"unemp_est{year}"]]

    # join and compute rate
    merged = labor_df.merge(unemp_df, on="GEOID", how="left")
    merged[f"unemp_rate{year}"] = (
        merged[f"unemp_est{year}"] / merged[f"labor_est{year}"]
    )
    merged = merged.fillna(0)

    return merged[["GEOID", f"unemp_rate{year}"]]


# ---------------------------------------------------------------------------
# Per-year processing
# ---------------------------------------------------------------------------


def process_year(c: Census, year: int) -> pd.DataFrame:
    """
    Pulls and processes all demographic variables for one year.

    Returns a DataFrame with columns:
        GEOID, tot_pop_{year}, median_income_{year}, ..., unemp_rate{year}

    Note on ACS availability:
        ACS 5-year data is available from 2009 onward.
        For years > 2022, check Census API availability as newer data
        may not yet be released.
    """
    print(f"\n--- Processing {year} ---")

    # fetch main variables
    variables = list(ACS_VARS.keys())
    # deduplicate
    variables = list(dict.fromkeys(variables))

    print(f"  Fetching ACS variables ...")
    df = fetch_tracts(c, variables, year)

    # rename columns and coerce to numeric
    df = df.rename(columns={k: v for k, v in ACS_VARS.items() if k in df.columns})
    for col in df.columns:
        if col != "GEOID":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # compute derived variables
    df["bachelor"] = df["male_bachelor"] + df["female_bachelor"]
    df["masters"] = df["male_masters"] + df["female_masters"]
    df["phd"] = df["male_phd"] + df["female_phd"]
    df["bachelor_higher"] = df["bachelor"] + df["masters"] + df["phd"]
    df["emp_sci_manag"] = (
        df["male_emp_sci"]
        + df["female_emp_sci"]
        + df["male_edu_serv"]
        + df["female_edu_serv"]
    )
    df["manage_prof"] = df["male_manage_prof"] + df["female_manage_prof"]

    # drop intermediate male/female columns
    drop_cols = [c for c in df.columns if c.startswith(("male_", "female_"))]
    df = df.drop(columns=drop_cols)

    # add suffix to all non-GEOID columns so years don't collide when merging
    df = df.rename(
        columns={col: f"{col}_{year}" for col in df.columns if col != "GEOID"}
    )

    # add unemployment rate
    print(f"  Computing unemployment rate ...")
    unemp = compute_unemployment_rate(c, year)
    df = df.merge(unemp, on="GEOID", how="left")

    print(f"  {year}: {len(df)} tracts, {df.columns.tolist()[:5]} ...")
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run():
    """
    Entry point. Pulls ACS data for all years and saves:
        data/intermediate/census_demographics.parquet

    Note on year availability:
        ACS 5-year estimates are typically released ~1 year after the
        reference year. As of 2025, data through 2023 is available.
        The script will skip years where API returns no data.
    """
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    c = get_census_client()

    yearly_dfs = []
    for year in YEARS:
        try:
            df = process_year(c, year)
            yearly_dfs.append(df)
        except Exception as e:
            print(f"  Skipping {year}: {e}")
            continue

    # merge all years into one wide dataframe on GEOID
    print("\nMerging all years ...")
    census_df = yearly_dfs[0]
    for df in yearly_dfs[1:]:
        census_df = census_df.merge(df, on="GEOID", how="inner")

    out_path = INTERMEDIATE_DIR / "census_demographics.parquet"
    census_df.to_parquet(out_path)
    print(f"\nSaved -> {out_path}")
    print(f"Shape: {census_df.shape}")
    print(census_df.head())

    return census_df


if __name__ == "__main__":
    run()
