import gc
from pathlib import Path

import geopandas as gpd
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/crime_raw")
INTERMEDIATE_DIR = Path("data/intermediate")
YEARS = range(2010, 2026)

CENSUS_CRS = "EPSG:4269"  # must match load_311.py and load_stop_and_frisk.py


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_year(year: int) -> pd.DataFrame:
    """
    Reads raw NYPD felony complaint CSV for a given year.
    Cleans column names, coerces lat/lon to numeric, drops missing coords.
    """
    path = RAW_DIR / f"NYPD_Complaint_Data_{year}.csv"
    print(f"  Loading {path.name} ...")

    df = pd.read_csv(path, low_memory=False)

    # lowercase column names for consistency
    df.columns = df.columns.str.strip().str.lower()

    # coerce coords to numeric (handles blank strings)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    dropped = before - len(df)

    print(f"  {year}: {len(df):,} rows with valid coordinates ({dropped:,} dropped)")
    return df


def load_nyc_tracts() -> gpd.GeoDataFrame:
    """
    Loads cached NYC tracts saved by load_311.py.
    """
    cache_path = INTERMEDIATE_DIR / "nyc_tracts.parquet"
    if not cache_path.exists():
        raise FileNotFoundError(
            "NYC tracts not found. Run load_311.py first to generate "
            "data/intermediate/nyc_tracts.parquet"
        )
    print("  Loading cached NYC tracts ...")
    return gpd.read_parquet(cache_path)


# ---------------------------------------------------------------------------
# Per-year processing
# ---------------------------------------------------------------------------


def process_year(
    year: int,
    nyc_tracts: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    print(f"\n--- Processing {year} ---")

    # 1. Load
    df = load_year(year)

    # 2. Convert to GeoDataFrame
    gdf_points = gpd.GeoDataFrame(
        df[["latitude", "longitude"]],
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    ).to_crs(CENSUS_CRS)
    del df
    gc.collect()

    # 3. Spatial join
    joined = gpd.sjoin(
        nyc_tracts[["GEOID", "geometry"]],
        gdf_points[["geometry"]],
        how="left",
        predicate="contains",
    )
    del gdf_points
    gc.collect()

    # 4. Count felonies per tract
    counts = joined.groupby("GEOID").size().reset_index(name=f"felony{year}")

    result = nyc_tracts[["GEOID", "geometry"]].merge(counts, on="GEOID", how="left")
    result[f"felony{year}"] = result[f"felony{year}"].fillna(0).astype(int)

    # 5. Replace 1 -> 0
    #    Spatial artifact: tracts with count of 1 treated as zero
    result[f"felony{year}"] = result[f"felony{year}"].replace(1, 0)

    result = gpd.GeoDataFrame(result, crs=CENSUS_CRS)

    print(
        f"  {year}: {result[f'felony{year}'].sum():,} total felonies across {len(result)} tracts"
    )
    return result


# ---------------------------------------------------------------------------
# Merge + save
# ---------------------------------------------------------------------------


def merge_years(yearly_gdfs: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """
    Merges per-year felony GeoDataFrames into one on GEOID.
    """
    base = yearly_gdfs[0]
    for gdf in yearly_gdfs[1:]:
        col = [c for c in gdf.columns if c.startswith("felony")][0]
        base = base.merge(gdf[["GEOID", col]], on="GEOID", how="left")

    felony_cols = sorted([c for c in base.columns if c.startswith("felony")])
    return gpd.GeoDataFrame(
        base[["GEOID"] + felony_cols + ["geometry"]], crs=CENSUS_CRS
    )


def merge_all_datasets(crime_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Joins crime counts with existing 311 + stop & frisk combined dataset.
    Produces the final analysis-ready parquet.
    """
    combined_path = INTERMEDIATE_DIR / "stopfrisk_311.parquet"
    if not combined_path.exists():
        print("Warning: stopfrisk_311.parquet not found, saving crime data only.")
        return crime_gdf

    print("\nLoading stop & frisk + 311 combined dataset ...")
    existing = gpd.read_parquet(combined_path)

    felony_cols = [c for c in crime_gdf.columns if c.startswith("felony")]
    merged = existing.merge(
        crime_gdf[["GEOID"] + felony_cols],
        on="GEOID",
        how="left",
    )
    return gpd.GeoDataFrame(merged, crs=CENSUS_CRS)


def run():
    """
    Entry point. Processes all years and saves:
      data/intermediate/crime_counts.parquet   (crime only)
      data/final/analysis_ready.parquet        (311 + stop & frisk + crime)
    """
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    Path("data/final").mkdir(parents=True, exist_ok=True)

    nyc_tracts = load_nyc_tracts()

    yearly_gdfs = []
    for year in YEARS:
        gdf = process_year(year, nyc_tracts)
        yearly_gdfs.append(gdf)
        gc.collect()

    print("\nMerging crime years ...")
    crime_all = merge_years(yearly_gdfs)

    # Save crime-only output
    crime_path = INTERMEDIATE_DIR / "crime_counts.parquet"
    crime_all.to_parquet(crime_path)
    print(f"Saved -> {crime_path}")

    # Merge with 311 + stop & frisk -> final analysis dataset
    final = merge_all_datasets(crime_all)
    final_path = Path("data/final/analysis_ready.parquet")
    final.to_parquet(final_path)
    print(f"Saved -> {final_path}")

    print("\nFinal dataset preview:")
    print(final.drop(columns="geometry").head())

    return final


if __name__ == "__main__":
    run()
