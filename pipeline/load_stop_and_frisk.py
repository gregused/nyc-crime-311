import gc
from pathlib import Path

import geopandas as gpd
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/stop_frisk_raw")  # e.g. data/stopfrisk_raw/2010.csv
INTERMEDIATE_DIR = Path("data/intermediate")
YEARS = range(2010, 2026)  # 2010-2025 inclusive

# Stop & frisk coords are in NY State Plane Long Island (feet)
STOPFRISK_CRS = "EPSG:2908"

# Must match the CRS used in load_311.py so spatial joins align
CENSUS_CRS = "EPSG:4269"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_year(year: int) -> pd.DataFrame:
    csv_path = RAW_DIR / f"{year}.csv"
    xlsx_path = RAW_DIR / f"{year}.xlsx"

    if csv_path.exists():
        print(f"  Loading {csv_path.name} ...")
        df = pd.read_csv(csv_path, low_memory=False, encoding="latin-1")

    elif xlsx_path.exists():
        print(f"  Loading {xlsx_path.name} ...")
        df = pd.read_excel(xlsx_path)
    else:
        raise FileNotFoundError(f"No stop & frisk file found for {year}")

    df.columns = df.columns.str.strip().str.lower()

    # Normalize coordinate column names
    # 2010-2016: xcoord/ycoord
    # 2017-2025: stop_location_x / stop_location_y
    if "stop_location_x" in df.columns:
        df = df.rename(
            columns={
                "stop_location_x": "xcoord",
                "stop_location_y": "ycoord",
            }
        )

    before = len(df)

    df["xcoord"] = pd.to_numeric(df["xcoord"], errors="coerce")
    df["ycoord"] = pd.to_numeric(df["ycoord"], errors="coerce")

    df = df.dropna(subset=["xcoord", "ycoord"])
    dropped = before - len(df)
    print(f"  {year}: {len(df):,} rows with valid coordinates ({dropped:,} dropped)")

    return df


def to_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Converts DataFrame with xcoord/ycoord (EPSG:2908) to GeoDataFrame
    reprojected to CENSUS_CRS (EPSG:4269).
    """
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["xcoord"], df["ycoord"]),
        crs=STOPFRISK_CRS,
    )
    return gdf.to_crs(CENSUS_CRS)


def load_nyc_tracts() -> gpd.GeoDataFrame:
    """
    Loads cached NYC tracts saved by load_311.py.
    Avoids re-downloading from the Census API.
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
# Main pipeline
# ---------------------------------------------------------------------------


def process_year(
    year: int,
    nyc_tracts: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Full pipeline for one year:
      1. Load raw CSV
      2. Reproject from EPSG:2908 -> EPSG:4269
      3. Spatial join -> count stops per tract

    Returns GeoDataFrame with columns [GEOID, geometry, stop{year}].
    """
    print(f"\n--- Processing {year} ---")

    # 1. Load & clean
    df = load_year(year)

    # 2. Convert to spatial + reproject
    gdf_points = to_geodataframe(df)
    del df
    gc.collect()

    # 3. Spatial join: which tract contains each stop?
    joined = gpd.sjoin(
        nyc_tracts[["GEOID", "geometry"]],
        gdf_points[["geometry"]],
        how="left",
        predicate="contains",
    )
    del gdf_points
    gc.collect()

    # 4. Count stops per tract
    counts = joined.groupby("GEOID").size().reset_index(name=f"stop{year}")

    # Re-attach geometry
    result = nyc_tracts[["GEOID", "geometry"]].merge(counts, on="GEOID", how="left")
    result[f"stop{year}"] = result[f"stop{year}"].fillna(0).astype(int)
    result = gpd.GeoDataFrame(result, crs=CENSUS_CRS)

    print(
        f"  {year}: {result[f'stop{year}'].sum():,} total stops across {len(result)} tracts"
    )
    return result


def merge_years(yearly_gdfs: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """
    Merges per-year stop & frisk GeoDataFrames into one on GEOID.
    """
    base = yearly_gdfs[0]
    for gdf in yearly_gdfs[1:]:
        stop_col = [c for c in gdf.columns if c.startswith("stop")][0]
        base = base.merge(gdf[["GEOID", stop_col]], on="GEOID", how="left")

    stop_cols = sorted([c for c in base.columns if c.startswith("stop")])
    return gpd.GeoDataFrame(base[["GEOID"] + stop_cols + ["geometry"]], crs=CENSUS_CRS)


def merge_with_311(
    stopfrisk_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Joins stop & frisk counts with 311 counts on GEOID, producing
    the combined analysis-ready dataset.
    """
    counts_311_path = INTERMEDIATE_DIR / "311_counts.parquet"
    if not counts_311_path.exists():
        raise FileNotFoundError(
            "311 counts not found. Run load_311.py first to generate "
            "data/intermediate/311_counts.parquet"
        )

    print("\nLoading 311 counts ...")
    df_311 = gpd.read_parquet(counts_311_path)

    n311_cols = [c for c in df_311.columns if c.startswith("n311_")]
    stop_cols = [c for c in stopfrisk_gdf.columns if c.startswith("stop")]

    merged = df_311[["GEOID", "geometry"] + n311_cols].merge(
        stopfrisk_gdf[["GEOID"] + stop_cols],
        on="GEOID",
        how="left",
    )

    return gpd.GeoDataFrame(merged, crs=CENSUS_CRS)


def run():
    """
    Entry point — runs the full stop & frisk ingestion pipeline
    and saves two outputs:
      - data/intermediate/stopfrisk_counts.parquet  (stops only)
      - data/intermediate/stopfrisk_311.parquet     (stops + 311 joined)
    """
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    nyc_tracts = load_nyc_tracts()

    yearly_gdfs = []
    for year in YEARS:
        gdf = process_year(year, nyc_tracts)
        yearly_gdfs.append(gdf)
        gc.collect()

    print("\nMerging stop & frisk years ...")
    stopfrisk_all = merge_years(yearly_gdfs)

    # Save stops-only output
    stops_path = INTERMEDIATE_DIR / "stopfrisk_counts.parquet"
    stopfrisk_all.to_parquet(stops_path)
    print(f"Saved -> {stops_path}")

    # Join with 311 and save combined dataset
    combined = merge_with_311(stopfrisk_all)
    combined_path = INTERMEDIATE_DIR / "stopfrisk_311.parquet"
    combined.to_parquet(combined_path)
    print(f"Saved -> {combined_path}")

    print("\nCombined dataset preview:")
    print(combined.drop(columns="geometry").head())

    return combined


if __name__ == "__main__":
    run()
