"""
pipeline/ingest/load_311.py

Ingests raw 311 CSV files (one per year), drops rows without coordinates,
spatially joins to NYC census tracts, erases water bodies, and returns
a single GeoDataFrame with one column per year (n311_YYYY) keyed by GEOID.

"""

# import re
import gc
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
import pygris

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/311_raw")
INTERMEDIATE_DIR = Path("data/intermediate")
YEARS = range(2010, 2015)  # 2010-2014 inclusive

NYC_COUNTIES = ["Kings", "Bronx", "Queens", "Richmond", "New York"]
STATE = "NY"

# CRS used by tigris/pygris shapefiles (NAD83)
CENSUS_CRS = "EPSG:4269"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicates janitor::clean_names() — lowercases and snake_cases all columns.
    e.g. 'Created Date' -> 'created_date'
    """
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df


def load_year(year: int) -> pd.DataFrame:
    """
    Reads raw 311 CSV for a given year, cleans column names,
    and drops rows missing lat/lon.
    """
    path = RAW_DIR / f"311_Service_Requests_from_{year}_to_end{year}.csv"
    print(f"  Loading {path.name} ...")

    df = pd.read_csv(path, low_memory=False)
    df = clean_columns(df)
    df = df.dropna(subset=["latitude", "longitude"])

    print(f"  {year}: {len(df):,} rows with valid coordinates")
    return df


def to_geodataframe(df: pd.DataFrame, crs: str) -> gpd.GeoDataFrame:
    """
    Converts a DataFrame with lat/lon columns to a GeoDataFrame.
    R equivalent: st_as_sf(df, coords = c("longitude", "latitude"))
    """
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=crs,
    )


def erase_water(
    tracts_gdf: gpd.GeoDataFrame, water_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Removes water bodies from tract geometries.
    R equivalent:
        st_erase <- function(x, y) st_difference(x, st_union(y))
    """
    water_union = unary_union(water_gdf.geometry)
    tracts_gdf = tracts_gdf.copy()
    tracts_gdf["geometry"] = tracts_gdf.geometry.difference(water_union)
    return tracts_gdf


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def get_nyc_tracts() -> gpd.GeoDataFrame:
    """
    Downloads NYC census tracts via pygris (mirrors tigris::tracts()).
    Cached to disk after first download.
    """
    cache_path = INTERMEDIATE_DIR / "nyc_tracts.parquet"
    if cache_path.exists():
        print("  Loading cached NYC tracts ...")
        return gpd.read_parquet(cache_path)

    print("  Downloading NYC census tracts via pygris ...")
    tracts = pygris.tracts(state=STATE, county=NYC_COUNTIES, year=2020)
    tracts = tracts.to_crs(CENSUS_CRS)

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    tracts.to_parquet(cache_path)
    return tracts


def get_nyc_water() -> gpd.GeoDataFrame:
    """
    Downloads NYC water areas via pygris (mirrors tigris::area_water()).
    Cached to disk after first download.
    """
    cache_path = INTERMEDIATE_DIR / "nyc_water.parquet"
    if cache_path.exists():
        print("  Loading cached NYC water ...")
        return gpd.read_parquet(cache_path)

    print("  Downloading NYC water bodies via pygris ...")
    water_frames = [
        pygris.area_water(state=STATE, county=county, year=2020)
        for county in NYC_COUNTIES
    ]
    water = pd.concat(water_frames, ignore_index=True)
    water = gpd.GeoDataFrame(water, crs=CENSUS_CRS)

    water.to_parquet(cache_path)
    return water


def process_year(
    year: int,
    nyc_tracts: gpd.GeoDataFrame,
    ny_water: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Full pipeline for one year:
      1. Load raw CSV
      2. Convert to GeoDataFrame
      3. Spatial join -> count 311 calls per tract
      4. Erase water bodies

    Returns a GeoDataFrame with columns [GEOID, geometry, n311_{year}].
    """
    print(f"\n--- Processing {year} ---")

    # 1. Load & clean
    df = load_year(year)

    # 2. Convert to spatial (same CRS as tracts)
    gdf_points = to_geodataframe(df, crs=CENSUS_CRS)
    del df
    gc.collect()

    # 3. Spatial join: which tract contains each 311 call?
    #    R: st_join(nyc_tracts, nyc_geo, join = st_contains)
    #    geopandas sjoin uses "contains" predicate equivalently
    joined = gpd.sjoin(
        nyc_tracts[["GEOID", "geometry"]],
        gdf_points[["geometry"]],
        how="left",
        predicate="contains",
    )
    del gdf_points
    gc.collect()

    # 4. Count calls per tract (R: group_by(GEOID) %>% summarise(n = n()))
    counts = joined.groupby("GEOID").size().reset_index(name=f"n311_{year}")

    # Re-attach geometry from tracts
    result = nyc_tracts[["GEOID", "geometry"]].merge(counts, on="GEOID", how="left")
    result[f"n311_{year}"] = result[f"n311_{year}"].fillna(0).astype(int)
    result = gpd.GeoDataFrame(result, crs=CENSUS_CRS)

    # 5. Erase water (R: st_erase(df_311, ny_water))
    result = erase_water(result, ny_water)

    print(
        f"  {year}: {result[f'n311_{year}'].sum():,} total 311 calls across {len(result)} tracts"
    )
    return result


def merge_years(yearly_gdfs: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """
    Merges per-year GeoDataFrames into one, joining on GEOID.
    R equivalent:
        df_311all <- df_311_10 %>%
          bind_cols(df_311_11 %>% st_drop_geometry() %>% select(-GEOID)) %>%
          ...
    """
    base = yearly_gdfs[0]
    for gdf in yearly_gdfs[1:]:
        count_col = [c for c in gdf.columns if c.startswith("n311_")][0]
        base = base.merge(gdf[["GEOID", count_col]], on="GEOID", how="left")

    # Reorder columns: GEOID, n311_* cols, geometry
    n311_cols = sorted([c for c in base.columns if c.startswith("n311_")])
    base = base[["GEOID"] + n311_cols + ["geometry"]]
    return gpd.GeoDataFrame(base, crs=CENSUS_CRS)


def run():
    """
    Entry point — runs the full 311 ingestion pipeline.
    Saves output to data/intermediate/311_counts.parquet
    """
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    # Download/cache reference geographies
    nyc_tracts = get_nyc_tracts()
    ny_water = get_nyc_water()

    # Process each year
    yearly_gdfs = []
    for year in YEARS:
        gdf = process_year(year, nyc_tracts, ny_water)
        yearly_gdfs.append(gdf)
        gc.collect()

    # Merge all years into one GeoDataFrame
    print("\nMerging all years ...")
    df_311_all = merge_years(yearly_gdfs)

    # Save — parquet preserves geometry; use .shp if you need ArcGIS compat
    out_path = INTERMEDIATE_DIR / "311_counts.parquet"
    df_311_all.to_parquet(out_path)
    print(f"\nSaved -> {out_path}")
    print(
        df_311_all[
            ["GEOID"] + [c for c in df_311_all.columns if c.startswith("n311_")]
        ].head()
    )

    return df_311_all


if __name__ == "__main__":
    run()
