"""Build ZIP-level population and density for San Diego County from WorldPop.

Run:  python network_rank_model/src/build_zip_population.py
Inputs  (network_rank_model/data/): ZIP_CODES_20261007.geojson
Outputs (network_rank_model/data/): usa_ppp_2020.tif, san_diego_county_ppp_2020.tif,
                                    san_diego_zip_population_density.geojson
Everything in data/ is gitignored.

=============================== GLOSSARY ====================================
    WorldPop        : research project that publishes gridded population estimates
                      (modeled from census + satellite data, not an exact count).
    ppp             : "people per pixel". Each cell of the raster holds the
                      estimated number of people living in that cell.
    raster / .tif   : a grid of numbers laid over a map, stored as a GeoTIFF file.
                      usa_ppp_2020.tif = whole US, ~100 m cells, 3.8 GB.
    clip            : cut a raster down to a smaller area. Cells outside the area
                      are set to nodata.
    nodata          : value meaning "no data here" (WorldPop typically -99999).
    GeoJSON         : text file of map shapes (polygons) plus a table of
                      attributes (zip code, community name, ...).
    polygon         : a closed shape on a map; here, the outline of a ZIP code.
    ZIP boundaries  : ZIP_CODES_20261007.geojson, one polygon per ZIP code.
    TIGER           : US Census Bureau's boundary files. We use it for the
                      San Diego County outline.
    FIPS / STATEFP  : numeric ID codes; "06" = California.
    CRS             : coordinate reference system, i.e. how coordinates map to
                      the Earth.
    EPSG:4326       : plain longitude/latitude in degrees (WorldPop and GeoJSON use this).
    EPSG:6933       : an equal-area projection (units = metres), used so that
                      polygon areas in km^2 are accurate.
    zonal sum       : for each polygon, add up the raster cells that fall inside it.
    all_touched     : False = a cell counts for a ZIP only if its centre is inside
                      the ZIP (avoids double-counting cells on borders).
    density_per_km2 : population / area_km2 for each ZIP.
    .part file      : temporary name used while downloading; renamed to the real
                      name only once the download is complete.

=============================== PIPELINE ====================================

STEP 1  download_file(url, out_path)
    IN : URL of the national WorldPop raster, where to save it
    OUT: data/usa_ppp_2020.tif (skipped if it already exists)
    HOW: streams to a .part file, checks the byte count, then renames.

STEP 2  load_zips(path)
    IN : data/ZIP_CODES_20261007.geojson
    OUT: GeoDataFrame (table with a geometry column), one row per ZIP polygon,
         with columns including 'zip' and 'community'
    HOW: checks the CRS, bounds and column names early so it fails fast.

STEP 3  get_county_boundary()
    IN : (downloads Census TIGER county shapes)
    OUT: one-row GeoDataFrame, the San Diego County outline

STEP 4  clip_raster(county, national_raster, clipped_raster)
    IN : county outline, the 3.8 GB national raster
    OUT: data/san_diego_county_ppp_2020.tif (~8 MB), returns the nodata value
    HOW: keeps only cells inside the county outline and crops to its bounding
         rectangle. THIS is the file rank_network.py reads to place agents.

STEP 5  add_population(zips, clipped_raster, nodata)
    IN : ZIP polygons, clipped raster
    OUT: zips with a new 'population' column
    HOW: zonal sum of the raster cells inside each ZIP.

STEP 6  add_density(zips)
    IN : zips with population
    OUT: zips with 'area_km2' and 'density_per_km2' columns

STEP 7  main()
    Runs steps 2 -> 3 -> 1 -> 4 -> 5 -> 6, prints total population (expect
    ~3.3 million) and the 10 densest ZIPs, and saves
    data/san_diego_zip_population_density.geojson (same ZIP boundaries,
    plus the population / area / density columns).

=========================== HOW THIS FEEDS rank_network.py ==================
    san_diego_county_ppp_2020.tif -> density_raster: where agents are placed.
    san_diego_zip_population_density.geojson -> not used for placement; useful
    to check that sampled agents per ZIP match the real ZIP populations.
"""
import os
from pathlib import Path

import requests
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import rasterstats

# ---- Paths: everything lives in network_rank_model/data ----
#DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

NATIONAL_RASTER = DATA_DIR / "usa_ppp_2020.tif"
CLIPPED_RASTER  = DATA_DIR / "san_diego_county_ppp_2020.tif"
ZIP_GEOJSON     = DATA_DIR / "ZIP_CODES_20261007.geojson"
OUT_GEOJSON     = DATA_DIR / "san_diego_zip_population_density.geojson"

COUNTIES_URL  = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_500k.zip"
WORLDPOP_URL  = "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/USA/usa_ppp_2020.tif"


# ---------------------------------------------------------------
# 1. Helper: download to a .part file, rename only when complete
# ---------------------------------------------------------------
def download_file(url, out_path, chunk_size=1 << 20):
    out_path = Path(out_path)
    if out_path.exists():
        print(f"{out_path} already exists, skipping download")
        return out_path

    tmp = out_path.with_name(out_path.name + ".part")     # a partial file never looks finished
    print(f"Downloading {url} -> {out_path}")
    downloaded, total = 0, 0
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r{downloaded/1e6:.1f} / {total/1e6:.1f} MB", end="")
    print()
    if total and downloaded != total:
        raise IOError(f"Incomplete download: {downloaded} of {total} bytes (kept {tmp})")
    tmp.rename(out_path)
    return out_path


# ---------------------------------------------------------------
# 2. Load and check the ZIP boundaries first (fast; fails early)
# ---------------------------------------------------------------
def load_zips(path):
    zips = gpd.read_file(path)
    print("CRS in file:", zips.crs)
    if zips.crs is None:
        zips = zips.set_crs("EPSG:4326")                   # GeoJSON default: lon/lat
    print("bounds:", zips.total_bounds)                    # expect about [-117.6, 32.5, -116.1, 33.5]
    print("columns:", zips.columns.tolist())

    needed = {"zip", "community"}
    missing = needed - set(zips.columns)
    if missing:
        raise KeyError(f"Missing columns {missing}; edit the names in this script. Found: {zips.columns.tolist()}")

    print(len(zips), "polygons,", zips["zip"].duplicated().sum(), "duplicate ZIP rows")
    return zips


# ---------------------------------------------------------------
# 3. San Diego County boundary (Census TIGER)
# ---------------------------------------------------------------
def get_county_boundary():
    counties = gpd.read_file(COUNTIES_URL)
    sd = counties[(counties["STATEFP"] == "06") & (counties["NAME"] == "San Diego")]
    if sd.empty:
        raise ValueError("San Diego County not found in TIGER county file")
    return sd


# ---------------------------------------------------------------
# 4. Clip the national raster to the county (skipped if already done)
# ---------------------------------------------------------------
def clip_raster(county, national_raster, clipped_raster):
    if clipped_raster.exists():
        print(f"{clipped_raster} already exists, skipping clip")
        with rasterio.open(clipped_raster) as src:
            return src.nodata

    print("Clipping national raster to San Diego County...")
    with rasterio.open(national_raster) as src:
        nodata = src.nodata
        geom = county.to_crs(src.crs).geometry
        out_image, out_transform = mask(src, geom, crop=True)
        out_meta = src.meta.copy()

    out_meta.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
    })
    with rasterio.open(clipped_raster, "w", **out_meta) as dest:
        dest.write(out_image)
    print(f"Saved {clipped_raster}")
    return nodata


# ---------------------------------------------------------------
# 5. Population per ZIP (zonal sum) and density
# ---------------------------------------------------------------
def add_population(zips, clipped_raster, nodata):
    with rasterio.open(clipped_raster) as src:
        zips_proj = zips.to_crs(src.crs)

    stats = rasterstats.zonal_stats(
        zips_proj,
        str(clipped_raster),                               # some versions need a str, not a Path
        stats=["sum", "count"],
        nodata=nodata if nodata is not None else -99999,
        all_touched=False,                                 # center-of-cell rule
    )
    zips["population"] = [s["sum"] if s["sum"] is not None else 0 for s in stats]
    return zips


def add_density(zips):
    equal_area = zips.to_crs("EPSG:6933")                  # equal-area projection for km2
    zips["area_km2"] = equal_area.geometry.area / 1e6
    zips["density_per_km2"] = zips["population"] / zips["area_km2"]
    return zips


# ---------------------------------------------------------------
# 6. Run everything
# ---------------------------------------------------------------
def main():
    zips = load_zips(ZIP_GEOJSON)                          # fails fast on wrong columns
    county = get_county_boundary()

    download_file(WORLDPOP_URL, NATIONAL_RASTER)           # multi-GB, runs once
    nodata = clip_raster(county, NATIONAL_RASTER, CLIPPED_RASTER)
    print("Raster nodata value:", nodata)

    zips = add_population(zips, CLIPPED_RASTER, nodata)
    zips = add_density(zips)

    print(f"Total population in ZIPs: {zips['population'].sum():,.0f}  (county is about 3.3 million)")
    print(
        zips[["zip", "community", "population", "area_km2", "density_per_km2"]]
        .sort_values("density_per_km2", ascending=False)
        .head(10)
    )

    zips.to_file(OUT_GEOJSON, driver="GeoJSON")
    print(f"Saved {OUT_GEOJSON}")


if __name__ == "__main__":
    main()