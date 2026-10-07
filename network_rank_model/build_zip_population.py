import os
from pathlib import Path
import requests
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import rasterstats
import numpy as np

# ─────────────────────────────────────────────────────────────
# 1. Helper: download with resume-skip and progress
# ─────────────────────────────────────────────────────────────
def download_file(url, out_path, chunk_size=8192):
    if os.path.exists(out_path):
        print(f"{out_path} already exists, skipping download")
        return out_path
    print(f"Downloading {url} -> {out_path}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r{downloaded/1e6:.1f} / {total/1e6:.1f} MB", end="")
    print()
    return out_path

# ─────────────────────────────────────────────────────────────
# 2. Get San Diego County boundary (Census TIGER)
# ─────────────────────────────────────────────────────────────
counties_url = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_500k.zip"
counties = gpd.read_file(counties_url)
sd_county = counties[(counties["STATEFP"] == "06") & (counties["NAME"] == "San Diego")]
if sd_county.empty:
    raise ValueError("San Diego County not found in TIGER county file")

# ─────────────────────────────────────────────────────────────
# 3. Download full USA WorldPop raster (multi-GB — only runs once)
# ─────────────────────────────────────────────────────────────
worldpop_url = "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/USA/usa_ppp_2020.tif"
national_raster = "usa_ppp_2020.tif"
download_file(worldpop_url, national_raster)

# ─────────────────────────────────────────────────────────────
# 4. Clip to San Diego County (skip if already clipped)
# ─────────────────────────────────────────────────────────────
clipped_raster = "san_diego_county_ppp_2020.tif"

if not os.path.exists(clipped_raster):
    print("Clipping national raster to San Diego County...")
    with rasterio.open(national_raster) as src:
        raster_nodata = src.nodata
        geom = sd_county.to_crs(src.crs).geometry
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
else:
    print(f"{clipped_raster} already exists, skipping clip")
    with rasterio.open(clipped_raster) as src:
        raster_nodata = src.nodata

print("Raster nodata value:", raster_nodata)

# ─────────────────────────────────────────────────────────────
# 5. Load ZIP code boundaries
# ─────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[1] / "data"      # network_rank_model/data (parents[0] if the script sits directly in network_rank_model/)
ZIP_GEOJSON = DATA_DIR / "ZIP_CODES_20261007.geojson"

zips = gpd.read_file(ZIP_GEOJSON)
print("CRS in file:", zips.crs)                              # check this before trusting the next lines

if zips.crs is None:
    zips = zips.set_crs("EPSG:4326")  
    
#zips = gpd.read_file("/mnt/user-data/uploads/ZIP_CODES_20261007.geojson")
#zips = zips.set_crs("EPSG:4326")  # file has no CRS defined — standard for this kind of ZIP boundary data

# ─────────────────────────────────────────────────────────────
# 6. Zonal stats — population sum per ZIP
# ─────────────────────────────────────────────────────────────
with rasterio.open(clipped_raster) as src:
    zips_proj = zips.to_crs(src.crs)

stats = rasterstats.zonal_stats(
    zips_proj,
    clipped_raster,
    stats=["sum", "count"],
    nodata=raster_nodata if raster_nodata is not None else -99999,
    all_touched=False,  # center-of-cell rule — avoids double counting across ZIP boundaries
)

zips["population"] = [s["sum"] if s["sum"] is not None else 0 for s in stats]

# ─────────────────────────────────────────────────────────────
# 7. Area + density
# ─────────────────────────────────────────────────────────────
zips_equal_area = zips.to_crs("EPSG:6933")  # equal-area projection for accurate km² at this scale
zips["area_km2"] = zips_equal_area.geometry.area / 1e6
zips["density_per_km2"] = zips["population"] / zips["area_km2"]

# ─────────────────────────────────────────────────────────────
# 8. Results
# ─────────────────────────────────────────────────────────────
print(
    zips[["zip", "community", "population", "area_km2", "density_per_km2"]]
    .sort_values("density_per_km2", ascending=False)
    .head(10)
)

zips.to_file("san_diego_zip_population_density.geojson", driver="GeoJSON")
print("Saved san_diego_zip_population_density.geojson")