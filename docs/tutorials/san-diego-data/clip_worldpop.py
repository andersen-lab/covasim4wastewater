"""Make san_diego_county_ppp_2020.tif: WorldPop's 2020 US population-count raster clipped to San Diego County.

WorldPop's server does not support partial (range) downloads, so the whole ~4 GB US file is downloaded
once to --raw (kept outside this folder by default, delete it afterwards) and the county window is cut out.
The window is the bounding box of the ZIP-code file plus a small margin. The output keeps WorldPop's
float32 people-per-pixel values, EPSG:4326 and nodata (-99999), which is what covasim's `density_raster` expects.

Run with an environment that has rasterio and geopandas (e.g. covasim's):
    python -I clip_worldpop.py
"""

import argparse
import shutil
import tempfile
import urllib.request
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds

HERE = Path(__file__).resolve().parent
URL = 'https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/USA/usa_ppp_2020.tif'
MARGIN_DEG = 0.01


def download(url, path):
    print(f'Downloading {url} -> {path}')
    with urllib.request.urlopen(url) as r, open(path, 'wb') as f:
        shutil.copyfileobj(r, f, length=1 << 20)


def clip(raw, zips, out):
    b = gpd.read_file(zips).total_bounds
    bbox = (b[0] - MARGIN_DEG, b[1] - MARGIN_DEG, b[2] + MARGIN_DEG, b[3] + MARGIN_DEG)
    with rasterio.open(raw) as src:
        win = from_bounds(*bbox, transform=src.transform).round_offsets().round_lengths()
        data = src.read(1, window=win)
        profile = src.profile
        transform = src.window_transform(win)
    profile.update(height=data.shape[0], width=data.shape[1], transform=transform,
                   compress='deflate', predictor=3, tiled=False)
    for key in ('blockxsize', 'blockysize'):
        profile.pop(key, None)
    with rasterio.open(out, 'w', **profile) as dst:
        dst.write(data, 1)
    print(f'Wrote {out}: {data.shape[1]} x {data.shape[0]} pixels, dtype {data.dtype}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url', default=URL)
    ap.add_argument('--raw', type=Path, default=Path(tempfile.gettempdir()) / 'usa_ppp_2020.tif',
                    help='where the full US raster is (or will be) downloaded')
    ap.add_argument('--zips', type=Path, default=HERE / 'ZIP_CODES_20261007.geojson')
    ap.add_argument('--out', type=Path, default=HERE / 'san_diego_county_ppp_2020.tif')
    args = ap.parse_args()
    if not args.raw.exists():
        download(args.url, args.raw)
    clip(args.raw, args.zips, args.out)


if __name__ == '__main__':
    main()
