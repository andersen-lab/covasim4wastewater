"""San Diego ZIP boundaries and sewersheds.

Sewersheds are built by dissolving ZIP-code polygons by wastewater treatment plant,
using a CSV that assigns each ZIP to a plant. ZIPs missing from the CSV are treated
as unassigned (outside the sewersheds); ZIPs in the CSV with no polygon are reported.

Writes to --outdir (default: this folder):
  sandiego_zip_boundaries.geojson   one polygon per ZIP, with its sewershed
  sandiego_sewersheds.geojson       one polygon per treatment plant
  sandiego_zip_to_sewershed.csv     zip, community, sewershed
and sandiego_sewersheds.{pdf,png}.

Run with the sl_rota_noro environment (needs geopandas, matplotlib).
"""

import argparse
import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from shapely.geometry import Polygon
from shapely.ops import unary_union

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR

# Okabe-Ito colorblind-safe colors
PLANT_COLORS = {'Point Loma': '#0072B2', 'Encina': '#E69F00', 'South Bay': '#009E73'}
UNASSIGNED_COLOR = '#d9d9d9'
# Neighbouring ZIP polygons don't meet exactly, so dissolving leaves slivers. Close gaps up to
# 2 * CLOSE_DEG (~20 m) and fill interior holes under MAX_HOLE_KM2 (e.g. tiny unassigned ZIP 92161).
CLOSE_DEG = 1e-4
MAX_HOLE_KM2 = 1.0


def load_zips(path):
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(4326)
    gdf['geometry'] = gdf.geometry.make_valid()
    gdf['zip'] = gdf['zip'].astype(str).str.zfill(5)
    # several ZIPs are split across multiple features
    return gdf.dissolve(by='zip', as_index=False, aggfunc={'community': 'first'})


def load_catchments(path):
    df = pd.read_csv(path, dtype={'Zip_code': str})
    df['Zip_code'] = df['Zip_code'].str.zfill(5)
    df = df.rename(columns={'Zip_code': 'zip', 'Wastewater_treatment_plant': 'sewershed'})
    if df['zip'].duplicated().any():
        raise ValueError(f"ZIPs assigned to more than one catchment: {sorted(df.loc[df['zip'].duplicated(), 'zip'])}")
    return df[['zip', 'sewershed']]


def clean_sewershed(geom):
    geom = geom.buffer(CLOSE_DEG, join_style='mitre').buffer(-CLOSE_DEG, join_style='mitre')
    deg2_per_km2 = 1 / (111.0 ** 2 * math.cos(math.radians(geom.centroid.y)))
    polys = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
    kept = [Polygon(p.exterior, [i for i in p.interiors if Polygon(i).area > MAX_HOLE_KM2 * deg2_per_km2])
            for p in polys]
    return unary_union(kept)


def plot_map(zips, sewersheds, out_base):
    fig, ax = plt.subplots(figsize=(7, 7))
    unassigned = zips[zips['sewershed'].isna()]
    unassigned.plot(ax=ax, color=UNASSIGNED_COLOR, edgecolor='white', linewidth=0.3)
    assigned = zips[zips['sewershed'].notna()]
    assigned.plot(ax=ax, color=assigned['sewershed'].map(PLANT_COLORS), alpha=0.55,
                  edgecolor='white', linewidth=0.3)
    sewersheds.boundary.plot(ax=ax, color=sewersheds['sewershed'].map(PLANT_COLORS), linewidth=1.8)
    handles = [Patch(facecolor=PLANT_COLORS.get(n, '#888'), edgecolor='none', alpha=0.8, label=n)
               for n in sewersheds['sewershed']]
    handles.append(Patch(facecolor=UNASSIGNED_COLOR, edgecolor='none', label='Not in a sewershed'))
    ax.legend(handles=handles, title='Treatment plant', loc='lower left', frameon=False, fontsize=8, title_fontsize=9)
    ax.set_aspect(1 / cos_lat(zips))
    ax.set_axis_off()
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_base}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def cos_lat(gdf):
    """cos(mean latitude), so lon/lat degrees plot with roughly correct proportions."""
    return math.cos(math.radians(gdf.geometry.union_all().centroid.y))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--zips', type=Path, default=DATA_DIR / 'ZIP_CODES_20261007.geojson')
    ap.add_argument('--catchments', type=Path, default=DATA_DIR / 'catchment_zipcodes.csv')
    ap.add_argument('--outdir', type=Path, default=DATA_DIR)
    ap.add_argument('--figdir', type=Path, default=SCRIPT_DIR)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    zips = load_zips(args.zips)
    catch = load_catchments(args.catchments)

    no_polygon = sorted(set(catch['zip']) - set(zips['zip']))
    if no_polygon:
        print(f'WARNING: ZIPs in catchment CSV with no polygon (not drawn): {no_polygon}')
    unassigned = sorted(set(zips['zip']) - set(catch['zip']))
    print(f'{len(unassigned)} ZIP polygons not in the catchment CSV (left unassigned)')

    zips = zips.merge(catch, on='zip', how='left')
    sewersheds = zips.dropna(subset=['sewershed']).dissolve(by='sewershed', as_index=False)[['sewershed', 'geometry']]
    sewersheds['geometry'] = sewersheds.geometry.apply(clean_sewershed)
    sewersheds['n_zips'] = sewersheds['sewershed'].map(zips.groupby('sewershed').size())

    zips.to_file(args.outdir / 'sandiego_zip_boundaries.geojson', driver='GeoJSON')
    sewersheds.to_file(args.outdir / 'sandiego_sewersheds.geojson', driver='GeoJSON')
    zips[['zip', 'community', 'sewershed']].sort_values('zip').to_csv(
        args.outdir / 'sandiego_zip_to_sewershed.csv', index=False)

    plot_map(zips, sewersheds, args.figdir / 'sandiego_sewersheds')
    print(sewersheds.drop(columns='geometry').to_string(index=False))


if __name__ == '__main__':
    main()
