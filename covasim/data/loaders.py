'''
Load data
'''

#%% Housekeeping
import numpy as np
import pandas as pd
import os
import sciris as sc
from . import country_age_data    as cad
from . import state_age_data      as sad
from . import household_size_data as hsd
import functools
import requests
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import country_converter as coco


__all__ = ['get_country_aliases', 'map_entries', 'show_locations', 'get_age_distribution', 'get_household_size', 'get_population_data']


def get_country_aliases():
    ''' Define aliases for countries with odd names in the data '''
    country_mappings = {
       'Bolivia':        'Bolivia (Plurinational State of)',
       'Burkina':        'Burkina Faso',
       'Cape Verde':     'Cabo Verdeo',
       'Hong Kong':      'China, Hong Kong Special Administrative Region',
       'Macao':          'China, Macao Special Administrative Region',
       "Cote d'Ivore":   'Côte d’Ivoire',
       "Ivory Coast":    'Côte d’Ivoire',
       'DRC':            'Democratic Republic of the Congo',
       'Iran':           'Iran (Islamic Republic of)',
       'Laos':           "Lao People's Democratic Republic",
       'Micronesia':     'Micronesia (Federated States of)',
       'Korea':          'Republic of Korea',
       'South Korea':    'Republic of Korea',
       'Moldova':        'Republic of Moldova',
       'Russia':         'Russian Federation',
       'Palestine':      'State of Palestine',
       'Syria':          'Syrian Arab Republic',
       'Taiwan':         'Taiwan Province of China',
       'Macedonia':      'The former Yugoslav Republic of Macedonia',
       'UK':             'United Kingdom of Great Britain and Northern Ireland',
       'United Kingdom': 'United Kingdom of Great Britain and Northern Ireland',
       'Tanzania':       'United Republic of Tanzania',
       'USA':            'United States of America',
       'United States':  'United States of America',
       'Venezuela':      'Venezuela (Bolivarian Republic of)',
       'Vietnam':        'Viet Nam',
        }

    return country_mappings # Convert to lowercase


def map_entries(json, location):
    '''
    Find a match between the JSON file and the provided location(s).

    Args:
        json (list or dict): the data being loaded
        location (list or str): the list of locations to pull from
    '''

    # The data have slightly different formats: list of dicts or just a dict
    countries = [key.lower() for key in json.keys()]

    # Set parameters
    if location is None:
        location = countries
    else:
        location = sc.promotetolist(location)

    # Define a mapping for common mistakes
    mapping = get_country_aliases()
    mapping = {key.lower(): val.lower() for key, val in mapping.items()}

    entries = {}
    for loc in location:
        lloc = loc.lower()
        if lloc not in countries and lloc in mapping:
            lloc = mapping[lloc]
        try:
            ind = countries.index(lloc)
            entry = list(json.values())[ind]
            entries[loc] = entry
        except ValueError as E:
            suggestions = sc.suggest(loc, countries, n=4)
            if suggestions:
                errormsg = f'Location "{loc}" not recognized, did you mean {suggestions}? ({str(E)})'
            else:
                errormsg = f'Location "{loc}" not recognized ({str(E)})'
            raise ValueError(errormsg)

    return entries


def show_locations(location=None, output=False):
    '''
    Print a list of available locations.

    Args:
        location (str): if provided, only check if this location is in the list
        output (bool): whether to return the list (else print)

    **Examples**::

        cv.data.show_locations() # Print a list of valid locations
        cv.data.show_locations('lithuania') # Check if Lithuania is a valid location
        cv.data.show_locations('Viet-Nam') # Check if Viet-Nam is a valid location
    '''
    country_json   = sc.dcp(cad.data)
    state_json     = sc.dcp(sad.data)
    aliases        = get_country_aliases()

    age_data       = sc.mergedicts(state_json, country_json, aliases) # Countries will overwrite states, e.g. Georgia
    household_data = sc.dcp(hsd.data)

    loclist = sc.objdict()
    loclist.age_distributions = sorted(list(age_data.keys()))
    loclist.household_size_distributions = sorted(list(household_data.keys()))

    if location is not None:
        age_available = location.lower() in [v.lower() for v in loclist.age_distributions]
        hh_available = location.lower() in [v.lower() for v in loclist.household_size_distributions]
        age_sugg = ''
        hh_sugg = ''
        age_sugg = f'(closest match: {sc.suggest(location, loclist.age_distributions)})' if not age_available else ''
        hh_sugg = f'(closest match: {sc.suggest(location, loclist.household_size_distributions)})' if not hh_available else ''
        print(f'For location "{location}":')
        print(f'  Population age distribution is available: {age_available} {age_sugg}')
        print(f'  Household size distribution is available: {hh_available} {hh_sugg}')
        return

    if output:
        return loclist
    else:
        print(f'There are {len(loclist.age_distributions)} age distributions and {len(loclist.household_size_distributions)} household size distributions.')
        print('\nList of available locations (case insensitive):\n')
        sc.pp(loclist)
        return


RASTER_CACHE_DIR = 'data/cache/worldpop'
os.makedirs(RASTER_CACHE_DIR, exist_ok=True)

def download_pop_data(url, out_path, chunk_size=8192):
    """Streams and saves a remote file with progress printouts."""
    if os.path.exists(out_path):
        return out_path
        
    print(f"Downloading: {url}")
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


@functools.lru_cache(maxsize=32)
def get_population_data(
    country_name=None, 
    year=None, 
    admin_level=None
):
    """
    Takes a user-friendly country name (e.g., 'Zambia', 'United States', 'Kenya'),
    converts it to ISO codes, fetches the matching WorldPop raster and GADM shapefile,
    and returns a DataFrame with integer region_codes and probabilities.
    """
    population_data = None

    # 1. Convert user country name to standard ISO-alpha3 (e.g., 'Zambia' -> 'ZMB')
    iso_upper = coco.convert(names=country_name, to='ISO3')
    
    if iso_upper == 'not found':
        raise ValueError(f"Could not resolve country name: '{country_name}'")
        
    iso_lower = iso_upper.lower()

    # Dynamic URLs based on resolved ISO code
    worldpop_url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020/{year}/{iso_upper}/{iso_lower}_ppp_{year}.tif"
    gadm_url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{iso_upper}.gpkg"
    layer_name = f"ADM_ADM_{admin_level}"
    region_name_col = f"NAME_{admin_level}"

    try:
        # 2. Download/Cache WorldPop GeoTIFF & GADM Boundaries
        raster_path = os.path.join(RASTER_CACHE_DIR, f"{iso_lower}_ppp_{year}.tif")
        download_pop_data(worldpop_url, raster_path)

        gadm_path = os.path.join(RASTER_CACHE_DIR, f"gadm41_{iso_upper}.gpkg")
        download_pop_data(gadm_url, gadm_path)

        # 3. Read Regions Boundary file
        regions_gdf = gpd.read_file(gadm_path, layer=layer_name)

        # 4. Extract population per region polygon
        region_names = []
        populations = []

        with rasterio.open(raster_path) as src:
            if regions_gdf.crs != src.crs:
                regions_gdf = regions_gdf.to_crs(src.crs)

            nodata_val = src.nodata if src.nodata is not None else -9999

            for idx, row in regions_gdf.iterrows():
                out_image, _ = mask(src, [row.geometry], crop=True)
                data = out_image[0].astype(float)
                
                data = np.where((data == nodata_val) | (data < 0), np.nan, data)
                pop_sum = float(np.nansum(data))
                
                # Handle boundary naming falls back cleanly
                name = row.get(region_name_col, f"Region_{idx+1}")
                region_names.append(name)
                populations.append(pop_sum)

        # 5. Build output DataFrame with integer region_code & text region_name
        population_data = pd.DataFrame({
            'region_code': range(1, len(region_names) + 1),  # Integer IDs for Covasim
            'region_name': region_names,                    # Real names ('Lusaka', etc.)
            'population': populations
        })

    except Exception as e:
        print(f"WARNING: WorldPop processing failed ({e}). Returning default fallback data.")
        population_data = pd.DataFrame({
            'region_code': [1, 2, 3, 4],
            'region_name': ['Region 1', 'Region 2', 'Region 3', 'Region 4'],
            'population': [1000000, 500000, 750000, 250000]
        })

    # 6. Compute probability
    if population_data is not None:
        total_pop = population_data['population'].sum()
        population_data['probability'] = population_data['population'] / (total_pop if total_pop > 0 else 1.0)
    population_data.to_csv('data/population/population_data.csv', index=False)
    return population_data


def get_age_distribution(location=None):
    '''
    Load age distribution for a given country or countries.

    Args:
        location (str or list): name of the country or countries to load the age distribution for

    Returns:
        age_data (array): Numpy array of age distributions, or dict if multiple locations
    '''

    # Load the raw data
    country_json   = sc.dcp(cad.data)
    state_json     = sc.dcp(sad.data)
    json = sc.mergedicts(state_json, country_json) # Countries will overwrite states, e.g. Georgia
    entries = map_entries(json, location)

    max_age = 99
    result = {}
    for loc,age_distribution in entries.items():
        total_pop = sum(list(age_distribution.values()))
        local_pop = []

        for age, age_pop in age_distribution.items():
            if age[-1] == '+':
                val = [int(age[:-1]), max_age, age_pop/total_pop]
            else:
                ages = age.split('-')
                val = [int(ages[0]), int(ages[1]), age_pop/total_pop]
            local_pop.append(val)
        result[loc] = np.array(local_pop)

    if len(result) == 1:
        result = list(result.values())[0]

    return result


def get_household_size(location=None):
    '''
    Load household size distribution for a given country or countries.

    Args:
        location (str or list): name of the country or countries to load the household size distribution for

    Returns:
        house_size (float): Size of household, or dict if multiple locations
    '''
    # Load the raw data
    json = sc.dcp(hsd.data)

    result = map_entries(json, location)
    if len(result) == 1:
        result = list(result.values())[0]

    return result
