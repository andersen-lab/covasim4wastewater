
import pandas as pd
import json
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.prepared import prep



def load_sewersheds(filename):
    df = pd.read_csv(filename)
    if set(df.columns) != {'sid', 'polygon'}:
        raise ValueError('Sewershed CSV must have exactly the columns sid,polygon')
    polygons = {}
    for line, row in df.iterrows():
        try:
            sid = int(row['sid'])
            if sid < 0 or sid > np.iinfo(np.int32).max or sid in polygons:
                raise ValueError('sid must be unique and between 0 and 2147483647')
            coords = np.asarray(json.loads(row['polygon']), dtype=float)
            if (coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 3
                    or not np.isfinite(coords).all()):
                raise ValueError('polygon must contain at least three finite [x,y] pairs')
            polygon = Polygon(coords)
            if not polygon.is_valid or polygon.is_empty or polygon.area <= 0:
                raise ValueError('polygon must be valid and have positive area')
            for other_sid, other in polygons.items():
                if polygon.intersection(other).area > 0:
                    raise ValueError(f'polygon interiors overlap with sid {other_sid}')
            polygons[sid] = polygon
        except Exception as e:
            raise ValueError(f'Invalid sewershed CSV row {line + 2}: {e}') from e
    if not polygons:
        raise ValueError('Sewershed CSV must contain at least one polygon')
    return polygons

def assign_sewersheds(people, pars):
    coords = pars.get('people_coords')
    if coords is not None:
        coords = np.asarray(coords, dtype=float)
        if coords.shape != (len(people), 2) or not np.isfinite(coords).all():
            raise ValueError('people_coords must have shape (pop_size, 2) with finite x,y values')
        people.x[:] = coords[:, 0]
        people.y[:] = coords[:, 1]

    filename = pars.get('sewershed_file')
    if filename is None:
        people.sewershed[:] = 0
        return

    if not np.isfinite(people.x).all() or not np.isfinite(people.y).all():
        raise ValueError('sewershed_file requires per-person coordinates: supply people_coords '
                         'or x and y arrays in the population. Region IDs do not determine locations.')

    polygons = load_sewersheds(filename)
    assignments = np.full(len(people), -1, dtype=people.sewershed.dtype)

    prepared = [(sid, prep(polygons[sid])) for sid in sorted(polygons)]
    for i, (x, y) in enumerate(zip(people.x, people.y)):
        point = Point(float(x), float(y))
        for sid, polygon in prepared:
            if polygon.covers(point):
                assignments[i] = sid
                break
    people.sewershed[:] = assignments
