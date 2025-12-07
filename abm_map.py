#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_leaflet_map_ship_sat_anim_no_ship_circle_fast_onega.py

Version : force trajectoire visuelle via Onega (pixel interpolation)
 - Attacker path and defender path interpolated in pixel (layer) space
 - Defender launches from Moscow -> Onega (pixel straight), explosion occurs
   exactly where the attacking missile disappears on success.
 - Full filters UI restored.
"""
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict
import math
from typing import Any, Optional, Tuple

# --- CONFIG ---
CSV_PATH = "cleaned_systems.csv"    # adapte si nécessaire
OUT_HTML_PATH = "map_leaflet_ship_sat_anim_no_ship_circle_fast.html"
MIN_RADIUS = 4
BASE_RADIUS = max(1.0, MIN_RADIUS * 0.2)      # ~= MIN_RADIUS/5
DEFAULT_SCALE = 3

# Fallback centroid (middle of Russia)
GLOBAL_CENTROID = (66.417, 94.250)

# Icons (remplace par tes URLs)
SAT_ICON_URL = "https://www.svgrepo.com/download/205676/satellite.svg"
SHIP_ICON_URL = "https://www.svgrepo.com/download/381291/battleship-ship-boat-army-military.svg"
MISSILE_ICON_URL = "https://www.svgrepo.com/download/477346/missile-1.svg"
EXPLOSION_ICON_URL = "https://www.svgrepo.com/download/444803/explosion.svg"
DEFENSE_MISSILE_ICON_URL = "https://www.svgrepo.com/download/193156/rocket-bomb.svg"
DEFENSE_EXPLOSION_ICON_URL = "https://www.svgrepo.com/download/499024/bomb-explosion.svg"

# geocodes file (mapping "Place name" -> [lat, lon])
GEOCODES_PATH = "geocodes.json"

# --- Read CSV ---
df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False).fillna("")
df.columns = [c.strip() for c in df.columns]

expected_cols = [
    "id","system_id","system_name","category","subtype","location_name",
    "latitude","longitude","number","operability","year_operational",
    "range_km","specific","degree","azimuth","route","navy_point","defaut_centroid"
]
for col in expected_cols:
    if col not in df.columns:
        df[col] = ""


def to_bool_like(v):
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1","true","t","yes","y","on"):
        return True
    return False


def to_float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None or x == "" or pd.isna(x):
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None

def to_int_or_none(x: Any) -> Optional[int]:
    try:
        if x is None or x == "" or pd.isna(x):
            return None
        return int(float(x))
    except Exception:
        return None

# normalize numeric-ish columns on the dataframe
df["latitude"] = df["latitude"].apply(to_float_or_none)
df["longitude"] = df["longitude"].apply(to_float_or_none)
df["number"] = df["number"].apply(to_int_or_none)
df["range_km"] = df["range_km"].apply(to_float_or_none)
df["year_operational"] = df["year_operational"].apply(to_int_or_none)
df["degree"] = df["degree"].apply(to_float_or_none)
df["azimuth"] = df["azimuth"].apply(to_float_or_none)

# split rows with coords / without coords
with_coords = df.dropna(subset=["latitude","longitude"]).copy()
without_coords = df[df[["latitude","longitude"]].isnull().any(axis=1)].copy()

# satellites without coords
sat_no_pos = without_coords[without_coords['category'].str.upper().str.contains('SATELLITE', na=False, regex=False)].copy()

# --- read geocodes JSON (missile places) ---
try:
    geocodes_path = Path(GEOCODES_PATH)
    if geocodes_path.exists():
        with geocodes_path.open("r", encoding="utf-8") as fh:
            GEOCODES = json.load(fh)
            # normalize: ensure floats and small rounding
            tmp_gc = {}
            for k, v in GEOCODES.items():
                try:
                    lat = float(v[0]); lon = float(v[1])
                    if math.isfinite(lat) and math.isfinite(lon):
                        tmp_gc[str(k)] = [round(lat, 6), round(lon, 6)]
                except Exception:
                    continue
            GEOCODES = tmp_gc
    else:
        GEOCODES = {}
except Exception:
    GEOCODES = {}

# --- Geo helpers --- (same as before)
def generate_orbit_coords(center_lat=0.0, center_lon=0.0, inclination_deg=20.0, steps=240, phase_deg=0.0):
    pts = []
    for i in range(steps + 1):
        frac = i / steps
        lon = -180.0 + 360.0 * frac
        lon_phase = (lon + phase_deg) % 360.0
        if lon_phase > 180.0:
            lon_phase -= 360.0
        lat = center_lat + inclination_deg * math.sin(math.radians(lon_phase))
        lon_shifted = lon + center_lon
        if lon_shifted > 180.0: lon_shifted -= 360.0
        if lon_shifted < -180.0: lon_shifted += 360.0
        pts.append([round(lat, 6), round(lon_shifted, 6)])
    return pts

def generate_local_circular_orbit(center_lat, center_lon, radius_deg=0.5, steps=180, phase_deg=0.0):
    pts = []
    for i in range(steps + 1):
        ang = (i / steps) * 360.0
        ang = (ang + phase_deg) % 360.0
        theta = math.radians(ang)
        lat = center_lat + radius_deg * math.cos(theta)
        lon = center_lon + (radius_deg * math.sin(theta)) / max(0.0001, math.cos(math.radians(center_lat)))
        if lon > 180.0: lon -= 360.0
        if lon < -180.0: lon += 360.0
        pts.append([round(lat, 6), round(lon, 6)])
    return pts

def interpolate_route_between(p1, p2, steps=40):
    lat1, lon1 = float(p1[0]), float(p1[1])
    lat2, lon2 = float(p2[0]), float(p2[1])
    dlon = lon2 - lon1
    if dlon > 180.0:
        dlon -= 360.0
    elif dlon < -180.0:
        dlon += 360.0
    pts = []
    for i in range(steps + 1):
        t = i / steps
        lat = lat1 + t * (lat2 - lat1)
        lon = lon1 + t * dlon
        if lon > 180.0: lon -= 360.0
        if lon < -180.0: lon += 360.0
        pts.append([round(lat, 6), round(lon, 6)])
    return pts

def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> Optional[float]:
    try:
        lat1 = float(lat1); lon1 = float(lon1); lat2 = float(lat2); lon2 = float(lon2)
        lat1_r = math.radians(lat1)
        lat2_r = math.radians(lat2)
        dlon_r = math.radians(lon2 - lon1)
        x = math.sin(dlon_r) * math.cos(lat2_r)
        y = math.cos(lat1_r)*math.sin(lat2_r) - math.sin(lat1_r)*math.cos(lat2_r)*math.cos(dlon_r)
        if abs(x) < 1e-15 and abs(y) < 1e-15:
            return 0.0
        br = math.degrees(math.atan2(x, y))
        br = (br + 360.0) % 360.0
        if not math.isfinite(br):
            return None
        return br
    except Exception:
        return None

def parse_centroid(raw: str) -> Optional[Tuple[float, float]]:
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "":
        return None
    s = s.replace('"', '').replace("'", '').strip()
    parts = []
    if ',' in s and ';' not in s and ' ' in s and s.count(',') == 1:
        parts = [p.strip() for p in s.split(',') if p.strip() != ""]
    else:
        if ';' in s:
            parts = [p.strip() for p in s.split(';') if p.strip() != ""]
        else:
            parts = [p.strip() for p in s.split() if p.strip() != ""]
    if len(parts) < 2:
        return None
    lat = to_float_or_none(parts[0])
    lon = to_float_or_none(parts[1])
    if lat is None or lon is None:
        return None
    return (lat, lon)

def parse_navy_point(raw: str) -> Optional[list]:
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "":
        return None
    s = s.replace('","', ';').replace('";"', ';')
    s = s.replace('"', '').replace("'", '')
    parts = [p.strip() for p in s.split(';') if p.strip() != ""]
    pts = []
    for part in parts:
        if ',' in part:
            a, b = [q.strip() for q in part.split(',', 1)]
        else:
            parts2 = part.split()
            if len(parts2) < 2:
                continue
            a, b = parts2[0], parts2[1]
        la = to_float_or_none(a)
        lo = to_float_or_none(b)
        if la is None or lo is None:
            continue
        pts.append([round(float(la), 6), round(float(lo), 6)])
    return pts if len(pts) >= 1 else None

def sanitize_for_json(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[str(k)] = sanitize_for_json(v)
        return out
    try:
        if hasattr(obj, "item"):
            return sanitize_for_json(obj.item())
    except Exception:
        pass
    return obj

# --- Build features ---
features = []
coords_by_id = defaultdict(list)
connections = []

for _, row in with_coords.iterrows():
    cat_txt = str(row.get("category", "")).upper()
    is_ship = 'SHIP' in cat_txt

    # normalized values
    row_range_km = to_float_or_none(row.get("range_km"))
    row_degree = to_float_or_none(row.get("degree"))
    row_az = to_float_or_none(row.get("azimuth"))
    row_number = to_int_or_none(row.get("number"))
    row_year = to_int_or_none(row.get("year_operational"))

    props = {
        "id": str(row.get("id", "")),
        "system_id": str(row.get("system_id", "")),
        "system_name": str(row.get("system_name", "")),
        "category": str(row.get("category", "")),
        "subtype": str(row.get("subtype", "")),
        "location_name": str(row.get("location_name", "")),
        "number": (int(row_number) if row_number is not None else None),
        "operability": str(row.get("operability", "")),
        "year_operational": (int(row_year) if row_year is not None else None),
        "range_km": (float(row_range_km) if row_range_km is not None else None),
        "degree": (float(row_degree) if row_degree is not None else None),
        "azimuth": (float(row_az) if row_az is not None else None),
        "specific": str(row.get("specific", "")),
        "viz_range_km": None,
        "viz_degree": None,
        "viz_azimuth": None,
        "is_virtual_satellite": False,
        "is_ship": is_ship,
        "death_zone": to_bool_like(row.get("death_zone")),

    }

    # --- navy_point parsing for ships (new) ---
    raw_navy = row.get("navy_point") if row.get("navy_point") is not None else ""
    parsed_navy = parse_navy_point(raw_navy) if raw_navy else None

    # if parsed_navy present and ship: build orbit accordingly
    if parsed_navy is not None and is_ship:
        if len(parsed_navy) == 1:
            center_lat, center_lon = parsed_navy[0]
            props["orbit"] = generate_local_circular_orbit(center_lat, center_lon, radius_deg=0.5, steps=180, phase_deg=0.0)
            lat = props["orbit"][0][0]; lon = props["orbit"][0][1]
            props["specific"] = (props.get("specific","") + f" [navy_center:{center_lat},{center_lon}]")[:1024]
        else:
            segments = []
            for i in range(len(parsed_navy) - 1):
                p1 = parsed_navy[i]
                p2 = parsed_navy[i + 1]
                seg = interpolate_route_between(p1, p2, steps=40)
                if i > 0:
                    seg = seg[1:]
                segments.extend(seg)
            forward = segments
            back = list(reversed(forward))[1:-1] if len(forward) > 2 else []
            route = forward + back
            props["orbit"] = route
            lat = float(route[0][0]); lon = float(route[0][1])
            props["specific"] = (props.get("specific","") + f" [navy_route:{len(parsed_navy)}pts]")[:1024]
    else:
        lat = to_float_or_none(row.get("latitude"))
        lon = to_float_or_none(row.get("longitude"))
        if lat is None or lon is None:
            lat, lon = GLOBAL_CENTROID
            props['specific'] = (props.get('specific','') + ' [FALLBACK_COORD]')[:1024]

    try:
        lat = float(lat)
        lon = float(lon)
        if not (math.isfinite(lat) and math.isfinite(lon)):
            raise ValueError()
    except Exception:
        lat, lon = GLOBAL_CENTROID
        props['specific'] = (props.get('specific','') + ' [FALLBACK_COORD]')[:1024]

    if 'RADAR' in cat_txt:
        if props.get("range_km") and isinstance(props.get("range_km"), (int, float)) and math.isfinite(props.get("range_km")) and props.get("range_km") > 0:
            props["viz_range_km"] = float(props.get("range_km"))
        else:
            props["viz_range_km"] = 3000.0
        if props.get("degree") and isinstance(props.get("degree"), (int, float)) and math.isfinite(props.get("degree")) and props.get("degree") > 0:
            props["viz_degree"] = float(props.get("degree"))
        else:
            props["viz_degree"] = 30.0
        az_val = props.get("azimuth")
        if az_val is not None and isinstance(az_val, (int, float)) and math.isfinite(az_val):
            try:
                raw_az = float(az_val)
                raw_az = raw_az % 360.0
                if math.isfinite(raw_az):
                    props["viz_azimuth"] = raw_az
                else:
                    props["viz_azimuth"] = None
            except Exception:
                props["viz_azimuth"] = None
        else:
            raw_centroid = row.get("defaut_centroid") if row.get("defaut_centroid") is not None else ""
            parsed = parse_centroid(raw_centroid) if raw_centroid else None
            if parsed is None:
                parsed = GLOBAL_CENTROID
            centroid_lat, centroid_lon = parsed
            b = bearing_deg(centroid_lat, centroid_lon, lat, lon)
            if b is None or not math.isfinite(b):
                try:
                    mean_lat_rad = math.radians((centroid_lat + lat) / 2.0)
                    dy = lat - centroid_lat
                    dx = (lon - centroid_lon) * math.cos(mean_lat_rad)
                    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
                        b2 = 0.0
                    else:
                        theta = math.atan2(dx, dy)
                        b2 = (math.degrees(theta) + 360.0) % 360.0
                    if math.isfinite(b2):
                        props["viz_azimuth"] = float(b2)
                    else:
                        props["viz_azimuth"] = None
                except Exception:
                    props["viz_azimuth"] = None
            else:
                props["viz_azimuth"] = float(b)
    else:
        if props.get("range_km") and isinstance(props.get("range_km"), (int, float)) and math.isfinite(props.get("range_km")) and props.get("range_km") > 0:
            props["viz_range_km"] = float(props.get("range_km"))
        else:
            props["viz_range_km"] = None
        if props.get("degree") and isinstance(props.get("degree"), (int, float)) and math.isfinite(props.get("degree")) and props.get("degree") > 0:
            props["viz_degree"] = float(props.get("degree"))
        else:
            props["viz_degree"] = None
        if props.get("azimuth") is not None and isinstance(props.get("azimuth"), (int, float)) and math.isfinite(props.get("azimuth")):
            props["viz_azimuth"] = float(props.get("azimuth"))
        else:
            props["viz_azimuth"] = None

    if is_ship:
        if "orbit" not in props or not props.get("orbit"):
            radius_deg = 1.0
            try:
                phase = (int(float(row.get("id", 0))) * 37) % 360
            except Exception:
                phase = 0
            props["orbit"] = generate_local_circular_orbit(lat, lon, radius_deg=radius_deg, steps=180, phase_deg=phase)

    if props.get("viz_azimuth") is not None:
        try:
            if not (isinstance(props["viz_azimuth"], (int, float)) and math.isfinite(props["viz_azimuth"])):
                props["viz_azimuth"] = None
        except Exception:
            props["viz_azimuth"] = None

    feat = {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]}, "properties": props}
    features.append(feat)
    coords_by_id[props["id"]].append({"lon": lon, "lat": lat, "props": props})

# synthetic satellites
sat_index = 0
for _, row in sat_no_pos.iterrows():
    props = {
        "id": str(row.get("id", "")),
        "system_id": str(row.get("system_id", "")),
        "system_name": str(row.get("system_name", "")),
        "category": str(row.get("category", "")),
        "subtype": str(row.get("subtype", "")),
        "location_name": str(row.get("location_name", "")),
        "number": (to_int_or_none(row.get("number")) if row.get("number") is not None else None),
        "operability": str(row.get("operability", "")),
        "year_operational": (to_int_or_none(row.get("year_operational")) if row.get("year_operational") is not None else None),
        "range_km": None,
        "degree": (to_float_or_none(row.get("degree")) if row.get("degree") is not None else None),
        "azimuth": (to_float_or_none(row.get("azimuth")) if row.get("azimuth") is not None else None),
        "specific": str(row.get("specific", "")),
        "viz_range_km": None,
        "viz_degree": None,
        "viz_azimuth": None,
        "is_virtual_satellite": True,
        "is_ship": False
    }

    phase = (sat_index * 40.0) % 360.0
    inclination = 10.0 + (sat_index % 4) * 8.0
    orbit_pts = generate_orbit_coords(center_lat=0.0, center_lon=0.0, inclination_deg=inclination, steps=360, phase_deg=phase)
    mid_idx = len(orbit_pts) // 2
    lat_mid, lon_mid = orbit_pts[mid_idx]
    props["orbit"] = orbit_pts
    feat = {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon_mid, lat_mid]}, "properties": props}
    features.append(feat)
    coords_by_id[props["id"]].append({"lon": lon_mid, "lat": lat_mid, "props": props})
    sat_index += 1

# connections
for id_, lst in coords_by_id.items():
    if len(lst) >= 2:
        coords = [[p["lon"], p["lat"]] for p in lst]
        connections.append({"id": id_, "coords": coords, "props": lst[0]["props"]})

geojson = {"type": "FeatureCollection", "features": features}

# hierarchy
hier = {}
for f in features:
    p = f["properties"]
    cat = p.get("category") or ""
    sys = p.get("system_name") or p.get("system_id") or ""
    st = p.get("subtype") or ""
    hier.setdefault(cat, {}).setdefault(sys, set()).add(st)
hier_sorted = {cat: {sys: sorted(list(sts)) for sys, sts in sysmap.items()} for cat, sysmap in hier.items()}

# sanitize and dump JSON strings
def sanitize_for_json_top(o):
    return json.dumps(sanitize_for_json(o), ensure_ascii=False)

GEOJSON_STR = sanitize_for_json_top(geojson)
HIER_STR = sanitize_for_json_top(hier_sorted)
CONNS_STR = sanitize_for_json_top(connections)
GEOCODES_STR = sanitize_for_json_top(GEOCODES)

# --- HTML template (complete) ---
html_template = r"""
<!doctype html>
<html>
<head><meta charset="utf-8"/><title>Leaflet map — ships & satellites anim (missiles, Onega)</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
  html,body,#map{height:100%;margin:0;padding:0}
  .control-panel{position:absolute;top:10px;left:10px;z-index:1000;background:rgba(255,255,255,0.97);padding:10px;border-radius:6px;max-width:520px;font-family:Arial,sans-serif;font-size:13px}
  .filter-block{margin-bottom:8px}
  #log{position:absolute;right:10px;bottom:10px;width:360px;max-height:240px;overflow:auto;background:rgba(0,0,0,0.75);color:#fff;padding:8px;font-family:monospace;font-size:12px;border-radius:6px;z-index:1000}
  .small{font-size:11px;color:#555}
  .modal { position: absolute; left:50%; top:50%; transform:translate(-50%,-50%); z-index:2000; background: white; padding:12px; border-radius:8px; box-shadow:0 6px 30px rgba(0,0,0,0.4); display:none; width:360px; }
  .modal h4{margin:0 0 8px 0;font-size:15px}
  .modal .row{margin:6px 0}
  .modal input[type="text"]{width:100%; padding:6px; box-sizing:border-box}
  .modal .btn{padding:6px 8px; margin-right:6px; cursor:pointer}
</style>
</head>
<body>
<div id="map"></div>
<div class="control-panel" id="controls">
  <h4>Filtres</h4>
  <div class="filter-block"><strong>Category</strong><br/><div id="cat-container"></div></div>
  <div class="filter-block"><strong>System name</strong><br/><div id="system-container"></div></div>
  <div class="filter-block"><strong>Subtype</strong><br/><div id="subtype-container"></div></div>
  <div class="filter-block"><strong>Operability</strong><br/>
    <label><input type="checkbox" class="op" value="operational" checked/> operational</label><br/>
    <label><input type="checkbox" class="op" value="non-operational" checked/> non-operational</label><br/>
    <label><input type="checkbox" class="op" value="unsure" checked/> unsure</label>
  </div>
  <div class="filter-block"><strong>Year</strong><br/>
    <input id="year_min" type="number" placeholder="min" style="width:85px"/> - <input id="year_max" type="number" placeholder="max" style="width:85px"/>
  </div>
  <div class="filter-block"><strong>Taille des marqueurs</strong><br/>
    Multiplier: <input id="scale_mult" type="range" min="1" max="12" value="DEFAULT_SCALE_PLACEHOLDER"/> <span id="scale_val">DEFAULT_SCALE_PLACEHOLDER</span>
  </div>
  <div class="filter-block"><strong>Opacité portée</strong><br/>
    Opacité: <input id="range_op" type="range" min="0" max="1" step="0.05" value="0.25"/> <span id="op_val">0.25</span>
  </div>
    <div class="filter-block">
    <strong>Affichages</strong><br/>
    <label><input id="show_connections" type="checkbox" checked/> montrer les lignes id↔id</label><br/>
    <label><input id="show_orbits" type="checkbox" checked/> montrer les orbites satellites (virtuelles)</label><br/>
    <label><input id="animate_sat" type="checkbox" checked/> animer les satellites & ships</label><br/>
    <label><input id="show_all_ranges" type="checkbox" /> afficher toutes les portées RADAR</label><br/>
  </div>


  <!-- missile place button -->
  <div class="filter-block">
    <button id="place_missile_btn">Place ballistic missile</button>
    <div class="small">Choisir un lieu pré-géocodé</div>
  </div>

  <div style="margin-top:6px;"><button id="reset">Réinitialiser</button></div>
</div>

<!-- modal: place missile -->
<div id="missile_modal" class="modal" role="dialog" aria-modal="true" aria-hidden="true">
  <h4>Place ballistic missile</h4>
  <div class="row">
    <label for="missile_place_input">Lieu (autocomplétion):</label>
    <input id="missile_place_input" list="geo_datalist" placeholder="start typing..."/>
    <datalist id="geo_datalist"></datalist>
  </div>
  <div class="row">
    <button id="missile_place_confirm" class="btn">Place</button>
    <button id="missile_place_cancel" class="btn">Cancel</button>
  </div>
</div>

<!-- modal: launch missile -->
<div id="launch_modal" class="modal" role="dialog" aria-modal="true" aria-hidden="true">
  <h4>Launch missile - choose target</h4>
  <div class="row">
    <label for="launch_target_input">Target (autocomplétion):</label>
    <input id="launch_target_input" list="geo_datalist" placeholder="choose a target..."/>
  </div>
  <div class="row">
    <label for="launch_count">Count:</label>
    <input id="launch_count" type="number" min="1" max="50" value="1" style="width:80px;"/> missiles
    &nbsp; <label for="launch_interval_ms">Interval (ms):</label>
    <input id="launch_interval_ms" type="number" min="0" max="10000" value="1000" style="width:100px;"/> ms
  </div>
  <div class="row">
    <label for="flight_speed">Flight speed (visual):</label>
    <input id="flight_speed" type="range" min="20" max="500" step="10" value="180"/> <span id="flight_speed_val">180</span> ms per tick
  </div>
  <div class="row">
    <button id="launch_confirm" class="btn">Launch</button>
    <button id="launch_cancel" class="btn">Cancel</button>
  </div>
</div>

<div id="log" aria-live="polite"></div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
/* embedded JSONs */
const GEOJSON = GEOJSON_PLACEHOLDER;
const HIER = HIER_PLACEHOLDER;
const CONNECTIONS = CONNECTIONS_PLACEHOLDER;
const GEOCODES = GEOCODES_PLACEHOLDER;
const SAT_ICON_URL = "docs/icons/satellite.svg";
const SHIP_ICON_URL = "docs/icons/ship.svg";
const MISSILE_ICON_URL = "docs/icons/missile.svg";
const EXPLOSION_ICON_URL = "docs/icons/explosion.svg";
const DEFENSE_MISSILE_ICON_URL = "docs/icons/def_missile.svg";
const DEFENSE_EXPLOSION_ICON_URL = "docs/icons/def_expl.svg";
const BASE_RADIUS = BASE_RADIUS_PLACEHOLDER;
const MIN_RADIUS = MIN_RADIUS_PLACEHOLDER;
const DEFAULT_SCALE = DEFAULT_SCALE_PLACEHOLDER;

/* FORCE POINT: Onega mouth (visual intercept point) */
const ONEGA_LAT = 85.9166667;
const ONEGA_LON = 38.0833333;

/* Other config */
const SUCCESS_PROB_SMALL = 0.5;
const SUCCESS_PROB_LARGE = 0.1;

function log(msg){ const b=document.getElementById('log'); const d=document.createElement('div'); d.textContent = new Date().toISOString() + " - " + msg; b.prepend(d); }
function safeText(s){ return (s===null||s===undefined) ? '' : String(s); }

/* Compute great-circle-ish sector points (used previously) */
function computeSectorLatLngs(lat, lon, radiusMeters, startDeg, endDeg, steps){
  const R = 6371000.0;
  const lat1 = lat * Math.PI/180.0;
  const lon1 = lon * Math.PI/180.0;
  let s = startDeg % 360; if(s < 0) s += 360;
  let e = endDeg % 360; if(e < 0) e += 360;
  let total = (e >= s) ? (e - s) : (360 - (s - e));
  const stepAngle = total / Math.max(1, steps);
  const pts = [];
  const delta = radiusMeters / R;
  for(let i=0;i<=steps;i++){
    const ang = (s + i*stepAngle) % 360;
    const theta = ang * Math.PI/180.0;
    const sinLat2 = Math.sin(lat1)*Math.cos(delta) + Math.cos(lat1)*Math.sin(delta)*Math.cos(theta);
    const lat2 = Math.asin(sinLat2);
    const y = Math.sin(theta)*Math.sin(delta)*Math.cos(lat1);
    const x = Math.cos(delta) - Math.sin(lat1)*Math.sin(lat2);
    const lon2 = lon1 + Math.atan2(y, x);
    pts.push([lat2 * 180.0/Math.PI, lon2 * 180.0/Math.PI]);
  }
  return pts;
}

/* build basic controls */
function buildCategoryControls(){
  const catContainer = document.getElementById('cat-container');
  catContainer.innerHTML = '';
  Object.keys(HIER).sort().forEach(cat => {
    const div = document.createElement('div');
    div.innerHTML = `<label><input type="checkbox" class="cat" value="${cat}" checked/> ${cat || '(no category)'}</label>`;
    catContainer.appendChild(div);
  });
}
function getSelectedCategories(){ return Array.from(document.querySelectorAll('input.cat:checked')).map(n=>n.value); }
function buildSystemControls(){
  const sysContainer = document.getElementById('system-container'); sysContainer.innerHTML = '';
  const selCats = getSelectedCategories();
  selCats.forEach(cat => {
    const systems = Object.keys(HIER[cat] || {}).sort();
    if(systems.length === 0) return;
    const details = document.createElement('details'); details.open = true;
    const summary = document.createElement('summary'); summary.textContent = cat + " → systems";
    details.appendChild(summary);
    const div = document.createElement('div'); div.style.marginLeft = '12px';
    systems.forEach(sys => {
      const wrapper = document.createElement('div');
      wrapper.innerHTML = `<label><input type="checkbox" class="system_name" data-category="${cat}" value="${sys}" checked/> ${sys}</label>`;
      div.appendChild(wrapper);
    });
    details.appendChild(div);
    sysContainer.appendChild(details);
  });
}
function buildSubtypeControls(){
  const subContainer = document.getElementById('subtype-container'); subContainer.innerHTML = '';
  const selSystems = Array.from(document.querySelectorAll('input.system_name:checked')).map(n => ({category:n.dataset.category, system:n.value}));
  const subtypeSet = new Set();
  selSystems.forEach(o => {
    const arr = (HIER[o.category] && HIER[o.category][o.system]) ? HIER[o.category][o.system] : [];
    arr.forEach(s => { if(s !== '') subtypeSet.add(s); });
  });
  const arr = Array.from(subtypeSet).sort();
  if(arr.length === 0){ subContainer.innerHTML = "<div class='small'>(aucun subtype trouvé)</div>"; return; }
  arr.forEach(s => {
    const wrapper = document.createElement('div');
    wrapper.innerHTML = `<label><input type="checkbox" class="subtype" value="${s}" checked/> ${s}</label>`;
    subContainer.appendChild(wrapper);
  });
}

/* small helper to find Moscow coords in GEOCODES (fallback to fixed coord) */
function findMoscowCoords(){
  const keys = Object.keys(GEOCODES || {});
  for (let k of keys){
    if (/moscow/i.test(k) || /moskva/i.test(k) || /mosk\./i.test(k)) return GEOCODES[k];
  }
  if (GEOCODES["Moscow"]) return GEOCODES["Moscow"];
  return [55.7558, 37.6173];
}

/* robust check: does the geocode name indicate Russia? */
function geocodeIsRussia(name){
  if(!name) return false;
  try {
    const s = String(name).toLowerCase();
    if (s.indexOf('russia') !== -1) return true;
    if (s.indexOf('россия') !== -1) return true;
    return false;
  } catch (e) { return false; }
}

function geocodeIsThreatGroup(name){
  if(!name) return false;
  try {
    const s = String(name).toLowerCase();
    if (s.indexOf('russia') !== -1 || s.indexOf('россия') !== -1) return true;
    if (s.indexOf('china') !== -1 || s.indexOf('prc') !== -1) return true;
    if (s.indexOf('north korea') !== -1 || s.indexOf('dprk') !== -1 || s.indexOf('korea, north') !== -1) return true;
    return false;
  } catch(e) { return false; }
}


document.addEventListener('DOMContentLoaded', function(){
  buildCategoryControls(); buildSystemControls(); buildSubtypeControls();

  document.getElementById('cat-container').addEventListener('change', function(e){ if(e.target && e.target.classList && e.target.classList.contains('cat')){ buildSystemControls(); buildSubtypeControls(); applyFilters(); }});
  document.getElementById('system-container').addEventListener('change', function(e){ if(e.target && e.target.classList && e.target.classList.contains('system_name')){ buildSubtypeControls(); applyFilters(); }});
  document.getElementById('subtype-container').addEventListener('change', function(e){ if(e.target && e.target.classList && e.target.classList.contains('subtype')) applyFilters(); });
  document.querySelectorAll('input.op').forEach(el => el.addEventListener('change', applyFilters));
  document.getElementById('year_min').addEventListener('input', applyFilters);
  document.getElementById('year_max').addEventListener('input', applyFilters);
  document.getElementById('show_connections').addEventListener('change', applyFilters);
  document.getElementById('show_orbits').addEventListener('change', applyFilters);
  document.getElementById('animate_sat').addEventListener('change', applyFilters);
  document.getElementById('show_all_ranges').addEventListener('change', applyFilters);

  const scaleInput = document.getElementById('scale_mult');
  const scaleVal = document.getElementById('scale_val');
  scaleInput.addEventListener('input', function(e){ scaleVal.textContent = e.target.value; applyFilters();});

  const opInput = document.getElementById('range_op');
  const opVal = document.getElementById('op_val');
  opInput.addEventListener('input', function(e){ opVal.textContent = e.target.value; applyFilters();});

  document.getElementById('reset').addEventListener('click', function(){
    document.querySelectorAll('#controls input[type=checkbox]').forEach(c => c.checked = true);
    document.getElementById('year_min').value = ''; document.getElementById('year_max').value = '';
    document.getElementById('scale_mult').value = DEFAULT_SCALE; document.getElementById('scale_val').textContent = DEFAULT_SCALE;
    document.getElementById('range_op').value = 0.25; document.getElementById('op_val').textContent = '0.25';
    document.getElementById('show_connections').checked = true; document.getElementById('show_orbits').checked = true; document.getElementById('animate_sat').checked = true;
    document.getElementById('show_all_ranges').checked = false;

    buildSystemControls(); buildSubtypeControls(); applyFilters();
  });

  // Map + panes
  const map = L.map('map').setView([55,45], 3);
  map.createPane('connectionsPane'); map.getPane('connectionsPane').style.zIndex = 350;
  map.createPane('orbitsPane'); map.getPane('orbitsPane').style.zIndex = 375;
  map.createPane('rangesPane'); map.getPane('rangesPane').style.zIndex = 410;
  map.createPane('markersPane'); map.getPane('markersPane').style.zIndex = 650;
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 18 }).addTo(map);

  const connectionsLayer = L.layerGroup([], { pane: 'connectionsPane' }).addTo(map);
  const orbitsLayer = L.layerGroup([], { pane: 'orbitsPane' }).addTo(map);
  const tempRangeLayer = L.layerGroup([], { pane: 'rangesPane' }).addTo(map);
  const persistentRangeLayer = L.layerGroup([], { pane: 'rangesPane' }).addTo(map);
  const markersLayer = L.layerGroup([], { pane: 'markersPane' }).addTo(map);

  // missile layer (persistent)
  const missileLayer = L.layerGroup([], { pane: 'markersPane' }).addTo(map);

  // ABM ranges layer (always visible)
  const abmRangeLayer = L.layerGroup([], { pane: 'rangesPane' }).addTo(map);
  const allRangesLayer = L.layerGroup([], { pane: 'rangesPane' }).addTo(map);

  let persistentFeatureId = null;
  const features = GEOJSON.features || [];
  log('Features total: ' + features.length + '; geocodes: ' + Object.keys(GEOCODES).length);

  // icons
  const SatIcon = L.Icon.extend({ options: { iconSize:[32,32], iconAnchor:[16,16], popupAnchor:[0,-14] }});
  const satIconInstance = new SatIcon({ iconUrl: SAT_ICON_URL });
  const ShipIcon = L.Icon.extend({ options: { iconSize:[36,36], iconAnchor:[18,18], popupAnchor:[0,-14] }});
  const shipIconInstance = new ShipIcon({ iconUrl: SHIP_ICON_URL });

  // missile + explosion icons
  const MissileIcon = L.Icon.extend({ options: { iconSize:[40,40], iconAnchor:[20,20], popupAnchor:[0,-18] }});
  const missileIconInstance = new MissileIcon({ iconUrl: MISSILE_ICON_URL });
  const ExplosionIcon = L.Icon.extend({ options: { iconSize:[96,96], iconAnchor:[48,48], popupAnchor:[0,-40] }});
  const explosionIconInstance = new ExplosionIcon({ iconUrl: EXPLOSION_ICON_URL });

  function makeImgDivIcon(url, sizePx, fallbackEmoji="💥", imgStyle="width:100%;height:100%;object-fit:contain;") {
    const html = `
      <div style="width:${sizePx}px;height:${sizePx}px;display:flex;align-items:center;justify-content:center;">
        <img src="${url}" style="${imgStyle}" crossorigin="anonymous"
             onerror="this.style.display='none';this.parentNode.innerHTML='<div style=\\'font-size:${Math.max(12, Math.round(sizePx*0.6))}px\\'>${fallbackEmoji}</div>';" />
      </div>`;
    return L.divIcon({ className: 'img-divicon', html: html, iconSize: [sizePx, sizePx], iconAnchor: [Math.round(sizePx/2), Math.round(sizePx/2)], popupAnchor: [0, -Math.round(sizePx/2)] });
  }

  const defenseMissileDivIcon = makeImgDivIcon(DEFENSE_MISSILE_ICON_URL, 36, "🚀");
  const defenseExplosionDivIcon = makeImgDivIcon(DEFENSE_EXPLOSION_ICON_URL, 80, "💥");

  // counters / timers
  let activeAttackingMissiles = 0;
  const scheduledTimers = [];

  // dessine/rafraîchit toutes les portées ABM depuis GEOJSON.features
  function drawABMRanges(){
    try {
      abmRangeLayer.clearLayers();
      if(!GEOJSON || !Array.isArray(GEOJSON.features)) return;
      (GEOJSON.features || []).forEach(f => {
        try {
          const p = f.properties || {};
          const cat = (p.category || '').toString().toUpperCase();
          if(cat.indexOf('ABM') === -1) return;
          const rkm = (p.viz_range_km !== null && p.viz_range_km !== undefined) ? Number(p.viz_range_km) : (p.range_km ? Number(p.range_km) : null);
          if(!rkm || isNaN(rkm)) return;
          const lat = Number(f.geometry.coordinates[1]);
          const lon = Number(f.geometry.coordinates[0]);
          const circle = L.circle([lat, lon], { radius: rkm*1000.0, color: '#d73027', fill: false, weight: 1.2, pane: 'rangesPane', interactive:false });
          abmRangeLayer.addLayer(circle);
        } catch(e){ /* ignore this feature */ }
      });
    } catch(e){ /* ignore whole function errors */ }
  }

  // populate datalist for geocodes
  (function populateGeocodeDatalist(){
    const dl = document.getElementById('geo_datalist');
    try{
      Object.keys(GEOCODES || {}).sort().forEach(name => {
        const opt = document.createElement('option');
        opt.value = name;
        dl.appendChild(opt);
      });
    }catch(e){}
  })();

  // missile place modal behavior
  document.getElementById('place_missile_btn').addEventListener('click', function(){
    const modal = document.getElementById('missile_modal');
    modal.style.display = 'block';
    modal.setAttribute('aria-hidden', 'false');
    document.getElementById('missile_place_input').value = '';
    document.getElementById('missile_place_input').focus();
  });
  document.getElementById('missile_place_cancel').addEventListener('click', function(){
    const modal = document.getElementById('missile_modal');
    modal.style.display = 'none';
    modal.setAttribute('aria-hidden', 'true');
  });

  // helper: find geocode by exact or partial match
  function findGeocodeByName(q){
    if(!q) return null;
    if(GEOCODES[q]) return [q, GEOCODES[q]];
    const qlow = q.toLowerCase();
    const keys = Object.keys(GEOCODES || {});
    for(let k of keys){
      if(k.toLowerCase() === qlow) return [k, GEOCODES[k]];
    }
    for(let k of keys){
      if(k.toLowerCase().indexOf(qlow) !== -1) return [k, GEOCODES[k]];
    }
    return null;
  }

  // create missile via modal
  document.getElementById('missile_place_confirm').addEventListener('click', function(){
    const q = document.getElementById('missile_place_input').value.trim();
    const found = findGeocodeByName(q);
    if(!found){
      alert('Lieu non trouvé dans les géocodes.');
      return;
    }
    const name = found[0];
    const coords = found[1];
    const lat = Number(coords[0]), lon = Number(coords[1]);
    // unique id for buttons
    const uid = 'm' + Date.now() + '_' + Math.round(Math.random()*10000);
    const popupHtml = `<strong>Missile</strong><br/>${safeText(name)}<br/>
      <button id="launch_${uid}">Launch</button> <button id="remove_${uid}">Remove</button>`;
    const m = L.marker([lat, lon], { icon: missileIconInstance, pane: 'markersPane' });
    m._uid = uid;
    m._place_name = name;
    m.bindPopup(popupHtml, { maxWidth: 320 });
    missileLayer.addLayer(m);
    map.setView([lat, lon], Math.max(5, Math.min(10, map.getZoom()+2)));
    // popup open handler to attach buttons
    m.on('popupopen', function(ev){
      // attach handlers
      setTimeout(()=>{
        try{
          const launchBtn = document.getElementById(`launch_${m._uid}`);
          const removeBtn = document.getElementById(`remove_${m._uid}`);
          if(launchBtn){
            launchBtn.addEventListener('click', function(){
              // open launch modal and remember missile marker
              currentLaunchMissile = m;
              const lm = document.getElementById('launch_modal');
              lm.style.display = 'block';
              lm.setAttribute('aria-hidden','false');
              document.getElementById('launch_target_input').value = '';
              document.getElementById('flight_speed').value = 180;
              document.getElementById('flight_speed_val').textContent = '180';
              document.getElementById('launch_target_input').focus();
            });
          }
          if(removeBtn){
            removeBtn.addEventListener('click', function(){
              missileLayer.removeLayer(m);
              map.closePopup();
            });
          }
        }catch(e){}
      }, 50);
    });
    // close modal
    const modal = document.getElementById('missile_modal');
    modal.style.display = 'none';
    modal.setAttribute('aria-hidden', 'true');
  });

  // launch modal interactions
  let currentLaunchMissile = null;
  document.getElementById('launch_cancel').addEventListener('click', function(){
    const lm = document.getElementById('launch_modal');
    lm.style.display = 'none';
    lm.setAttribute('aria-hidden','true');
    currentLaunchMissile = null;
  });
  const flightSpeedInput = document.getElementById('flight_speed');
  flightSpeedInput.addEventListener('input', function(e){
    document.getElementById('flight_speed_val').textContent = e.target.value;
  });

  // helper: linear interpolate between two lon that might cross dateline (still used for lat/lon fallback)
  function interpLatLon(p1, p2, t){
    let lat = p1[0] + t*(p2[0]-p1[0]);
    let lon1 = p1[1], lon2 = p2[1];
    let dlon = lon2 - lon1;
    if(dlon > 180) dlon -= 360;
    if(dlon < -180) dlon += 360;
    let lon = lon1 + t*dlon;
    if(lon > 180) lon -= 360;
    if(lon < -180) lon += 360;
    return [lat, lon];
  }

  // launch confirm: schedule N launches staggered by intervalMs
  document.getElementById('launch_confirm').addEventListener('click', function(){
    const q = document.getElementById('launch_target_input').value.trim();
    const found = findGeocodeByName(q);
    if(!found){
      alert('Target not found in geocodes.');
      return;
    }
    if(!currentLaunchMissile){
      alert('No missile selected (open its popup and click Launch).');
      const lm = document.getElementById('launch_modal');
      lm.style.display = 'none';
      lm.setAttribute('aria-hidden','true');
      return;
    }

    // Read parameters
    const targetName = found[0];
    const targetCoords = found[1];
    const startLatLng = currentLaunchMissile.getLatLng();
    const start = [startLatLng.lat, startLatLng.lng];
    const target = [Number(targetCoords[0]), Number(targetCoords[1])];

    const count = Math.max(1, parseInt(document.getElementById('launch_count').value) || 1);
    const intervalMs = Math.max(0, parseInt(document.getElementById('launch_interval_ms').value) || 1000);
    const baseInterval = Math.max(20, Math.min(300, parseInt(document.getElementById('flight_speed').value) || 120));

    // compute pixel distance for apex scaling
    const startPt = map.latLngToLayerPoint(L.latLng(start[0], start[1]));
    const targetPt = map.latLngToLayerPoint(L.latLng(target[0], target[1]));
    const pixelDist = startPt.distanceTo(targetPt);
    const apexPixelsBase = Math.max(12, Math.min(400, Math.round(pixelDist * 0.08)));
    const stepsBase = Math.max(30, Math.min(600, Math.round(pixelDist / 2)));

    // Close modal IMMEDIATELY so it doesn't stay open
    const lm = document.getElementById('launch_modal');
    lm.style.display = 'none';
    lm.setAttribute('aria-hidden','true');

    // schedule N launches staggered by intervalMs
    for(let mIndex = 0; mIndex < count; mIndex++){
      const delay = mIndex * intervalMs;
      const timer = setTimeout(() => {
        // for each missile: compute its own pts and animate
        launchAttackingMissile(start, target, {
          steps: stepsBase,
          baseInterval: baseInterval,
          apexPixels: apexPixelsBase,
          targetName: targetName
        });
      }, delay);
      scheduledTimers.push(timer);
    }

    // clear reference to currentLaunchMissile (we consumed it)
    currentLaunchMissile = null;
  });

  // -----------------------
  // NEW: launchAttackingMissile using pixel interpolation so the trajectory passes visually through ONEGA
  // -----------------------
  // Replace your existing launchAttackingMissile with this one
function launchAttackingMissile(start, target, opts) {
  const steps = opts.steps || 120;
  const baseInterval = opts.baseInterval || 120; // ms per step
  const apexPixels = opts.apexPixels || 80;
  const targetName = opts.targetName || '';
  const targetIsThreat = geocodeIsThreatGroup(targetName);

  // Precise Onega lat/lon (injected exactly)
  const onegaLatLon = [ONEGA_LAT, ONEGA_LON];

  // Prepare layer points for start/target/onega
  const startLatLng = L.latLng(start[0], start[1]);
  const targetLatLng = L.latLng(target[0], target[1]);
  const startLayer = map.latLngToLayerPoint(startLatLng);
  const targetLayer = map.latLngToLayerPoint(targetLatLng);
  const onegaLayer = map.latLngToLayerPoint(L.latLng(onegaLatLon[0], onegaLatLon[1]));

  const pts = []; // entries: { pos: [lat,lon], layerPt: L.Point, idx: integer }

  if (!targetIsRussia) {
    // single segment in pixel space
    for (let k = 0; k <= steps; k++) {
      const t = k / steps;
      const lx = startLayer.x + t * (targetLayer.x - startLayer.x);
      const ly = startLayer.y + t * (targetLayer.y - startLayer.y);
      const latlng = map.layerPointToLatLng(L.point(lx, ly));
      pts.push({ pos: [latlng.lat, latlng.lng], layerPt: L.point(lx, ly), idx: pts.length });
    }
      } else {
      // Force path start -> ONEGA -> target (pixel interpolation to guarantee visual crossing)
      // We split the total `steps` in two segments, ensuring ONEGA is exactly included as the join point.
      const minSeg = 6;
      const steps1 = Math.max(minSeg, Math.floor(steps / 2)); // start -> onega
      const steps2 = Math.max(minSeg, steps - steps1);         // onega -> target

      // start -> ONEGA
      for (let k = 0; k <= steps1; k++) {
        const tseg = steps1 === 0 ? 0 : (k / steps1);
        const lx = startLayer.x + tseg * (onegaLayer.x - startLayer.x);
        const ly = startLayer.y + tseg * (onegaLayer.y - startLayer.y);
        const latlng = map.layerPointToLatLng(L.point(lx, ly));
        const overallIdx = k; // 0 .. steps1
        pts.push({
          pos: [latlng.lat, latlng.lng],
          layerPt: L.point(lx, ly),
          t: (overallIdx / steps),
          idx: overallIdx
        });
      }

      // ONEGA -> target
      // start from k = 1 so we don't duplicate the exact ONEGA point
      for (let k = 1; k <= steps2; k++) {
        const tseg = steps2 === 0 ? 0 : (k / steps2);
        const lx = onegaLayer.x + tseg * (targetLayer.x - onegaLayer.x);
        const ly = onegaLayer.y + tseg * (targetLayer.y - onegaLayer.y);
        const latlng = map.layerPointToLatLng(L.point(lx, ly));
        const overallIdx = steps1 + k; // steps1+1 .. steps1+steps2
        pts.push({
          pos: [latlng.lat, latlng.lng],
          layerPt: L.point(lx, ly),
          t: (overallIdx / steps),
          idx: overallIdx
        });
      }

      // Safety: if numeric rounding left us short/long, trim or extend to match ~steps+1 length
      // (optional but keeps downstream logic robust)
      if (pts.length > steps + 1) {
        pts.length = steps + 1;
      } else {
        while (pts.length < steps + 1) {
          // push exact target as last point (guarantee landing)
          pts.push({ pos: [targetLatLng.lat, targetLatLng.lng], layerPt: targetLayer, t: 1.0, idx: pts.length });
        }
      }
    }


  // Create attacker marker
  const htmlImg = `<img src="${MISSILE_ICON_URL}" style="width:40px;height:40px;display:block;transform:translate3d(-50%,-50%,0)"/>`;
  const flyingIcon = L.divIcon({ className: 'flying-missile-divicon', html: htmlImg, iconSize: [40, 40], iconAnchor: [20, 20] });
  const flying = L.marker([pts[0].pos[0], pts[0].pos[1]], { icon: flyingIcon, pane: 'markersPane', zIndexOffset: 2000 }).addTo(map);

  activeAttackingMissiles += 1;
  let attackAborted = false;

  // Determine interceptIdx: for Russian targets must be EXACT index at ONEGA (by lat/lon)
  let interceptIdx = -1;
  if (targetIsRussia) {
    // find the index where pos exactly equals ONEGA lat/lon (we pushed it exactly), fallback to nearest by layer distance
    for (let j = 0; j < pts.length; j++) {
      const p = pts[j].pos;
      if (Math.abs(p[0] - ONEGA_LAT) < 1e-9 && Math.abs(p[1] - ONEGA_LON) < 1e-9) { interceptIdx = j; break; }
    }
    if (interceptIdx < 0) {
      // fallback: nearest in pixel space to onegaLayer
      let bestD = Infinity;
      for (let j = 0; j < pts.length; j++) {
        const d = pts[j].layerPt.distanceTo(onegaLayer);
        if (d < bestD) { bestD = d; interceptIdx = j; }
      }
    }
  } else {
    // non-Russia: choose nearest to Moscow column (as before)
    const moscowCoords = findMoscowCoords();
    const moscowLayer = map.latLngToLayerPoint(L.latLng(Number(moscowCoords[0]), Number(moscowCoords[1])));
    let bestDx = Infinity;
    for (let j = 0; j < pts.length; j++) {
      const dx = Math.abs(pts[j].layerPt.x - moscowLayer.x);
      if (dx < bestDx) { bestDx = dx; interceptIdx = j; }
    }
  }
  if (interceptIdx < 0) interceptIdx = Math.floor(pts.length / 2);

  // attack timing
  const attackStartTime = performance.now();
  const attackArrivalTime = attackStartTime + interceptIdx * baseInterval;

  // defender target — ensure we use the exact ONEGA position when targetIsRussia
  const defTargetLayer = (targetIsRussia ? onegaLayer : pts[interceptIdx].layerPt);
  const defTargetLatLng = map.layerPointToLatLng(defTargetLayer);
  const defTarget = [defTargetLatLng.lat, defTargetLatLng.lng];

  // defender start (Moscow)
  const moscowCoords = findMoscowCoords();
  const moscowLat = Number(moscowCoords[0]);
  const moscowLon = Number(moscowCoords[1]);
  const moscowLayer = map.latLngToLayerPoint(L.latLng(moscowLat, moscowLon));

  // compute defender travel and start offset so arrival == attackArrivalTime
  const dpx = moscowLayer.distanceTo(defTargetLayer);
  const DEFENDER_SPEED_PX_PER_MS = 0.5; // adjust if you want faster defender
  let travelMs = Math.max(12, Math.round(dpx / DEFENDER_SPEED_PX_PER_MS));
  travelMs = Math.max(12, Math.min(travelMs, 20000));
  let defenderStartTime = Math.round(attackArrivalTime - travelMs);
  let defenderStartOffset = Math.max(0, defenderStartTime - attackStartTime);

  log("Launch: targetName='" + targetName + "' russia=" + targetIsRussia + " pts=" + pts.length +
      " interceptIdx=" + interceptIdx + " attackOffset=" + Math.round(attackArrivalTime - attackStartTime) +
      " travelMs=" + travelMs + " defenderStartOffset=" + defenderStartOffset);

  // defender marker
  let defMarker = null;
  let defenderStarted = false;

  // decide success
  const successProb = (activeAttackingMissiles <= 10) ? SUCCESS_PROB_SMALL : SUCCESS_PROB_LARGE;
  const success = (Math.random() < successProb);

  // animation loop anchored on attackStartTime
  function tick() {
    const now = performance.now();
    const elapsed = now - attackStartTime;

    // attacker
    if (!attackAborted) {
      const posf = elapsed / baseInterval;
      if (posf >= pts.length - 1) {
        try { map.removeLayer(flying); } catch (e) {}
        if (!attackAborted) {
          const explosion = L.marker([target[0], target[1]], { icon: explosionIconInstance, pane: 'markersPane', zIndexOffset: 3000 }).addTo(map);
          explosion.bindPopup(`<strong>Impact</strong><br/>Target: ${safeText(targetName)}`).openPopup();
          setTimeout(()=>{ try{ map.removeLayer(explosion); }catch(e){} }, 3000);
        }
        activeAttackingMissiles = Math.max(0, activeAttackingMissiles - 1);
        return;
      } else {
        const i0 = Math.floor(posf);
        const i1 = Math.min(pts.length - 1, i0 + 1);
        const frac = posf - i0;
        // Interpolate in pixel space for smooth straight motion
        const p0 = pts[i0].layerPt;
        const p1 = pts[i1].layerPt;
        const lx = p0.x + frac * (p1.x - p0.x);
        const ly = p0.y + frac * (p1.y - p0.y);
        const latlng = map.layerPointToLatLng(L.point(lx, ly));
        try { flying.setLatLng([latlng.lat, latlng.lng]); } catch(e){}
        // visual parabola
        try {
          const normIdx = (i0 + frac) / pts.length;
          const pxnorm = 4 * normIdx * (1 - normIdx);
          const px = apexPixels * pxnorm;
          if (flying && flying._icon) {
            const img = flying._icon.querySelector && (flying._icon.querySelector('img') || flying._icon.querySelector('svg'));
            if (img) {
              img.style.transition = `transform ${Math.max(16, Math.round(baseInterval))}ms linear`;
              const scaleFactor = 1.0 + 0.0009 * px;
              img.style.transform = `translate3d(-50%,-50%,0) translateY(${-px}px) scale(${scaleFactor})`;
            }
          }
        } catch(e){}
      }
    }

    // defender start & animate
        // Defender start (only for Russian targets)
    if (targetIsRussia && !defenderStarted) {
      const elapsedSinceAttackStart = now - attackStartTime;
      if (elapsedSinceAttackStart >= defenderStartOffset) {
        defenderStarted = true;
        defMarker = L.marker([moscowLat, moscowLon], { icon: defenseMissileDivIcon, pane: 'markersPane', zIndexOffset: 2100 }).addTo(map);
        const defStartTime = performance.now();
        const defEndTime = defStartTime + travelMs;
        (function animateDef(){
          const nn = performance.now();
          const frac = Math.min(1.0, Math.max(0.0, (nn - defStartTime) / travelMs));
          const lx = moscowLayer.x + frac * (defTargetLayer.x - moscowLayer.x);
          const ly = moscowLayer.y + frac * (defTargetLayer.y - moscowLayer.y);
          const latlng = map.layerPointToLatLng(L.point(lx, ly));
          try { defMarker.setLatLng([latlng.lat, latlng.lng]); } catch(e){}
          if (frac < 1.0) requestAnimationFrame(animateDef);
          else setTimeout(()=>{ try{ map.removeLayer(defMarker); }catch(e){} }, 900);
        })();

        // schedule explosion exactly at attackArrivalTime if success
        if (success) {
          const delayUntilExplosion = Math.max(0, (attackArrivalTime - performance.now()));
          setTimeout(() => {
            if (attackAborted) return;
            let explLatLng;
            if (targetIsRussia) {
              explLatLng = { lat: onegaLatLon[0], lng: onegaLatLon[1] };
            } else {
              const atkLayer = pts[interceptIdx].layerPt;
              const atkLL = map.layerPointToLatLng(atkLayer);
              explLatLng = { lat: atkLL.lat, lng: atkLL.lng };
            }
            try { map.removeLayer(flying); } catch(e){}
            const expl = L.marker([explLatLng.lat, explLatLng.lng], { icon: defenseExplosionDivIcon, pane: 'markersPane', zIndexOffset: 3000 }).addTo(map);
            setTimeout(()=>{ try{ map.removeLayer(expl); }catch(e){} }, 2500);
            try { if (defMarker) map.removeLayer(defMarker); } catch(e){}
            attackAborted = true;
            activeAttackingMissiles = Math.max(0, activeAttackingMissiles - 1);
            log("Interception SUCCESS at forced Onega/fixed pixel position.");
          }, delayUntilExplosion);
        } else {
          log("Interception FAILED (defender did not sync).");
        }
      }
    }


    requestAnimationFrame(tick);
  }

  // start animation
  requestAnimationFrame(tick);
}

  // --- rest of original code: drawing, animations, filters ---
  function makeRangeLayerForFeature(f, opacity){
  const p = f.properties || {};
  if(p.is_ship) return null;
  const viz_range = (p.viz_range_km !== null && p.viz_range_km !== undefined) ? Number(p.viz_range_km) : (p.range_km ? Number(p.range_km) : null);
  if(!viz_range || isNaN(viz_range)) return null;
  const lat = Number(f.geometry.coordinates[1]);
  const lon = Number(f.geometry.coordinates[0]);

  // inner "dead" zone in kilometers (1000 km as requested)
  const inner_km = 1000;

  const viz_deg = (p.viz_degree !== null && p.viz_degree !== undefined && !isNaN(Number(p.viz_degree))) ? Number(p.viz_degree) : null;
  const isRadar = (String(p.category).toUpperCase().includes('RADAR'));

  // helper to decide if death_zone is truthy (boolean or string)
  const death_zone = (p.hasOwnProperty('death_zone') && (p.death_zone === true || String(p.death_zone).toLowerCase() === 'true'));

  // If death zone requested but inner radius >= outer radius -> nothing to draw
  if (death_zone && viz_range <= inner_km) return null;

  // Convert km to meters
  const outerMeters = viz_range * 1000.0;
  const innerMeters = inner_km * 1000.0;

  if(viz_deg !== null && isRadar){
    let az = null;
    if(p.viz_azimuth !== null && p.viz_azimuth !== undefined && !isNaN(Number(p.viz_azimuth))){
      az = Number(p.viz_azimuth);
    } else if(p.azimuth !== null && p.azimuth !== undefined && !isNaN(Number(p.azimuth))){
      az = Number(p.azimuth);
    } else {
      az = 0;
    }

    // full-circle case (360° or more)
    if(viz_deg >= 360){
      // if no death zone: draw circle as before
      if(!death_zone){
        return L.circle([lat, lon], { radius: outerMeters, color: colorFor(p.category), fill: true, fillOpacity: opacity, weight: 0.6, interactive: false, pane: 'rangesPane' });
      }
      // death_zone: build donut polygon (approximate circle with many points)
      const stepsOuter = Math.max(36, Math.round(360 / 5)); // ~72 pts by default
      const outerArc = computeSectorLatLngs(lat, lon, outerMeters, 0, 360, stepsOuter);
      const innerArc = computeSectorLatLngs(lat, lon, innerMeters, 0, 360, stepsOuter);
      if(outerArc.length === 0 || innerArc.length === 0) return null;
      // ensure closure and proper winding: outerArc as-is, innerArc reversed for correct hole orientation
      const innerHole = innerArc.slice().reverse();
      // build combined ring: outerArc followed by reversed innerArc (works as single closed ring)
      const ring = outerArc.concat(innerHole);
      return L.polygon(ring, { color: colorFor(p.category), fill: true, fillOpacity: opacity, weight: 0.6, interactive: false, pane: 'rangesPane' });
    }

    // sector case (viz_deg < 360)
    const start = az - viz_deg/2.0;
    const end = az + viz_deg/2.0;
    const steps = Math.max(6, Math.round(Math.min(360, Math.abs(viz_deg)) / 2));
    const outerArc = computeSectorLatLngs(lat, lon, outerMeters, start, end, steps);
    if(outerArc.length === 0) return null;

    if(!death_zone){
      // old behavior: polygon wedge from center out to outer arc and back
      const polyPts = [];
      polyPts.push([lat, lon]);
      outerArc.forEach(a => polyPts.push(a));
      polyPts.push([lat, lon]);
      return L.polygon(polyPts, { color: colorFor(p.category), fill: true, fillOpacity: opacity, weight: 0.6, interactive: false, pane: 'rangesPane' });
    } else {
      // death_zone true: build sector donut between innerMeters and outerMeters
      const innerArc = computeSectorLatLngs(lat, lon, innerMeters, start, end, steps);
      if(innerArc.length === 0) return null;
      // Build closed ring: outer arc (start->end) then inner arc reversed (end->start)
      const innerReversed = innerArc.slice().reverse();
      const ring = outerArc.concat(innerReversed);
      return L.polygon(ring, { color: colorFor(p.category), fill: true, fillOpacity: opacity, weight: 0.6, interactive: false, pane: 'rangesPane' });
    }
  } else {
    // non-radar (simple circle)
    if(!death_zone){
      return L.circle([lat, lon], { radius: outerMeters, color: colorFor(p.category), fill: true, fillOpacity: opacity, weight: 0.6, interactive: false, pane: 'rangesPane' });
    } else {
      // death_zone for non-radar: render donut circle between innerMeters and outerMeters
      if (innerMeters >= outerMeters) return null;
      const stepsOuter = Math.max(36, Math.round(360 / 5));
      const outerArc = computeSectorLatLngs(lat, lon, outerMeters, 0, 360, stepsOuter);
      const innerArc = computeSectorLatLngs(lat, lon, innerMeters, 0, 360, stepsOuter);
      if(outerArc.length === 0 || innerArc.length === 0) return null;
      const ring = outerArc.concat(innerArc.slice().reverse());
      return L.polygon(ring, { color: colorFor(p.category), fill: true, fillOpacity: opacity, weight: 0.6, interactive: false, pane: 'rangesPane' });
    }
  }
}

  function colorFor(cat){
    if(!cat) return '#777';
    if(String(cat).toUpperCase().includes('ABM')) return '#d73027';
    if(String(cat).toUpperCase().includes('RADAR')) return '#4575b4';
    if(String(cat).toUpperCase().includes('SATELLITE')) return '#91cf60';
    if(String(cat).toUpperCase().includes('SHIP')) return '#2b6cb0';
    return '#777';
  }

  function createMarkerFromFeature(f, scale){
    const p = f.properties || {};
    const lat = Number(f.geometry.coordinates[1]);
    const lon = Number(f.geometry.coordinates[0]);
    const cat = (p.category || '').toString();
    const num = (p.number && !isNaN(p.number)) ? Number(p.number) : 1;
    let radius_px;
    if(String(cat).toUpperCase().includes('ABM')){
      const minN = 8, maxN = 16;
      let n = Math.max(minN, Math.min(maxN, Math.round(num)));
      const multiplier = 1.0 + ((n - minN) / (maxN - minN));
      radius_px = Math.max(BASE_RADIUS, Math.round(BASE_RADIUS * multiplier * scale));
    } else {
      radius_px = Math.max(BASE_RADIUS, Math.round(BASE_RADIUS * scale));
    }

    if(p.is_ship){
      const marker = L.marker([lat, lon], { icon: shipIconInstance, pane: 'markersPane' });
      marker.feature = f;
      const popupHtml = `<strong>${safeText(p.system_name || p.system_id)}</strong><br/>${safeText(p.location_name)}<br/>Category: ${safeText(p.category)}<br/>Operability: ${safeText(p.operability)}<br/>Year: ${safeText(p.year_operational)}<br/>Note: ship (animated along orbit/route)`;
      marker.bindPopup(popupHtml, { maxWidth: 320 });
      return marker;
    }

    if(p.is_virtual_satellite){
      const marker = L.marker([lat, lon], { icon: satIconInstance, pane: 'markersPane' });
      marker.feature = f;
      const popupHtml = `<strong>${safeText(p.system_name || p.system_id)}</strong><br/>${safeText(p.location_name)}<br/>Category: ${safeText(p.category)}<br/>Operability: ${safeText(p.operability)}<br/>Year: ${safeText(p.year_operational)}<br/>Note: virtual orbit`;
      marker.bindPopup(popupHtml, { maxWidth: 320 });
      return marker;
    } else {
      const marker = L.circleMarker([lat, lon], {
        radius: radius_px,
        color: colorFor(cat),
        fillColor: colorFor(cat),
        fillOpacity: 0.85,
        weight: 1,
        pane: 'markersPane'
      });
      marker.feature = f;
      const popupHtml = `<strong>${safeText(p.system_name || p.system_id)}</strong><br/>${safeText(p.location_name)}<br/>Category: ${safeText(p.category)}<br/>Subtype: ${safeText(p.subtype)}<br/>Number: ${safeText(p.number)}<br/>Operability: ${safeText(p.operability)}<br/>Year: ${safeText(p.year_operational)}<br/>Range (km): ${safeText(p.range_km)}<br/>Degree: ${safeText(p.degree)}<br/>Azimuth: ${safeText(p.azimuth)}`;
      marker.bindPopup(popupHtml, { maxWidth: 320 });
      return marker;
    }
  }

  function drawConnectionsIfNeeded(){
    connectionsLayer.clearLayers();
    if(!document.getElementById('show_connections').checked) return;
    CONNECTIONS.forEach(conn => {
      const coords = conn.coords || [];
      if(coords.length < 2) return;
      const latlon = coords.map(c => [c[1], c[0]]);
      const pl = L.polyline(latlon, { color:'#444', dashArray:'6 6', weight:1.5, pane:'connectionsPane' });
      pl.bindPopup(`<strong>Connection id</strong>: ${safeText(conn.id)}<br/>Points: ${coords.length}`);
      connectionsLayer.addLayer(pl);
    });
  }

  function drawOrbitsForVisible(featuresVisible){
    orbitsLayer.clearLayers();
    if(!document.getElementById('show_orbits').checked) return;
    featuresVisible.forEach(f => {
      const p = f.properties || {};
      if(p.is_ship) return; // do not draw ship orbits under ship
      if(p.orbit && Array.isArray(p.orbit) && p.orbit.length>3){
        const latlon = p.orbit.map(pt => [pt[0], pt[1]]);
        const poly = L.polyline(latlon, { color:'#888', weight:1.2, dashArray:'4 6', pane:'orbitsPane', interactive:false });
        orbitsLayer.addLayer(poly);
      }
    });
  }

  // animation manager
  let animIntervals = {};
  function clearAllAnimations(){
    Object.keys(animIntervals).forEach(k => {
      try{ clearInterval(animIntervals[k]); } catch(e){}
    });
    animIntervals = {};
  }

  function applyFilters(){
    clearAllAnimations();
    const prevPersistent = persistentFeatureId;
    markersLayer.clearLayers();
    tempRangeLayer.clearLayers();

    const selCats = Array.from(document.querySelectorAll('input.cat:checked')).map(n=>n.value);
    const selSystems = Array.from(document.querySelectorAll('input.system_name:checked')).map(n=>({cat:n.dataset.category, system:n.value}));
    const selSubs = Array.from(document.querySelectorAll('input.subtype:checked')).map(n=>n.value);
    const selOps = Array.from(document.querySelectorAll('input.op:checked')).map(n=>n.value);
    const yearMin = parseInt(document.getElementById('year_min').value) || null;
    const yearMax = parseInt(document.getElementById('year_max').value) || null;
    const scale = parseFloat(document.getElementById('scale_mult').value) || DEFAULT_SCALE;
    const animateSat = document.getElementById('animate_sat').checked;

    const systemsSet = new Set(selSystems.map(o=>o.system));
    const effectiveSystems = new Set();
    if(systemsSet.size === 0){
      selCats.forEach(cat => { const sm = HIER[cat] || {}; Object.keys(sm).forEach(s => effectiveSystems.add(s)); });
    } else { systemsSet.forEach(s => effectiveSystems.add(s)); }
    const subSet = new Set(selSubs);
    const opSet = new Set(selOps);

    let added = 0;
    const visibleFeatures = [];

    features.forEach(f => {
      const p = f.properties || {};
      const cat = (p.category || '').toString();
      if(selCats.length > 0 && !selCats.includes(cat)) return;
      const sys = p.system_name || p.system_id || '';
      if(effectiveSystems.size > 0 && !effectiveSystems.has(sys)) return;
      if(subSet.size > 0){ const st = p.subtype || ''; if(!subSet.has(st)) return; }
      const op = (p.operability || '').toString();
      if(opSet.size > 0 && !opSet.has(op)) return;
      const y = p.year_operational || null;
      if(yearMin && y && y < yearMin) return;
      if(yearMax && y && y > yearMax) return;

      const mk = createMarkerFromFeature(f, scale);

      mk.on('mouseover', function(e){
        tempRangeLayer.clearLayers();
        const rlayer = makeRangeLayerForFeature(this.feature, parseFloat(document.getElementById('range_op').value) || 0.25);
        if(rlayer) tempRangeLayer.addLayer(rlayer);
        try{ if(this.bringToFront) this.bringToFront(); } catch(e){}
      });
      mk.on('mouseout', function(e){ tempRangeLayer.clearLayers(); });

      mk.on('click', function(e){
        try{ this.openPopup(); } catch(err){}
        const fid = this.feature.properties.id;
        if(persistentFeatureId === fid){ persistentRangeLayer.clearLayers(); persistentFeatureId = null; }
        else {
          persistentRangeLayer.clearLayers();
          const circ = makeRangeLayerForFeature(this.feature, parseFloat(document.getElementById('range_op').value) || 0.25);
          if(circ){ persistentRangeLayer.addLayer(circ); persistentFeatureId = fid; } else { persistentFeatureId = null; }
        }
        try{ if(this.bringToFront) this.bringToFront(); } catch(e){}
      });

      markersLayer.addLayer(mk);
      visibleFeatures.push(f);
      added += 1;
    });

    if(prevPersistent){
      const stillVisible = visibleFeatures.find(ff => ff.properties && String(ff.properties.id) === String(prevPersistent));
      if(!stillVisible){ persistentRangeLayer.clearLayers(); persistentFeatureId = null; }
      else {
        const circ = makeRangeLayerForFeature(stillVisible, parseFloat(document.getElementById('range_op').value) || 0.25);
        persistentRangeLayer.clearLayers();
        if(circ){ persistentRangeLayer.addLayer(circ); persistentFeatureId = prevPersistent; } else { persistentFeatureId = null; }
      }
    }

    drawConnectionsIfNeeded();
    drawOrbitsForVisible(visibleFeatures);

    try { allRangesLayer.clearLayers(); } catch(e){}
    if (document.getElementById('show_all_ranges').checked) {
      const opacityAll = parseFloat(document.getElementById('range_op').value) || 0.25;
      visibleFeatures.forEach(fv => {
        try {
          const rl = makeRangeLayerForFeature(fv, opacityAll);
          if (rl) allRangesLayer.addLayer(rl);
        } catch (e) {}
      });
    }

    if(animateSat){
      markersLayer.eachLayer(function(layer){
        if(layer.feature && layer.feature.properties && layer.feature.properties.orbit && Array.isArray(layer.feature.properties.orbit)){
          const id = layer.feature.properties.id;
          const orbit = layer.feature.properties.orbit;
          if(!Array.isArray(orbit) || orbit.length < 4) return;
          let idx = 0;
          try {
            const latlng = layer.getLatLng();
            let bestDist = Infinity;
            for(let i=0;i<orbit.length;i++){
              const dlat = latlng.lat - orbit[i][0];
              const dlon = latlng.lng - orbit[i][1];
              const dist = dlat*dlat + dlon*dlon;
              if(dist < bestDist){ bestDist = dist; idx = i; }
            }
          } catch(e){ idx = 0; }

          const isSat = !!layer.feature.properties.is_virtual_satellite;
          const isShip = !!layer.feature.properties.is_ship;

          // adaptive animation parameters
          const baseIntervalMs = isShip ? 40 : 120;
          const targetPointsPerLoop = isShip ? 400 : orbit.length;
          let stepInc = 1;
          if(orbit.length > 0){
            stepInc = Math.max(1, Math.round(orbit.length / Math.max(50, Math.min(targetPointsPerLoop, orbit.length))));
          }
          stepInc = Math.max(1, Math.min(stepInc, Math.max(1, Math.floor(orbit.length / 4))));

          try{ if(animIntervals[id]) { clearInterval(animIntervals[id]); delete animIntervals[id]; } } catch(e){}
          animIntervals[id] = setInterval(function(){
            idx = (idx + stepInc) % orbit.length;
            const p = orbit[idx];
            try { layer.setLatLng([p[0], p[1]]); } catch(e){}
          }, baseIntervalMs);
        }
      });
    }

    log('Markers affichés: ' + added);
    try{ if(markersLayer.getLayers().length > 0) map.fitBounds(markersLayer.getBounds().pad(0.2)); } catch(e){}
  }

  map.on('click', function(e){ persistentRangeLayer.clearLayers(); persistentFeatureId = null; });

  applyFilters();
});
</script>
</body>
</html>
"""

# Replace placeholders and write HTML file
html = html_template.replace("GEOJSON_PLACEHOLDER", GEOJSON_STR)
html = html.replace("HIER_PLACEHOLDER", HIER_STR)
html = html.replace("CONNECTIONS_PLACEHOLDER", CONNS_STR)
html = html.replace("GEOCODES_PLACEHOLDER", GEOCODES_STR)
html = html.replace("SAT_ICON_URL_PLACEHOLDER", SAT_ICON_URL)
html = html.replace("SHIP_ICON_URL_PLACEHOLDER", SHIP_ICON_URL)
html = html.replace("MISSILE_ICON_URL_PLACEHOLDER", MISSILE_ICON_URL)
html = html.replace("EXPLOSION_ICON_URL_PLACEHOLDER", EXPLOSION_ICON_URL)
html = html.replace("BASE_RADIUS_PLACEHOLDER", str(int(round(BASE_RADIUS))))
html = html.replace("MIN_RADIUS_PLACEHOLDER", str(int(round(MIN_RADIUS))))
html = html.replace("DEFAULT_SCALE_PLACEHOLDER", str(DEFAULT_SCALE))
html = html.replace("DEFENSE_MISSILE_ICON_URL_PLACEHOLDER", DEFENSE_MISSILE_ICON_URL)
html = html.replace("DEFENSE_EXPLOSION_ICON_URL_PLACEHOLDER", DEFENSE_EXPLOSION_ICON_URL)

outp = Path(OUT_HTML_PATH).expanduser()
outp.parent.mkdir(parents=True, exist_ok=True)
outp.write_text(html, encoding="utf-8")
print("Wrote map (forced north-point interception via Onega) to:", outp.resolve())
print("Serve with: python -m http.server 8000  then open http://localhost:8000/" + outp.name)
