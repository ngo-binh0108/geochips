"""Download a small demo dataset: a Sentinel-2 true-colour window and OSM land-use polygons.

    python examples/download_demo.py examples/data

Writes `utrecht_s2.tif` (10 m, EPSG:32631) and `utrecht_landuse.geojson`
(EPSG:4326). Imagery: Copernicus Sentinel-2 via Element 84 Earth Search.
Labels: (c) OpenStreetMap contributors, ODbL.
"""

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

BBOX = (5.06, 52.04, 5.20, 52.12)  # lon/lat around Utrecht, NL
SCENE = ("https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/"
         "31/U/FT/2025/8/S2A_31UFT_20250811_0_L2A/TCI.tif")
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
CLASSES = {
    ("landuse", "residential"): "residential",
    ("landuse", "industrial"): "industrial",
    ("landuse", "commercial"): "industrial",
    ("landuse", "farmland"): "farmland",
    ("landuse", "meadow"): "grass",
    ("landuse", "grass"): "grass",
    ("leisure", "park"): "grass",
    ("landuse", "forest"): "forest",
    ("natural", "wood"): "forest",
    ("natural", "water"): "water",
}


def download_image(out: Path) -> None:
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"), rasterio.open(SCENE) as src:
        window = from_bounds(*transform_bounds("EPSG:4326", src.crs, *BBOX), src.transform).round_offsets().round_lengths()
        data = src.read(window=window)
        profile = src.profile | {"width": window.width, "height": window.height,
                                 "transform": src.window_transform(window), "compress": "deflate", "tiled": False}
        profile.pop("blockxsize", None), profile.pop("blockysize", None)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(data)


def download_labels(out: Path) -> None:
    s, w, n, e = BBOX[1], BBOX[0], BBOX[3], BBOX[2]
    selectors = "".join(f'way["{k}"="{v}"]({s},{w},{n},{e});' for k, v in CLASSES)
    query = f"[out:json][timeout:120];({selectors});out geom;"
    for server in OVERPASS:
        req = urllib.request.Request(server, data=urllib.parse.urlencode({"data": query}).encode(),
                                     headers={"User-Agent": "geochips-demo"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                elements = json.load(r)["elements"]
            break
        except OSError as err:
            print(f"{server}: {err}, trying the next server")
    else:
        raise SystemExit("all Overpass servers failed; try again later")
    features = []
    for el in elements:
        geom = el.get("geometry") or []
        if len(geom) < 4 or geom[0] != geom[-1]:
            continue  # closed ways only
        tags = el.get("tags", {})
        cls = next((c for (k, v), c in CLASSES.items() if tags.get(k) == v), None)
        ring = [[p["lon"], p["lat"]] for p in geom]
        features.append({"type": "Feature", "properties": {"class": cls, "osm_id": el["id"]},
                         "geometry": {"type": "Polygon", "coordinates": [ring]}})
    out.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")


if __name__ == "__main__":
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "examples/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    download_image(out_dir / "utrecht_s2.tif")
    download_labels(out_dir / "utrecht_landuse.geojson")
    print(f"wrote {out_dir / 'utrecht_s2.tif'} and {out_dir / 'utrecht_landuse.geojson'}")
