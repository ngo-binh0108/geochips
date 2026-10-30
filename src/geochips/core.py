"""Tiling, label loading and label clipping."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_geom
from rasterio.windows import Window
from shapely import STRtree, affinity, box
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry

DEFAULT_LABEL_CRS = "EPSG:4326"  # RFC 7946: GeoJSON without a "crs" member is WGS84


@dataclass
class Label:
    """One labelled object: a class name and its geometry."""

    cls: str
    geometry: BaseGeometry


@dataclass
class Chip:
    """One output chip and the labels clipped to it, in chip pixel coordinates."""

    name: str
    source: Path
    window: Window
    image: np.ndarray  # (bands, height, width)
    labels: list[Label]
    transform: rasterio.Affine
    crs: CRS | None
    split: str = "train"


@dataclass
class ChipConfig:
    size: int = 512
    overlap: int = 0
    class_field: str = "class"
    bands: Sequence[int] | None = None  # 1-based, as in rasterio
    min_visible: float = 0.0  # fraction of an object's area that must fall inside the chip
    min_area_px: float = 4.0  # drop clipped fragments smaller than this many pixels
    max_nodata: float = 1.0  # skip chips with a larger nodata fraction
    skip_empty: bool = False  # skip chips that contain no labels
    val_fraction: float = 0.0
    split_by: str = "chip"  # "chip", or "image" to keep each raster in one split
    seed: int = 0
    classes: list[str] | None = None  # fixed class order; inferred from labels if None


def iter_windows(width: int, height: int, size: int, overlap: int = 0) -> Iterator[Window]:
    """Yield windows of `size` pixels that cover the whole image.

    Windows step by `size - overlap`. The last row and column are shifted back so
    they end on the image edge instead of running past it, which keeps every chip
    the same size without padding. Images smaller than `size` give one window
    that is as large as the image.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    for row in _offsets(height, size, size - overlap):
        for col in _offsets(width, size, size - overlap):
            yield Window(col, row, min(size, width), min(size, height))


def _offsets(length: int, size: int, step: int) -> list[int]:
    if length <= size:
        return [0]
    offsets = list(range(0, length - size + 1, step))
    if offsets[-1] != length - size:
        offsets.append(length - size)
    return offsets


def load_labels(path: str | Path, class_field: str = "class") -> tuple[list[Label], CRS]:
    """Read polygon labels and their CRS.

    GeoJSON is read directly. Other formats (Shapefile, GeoPackage, ...) are read
    through geopandas, if it is installed. Features without `class_field` get the
    class "object"; non-polygon geometries are ignored.
    """
    path = Path(path)
    if path.suffix.lower() in (".geojson", ".json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        features = data["features"] if data.get("type") == "FeatureCollection" else [data]
        crs = CRS.from_user_input(_geojson_crs(data))
        records = [(f.get("properties") or {}, f.get("geometry")) for f in features]
    else:
        try:
            import geopandas
        except ImportError as e:
            raise ImportError(f"reading {path.suffix} files needs geopandas; use GeoJSON or pip install geopandas") from e
        gdf = geopandas.read_file(path)
        crs = CRS.from_user_input(gdf.crs or DEFAULT_LABEL_CRS)
        props = gdf.drop(columns=gdf.geometry.name).to_dict("records")
        records = [(p, mapping(g) if g is not None else None) for p, g in zip(props, gdf.geometry)]

    labels = []
    for props, geom in records:
        if not geom:
            continue
        g = shape(geom)
        if not isinstance(g, (Polygon, MultiPolygon)) or g.is_empty:
            continue
        if not g.is_valid:
            g = g.buffer(0)
        labels.append(Label(str(props.get(class_field, "object")), g))
    return labels, crs


def _geojson_crs(data: dict) -> str:
    name = ((data.get("crs") or {}).get("properties") or {}).get("name")
    if not name or name.endswith("CRS84"):
        return DEFAULT_LABEL_CRS
    return name


def labels_to_pixels(labels: Iterable[Label], label_crs: CRS, src) -> list[Label]:
    """Reproject labels to the raster CRS, then to the raster's pixel grid."""
    inv = ~src.transform
    params = [inv.a, inv.b, inv.d, inv.e, inv.c, inv.f]
    bounds = box(*src.bounds)
    out = []
    for lab in labels:
        g = lab.geometry
        if src.crs and label_crs and CRS.from_user_input(src.crs) != label_crs:
            g = shape(transform_geom(label_crs, src.crs, mapping(g)))
        if not g.intersects(bounds):
            continue
        out.append(Label(lab.cls, affinity.affine_transform(g, params)))
    return out


def clip_labels(labels: Sequence[Label], tree: STRtree, window: Window, min_visible: float, min_area_px: float) -> list[Label]:
    """Clip pixel-space labels to a window; return them in window coordinates."""
    col, row = window.col_off, window.row_off
    frame = box(col, row, col + window.width, row + window.height)
    out = []
    for i in tree.query(frame, predicate="intersects"):
        lab = labels[i]
        part = lab.geometry.intersection(frame)
        if part.is_empty or lab.geometry.area == 0:
            continue
        if part.area / lab.geometry.area < min_visible:
            continue
        polys = [p for p in _polygons(part) if p.area >= min_area_px]
        if not polys:
            continue
        g = polys[0] if len(polys) == 1 else MultiPolygon(polys)
        out.append(Label(lab.cls, affinity.translate(g, -col, -row)))
    return out


def _polygons(g: BaseGeometry) -> list[Polygon]:
    if isinstance(g, Polygon):
        return [g]
    if hasattr(g, "geoms"):
        return [p for sub in g.geoms for p in _polygons(sub)]
    return []  # lines or points left over from touching edges


def iter_chips(rasters: Sequence[str | Path], labels: Sequence[Label], label_crs: CRS, config: ChipConfig) -> Iterator[Chip]:
    """Yield chips with their clipped labels for every raster."""
    rng = random.Random(config.seed)
    rasters = [Path(r) for r in rasters]
    val_images = set(_pick_val(len(rasters), config.val_fraction, rng)) if config.split_by == "image" else set()
    for i, raster in enumerate(rasters):
        with rasterio.open(raster) as src:
            bands = list(config.bands) if config.bands else list(range(1, src.count + 1))
            px_labels = labels_to_pixels(labels, label_crs, src)
            tree = STRtree([lab.geometry for lab in px_labels])
            windows = list(iter_windows(src.width, src.height, config.size, config.overlap))
            if config.split_by == "image":
                val_windows = set(range(len(windows))) if i in val_images else set()
            else:
                val_windows = set(_pick_val(len(windows), config.val_fraction, rng))
            for j, win in enumerate(windows):
                if config.max_nodata < 1.0:
                    valid = src.dataset_mask(window=win) > 0
                    if 1.0 - valid.mean() > config.max_nodata:
                        continue
                chip_labels = clip_labels(px_labels, tree, win, config.min_visible, config.min_area_px)
                if config.skip_empty and not chip_labels:
                    continue
                split = "val" if j in val_windows else "train"
                yield Chip(
                    name=f"{raster.stem}_{int(win.row_off)}_{int(win.col_off)}",
                    source=raster,
                    window=win,
                    image=src.read(bands, window=win),
                    labels=chip_labels,
                    transform=src.window_transform(win),
                    crs=src.crs,
                    split=split,
                )


def _pick_val(n: int, fraction: float, rng: random.Random) -> list[int]:
    """Pick round(n * fraction) random indices, at least one when fraction > 0 and n > 1."""
    k = round(n * fraction)
    if fraction > 0 and n > 1:
        k = max(k, 1)
    return rng.sample(range(n), min(k, n))


def make_chips(
    rasters: str | Path | Sequence[str | Path],
    labels: str | Path,
    out_dir: str | Path,
    fmt: str = "yolo",
    config: ChipConfig | None = None,
    image_format: str = "png",
    preview: int = 0,
) -> dict:
    """Cut rasters and labels into a training dataset in `out_dir`.

    `fmt` is "yolo" (boxes), "yolo-seg" (polygons) or "coco". Returns a summary
    with the number of chips and labels written per split.
    """
    from geochips.writers import Stretch, get_writer, write_preview

    config = config or ChipConfig()
    rasters = expand_rasters(rasters)
    all_labels, label_crs = load_labels(labels, config.class_field)
    classes = config.classes or sorted({lab.cls for lab in all_labels})
    writer = get_writer(fmt, Path(out_dir), classes, image_format)

    stats: dict = {"chips": {"train": 0, "val": 0}, "labels": {"train": 0, "val": 0}, "classes": classes}
    stretches: dict[Path, Stretch] = {}
    for chip in iter_chips(rasters, all_labels, label_crs, config):
        chip.labels = [lab for lab in chip.labels if lab.cls in classes]
        if image_format != "tif":
            if chip.source not in stretches:
                stretches[chip.source] = Stretch.from_raster(chip.source, config.bands)
            chip.image = stretches[chip.source](chip.image)
        writer.add(chip)
        if stats["chips"]["train"] + stats["chips"]["val"] < preview:
            write_preview(chip, Path(out_dir) / "preview", classes)
        stats["chips"][chip.split] += 1
        stats["labels"][chip.split] += len(chip.labels)
    writer.close()
    return stats


def expand_rasters(rasters: str | Path | Sequence[str | Path]) -> list[Path]:
    """Accept one path, a list of paths, or directories of .tif/.tiff files."""
    if isinstance(rasters, (str, Path)):
        rasters = [rasters]
    out: list[Path] = []
    for r in rasters:
        r = Path(r)
        if r.is_dir():
            out.extend(sorted(p for p in r.iterdir() if p.suffix.lower() in (".tif", ".tiff", ".vrt")))
        else:
            out.append(r)
    if not out:
        raise FileNotFoundError("no rasters found")
    return out
