"""Dataset writers (YOLO, YOLO-seg, COCO), image conversion and previews."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw

from geochips.core import Chip, _polygons

IMAGE_FORMATS = {"png": "PNG", "jpg": "JPEG", "tif": "GTiff"}


class Stretch:
    """Per-band percentile stretch to uint8, computed once for a whole raster.

    Using the same limits for every chip of a raster keeps brightness consistent
    across chips. uint8 rasters are passed through unchanged.
    """

    def __init__(self, lows: np.ndarray | None, highs: np.ndarray | None):
        self.lows, self.highs = lows, highs

    @classmethod
    def from_raster(cls, path: Path, bands: Sequence[int] | None = None, percentiles=(2, 98), max_side: int = 1024):
        with rasterio.open(path) as src:
            if src.dtypes[0] == "uint8":
                return cls(None, None)
            bands = list(bands) if bands else list(range(1, src.count + 1))
            scale = max(1, max(src.width, src.height) // max_side)
            shape = (len(bands), max(1, src.height // scale), max(1, src.width // scale))
            data = src.read(bands, out_shape=shape, masked=True).astype("float64")
        lows, highs = [], []
        for band in data:
            values = band.compressed()
            lo, hi = np.percentile(values, percentiles) if values.size else (0.0, 1.0)
            lows.append(lo)
            highs.append(hi if hi > lo else lo + 1)
        return cls(np.array(lows), np.array(highs))

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.lows is None:
            return image
        scaled = (image.astype("float64") - self.lows[:, None, None]) / (self.highs - self.lows)[:, None, None]
        return (np.clip(scaled, 0, 1) * 255).round().astype("uint8")


def to_pil(image: np.ndarray) -> Image.Image:
    """(bands, h, w) uint8 -> PIL image. One band is grey, three or more use the first three."""
    if image.shape[0] == 1:
        return Image.fromarray(image[0], "L")
    if image.shape[0] == 2:
        image = np.concatenate([image, image[:1]])
    return Image.fromarray(np.moveaxis(image[:3], 0, -1), "RGB")


def save_image(chip: Chip, path: Path, image_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image_format == "tif":
        profile = {
            "driver": "GTiff",
            "width": chip.image.shape[2],
            "height": chip.image.shape[1],
            "count": chip.image.shape[0],
            "dtype": chip.image.dtype,
            "crs": chip.crs,
            "transform": chip.transform,
            "compress": "deflate",
        }
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(chip.image)
    else:
        to_pil(chip.image).save(path, IMAGE_FORMATS[image_format])


def get_writer(fmt: str, out_dir: Path, classes: list[str], image_format: str):
    writers = {"yolo": YoloWriter, "yolo-seg": YoloWriter, "coco": CocoWriter}
    if fmt not in writers:
        raise ValueError(f"unknown format {fmt!r}; choose from {', '.join(writers)}")
    if image_format not in IMAGE_FORMATS:
        raise ValueError(f"unknown image format {image_format!r}; choose from {', '.join(IMAGE_FORMATS)}")
    return writers[fmt](out_dir, classes, image_format, segment=fmt == "yolo-seg")


class YoloWriter:
    """Ultralytics layout: images/{split}, labels/{split}, data.yaml.

    Boxes are `cls cx cy w h`; with segment=True each polygon part is
    `cls x1 y1 x2 y2 ...`. All values are normalised to the chip size.
    """

    def __init__(self, out_dir: Path, classes: list[str], image_format: str, segment: bool = False):
        self.out, self.classes, self.image_format, self.segment = out_dir, classes, image_format, segment
        self.ids = {c: i for i, c in enumerate(classes)}
        self.splits: set[str] = set()

    def add(self, chip: Chip) -> None:
        self.splits.add(chip.split)
        save_image(chip, self.out / "images" / chip.split / f"{chip.name}.{self.image_format}", self.image_format)
        h, w = chip.image.shape[1:]
        lines = []
        for lab in chip.labels:
            cid = self.ids[lab.cls]
            if self.segment:
                for poly in _polygons(lab.geometry):
                    xy = np.asarray(poly.exterior.coords)[:-1] / [w, h]
                    lines.append(f"{cid} " + " ".join(f"{v:.6f}" for v in xy.ravel()))
            else:
                x0, y0, x1, y1 = lab.geometry.bounds
                lines.append(f"{cid} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
        label_path = self.out / "labels" / chip.split / f"{chip.name}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def close(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        val = "images/val" if "val" in self.splits else "images/train"
        names = "\n".join(f"  {i}: {json.dumps(c)}" for i, c in enumerate(self.classes))
        (self.out / "data.yaml").write_text(
            f"path: {self.out.resolve().as_posix()}\ntrain: images/train\nval: {val}\nnames:\n{names}\n", encoding="utf-8"
        )


class CocoWriter:
    """COCO detection/segmentation JSON: images/{split} and annotations/instances_{split}.json."""

    def __init__(self, out_dir: Path, classes: list[str], image_format: str, segment: bool = False):
        self.out, self.image_format = out_dir, image_format
        self.categories = [{"id": i + 1, "name": c, "supercategory": "none"} for i, c in enumerate(classes)]
        self.ids = {c: i + 1 for i, c in enumerate(classes)}
        self.data: dict[str, dict] = {}
        self.next_ann = 1
        self.next_img = 1

    def add(self, chip: Chip) -> None:
        split = self.data.setdefault(chip.split, {"images": [], "annotations": [], "categories": self.categories})
        file_name = f"{chip.name}.{self.image_format}"
        save_image(chip, self.out / "images" / chip.split / file_name, self.image_format)
        h, w = chip.image.shape[1:]
        image_id = self.next_img
        self.next_img += 1
        split["images"].append(
            {"id": image_id, "file_name": file_name, "width": int(w), "height": int(h), "source": chip.source.name,
             "window": [int(chip.window.col_off), int(chip.window.row_off), int(chip.window.width), int(chip.window.height)]}
        )
        for lab in chip.labels:
            x0, y0, x1, y1 = lab.geometry.bounds
            seg = [[round(v, 2) for v in np.asarray(p.exterior.coords)[:-1].ravel().tolist()] for p in _polygons(lab.geometry)]
            split["annotations"].append(
                {"id": self.next_ann, "image_id": image_id, "category_id": self.ids[lab.cls], "segmentation": seg,
                 "area": round(lab.geometry.area, 2), "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
                 "iscrowd": 0}
            )
            self.next_ann += 1

    def close(self) -> None:
        ann_dir = self.out / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)
        for split, data in self.data.items():
            (ann_dir / f"instances_{split}.json").write_text(json.dumps(data), encoding="utf-8")


PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230)]


def write_preview(chip: Chip, out_dir: Path, classes: list[str]) -> None:
    """Draw the chip's labels on it, for a quick visual check of the alignment."""
    out_dir.mkdir(parents=True, exist_ok=True)
    image = chip.image
    if image.dtype != np.uint8:  # GeoTIFF chips keep their dtype; stretch this chip alone for display
        flat = image.reshape(image.shape[0], -1).astype("float64")
        lows, highs = np.percentile(flat, 2, axis=1), np.percentile(flat, 98, axis=1)
        image = Stretch(lows, np.where(highs > lows, highs, lows + 1))(image)
    pil = to_pil(image).convert("RGBA")
    overlay = Image.new("RGBA", pil.size)
    draw = ImageDraw.Draw(overlay)
    for lab in chip.labels:
        color = PALETTE[classes.index(lab.cls) % len(PALETTE)]
        for poly in _polygons(lab.geometry):
            ring = [tuple(xy) for xy in poly.exterior.coords]
            draw.polygon(ring, fill=color + (70,))
            draw.line(ring, fill=color + (255,), width=1)
        x0, y0, x1, y1 = lab.geometry.bounds
        if min(x1 - x0, y1 - y0) >= 48:  # name only objects big enough to read
            draw.text((x0 + 3, y0 + 2), lab.cls, fill=(255, 255, 255, 255))
    pil = Image.alpha_composite(pil, overlay).convert("RGB")
    pil.save(out_dir / f"{chip.name}.png")
