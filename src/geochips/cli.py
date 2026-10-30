"""Command line interface: `geochips RASTER... -l LABELS -o OUT`."""

from __future__ import annotations

import argparse
import sys

from geochips import __version__
from geochips.core import ChipConfig, make_chips


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="geochips",
        description="Cut GeoTIFFs and vector labels into YOLO or COCO training chips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("rasters", nargs="+", help="GeoTIFF files or directories of them")
    p.add_argument("-l", "--labels", required=True, help="polygon labels (GeoJSON; Shapefile/GeoPackage with geopandas)")
    p.add_argument("-o", "--out", required=True, help="output dataset directory")
    p.add_argument("-f", "--format", default="yolo", choices=["yolo", "yolo-seg", "coco"], help="label format")
    p.add_argument("-s", "--size", type=int, default=512, help="chip size in pixels")
    p.add_argument("--overlap", type=int, default=0, help="overlap between neighbouring chips in pixels")
    p.add_argument("--class-field", default="class", help="label property holding the class name")
    p.add_argument("--classes", help="comma-separated class names, in id order (default: all, sorted)")
    p.add_argument("--bands", help="comma-separated 1-based band indices, e.g. 4,3,2")
    p.add_argument("--image-format", default="png", choices=["png", "jpg", "tif"],
                   help="chip image format; tif keeps the original dtype and georeferencing")
    p.add_argument("--min-visible", type=float, default=0.0,
                   help="drop objects with less than this fraction of their area inside the chip")
    p.add_argument("--min-area", type=float, default=4.0, help="drop clipped fragments smaller than this many pixels")
    p.add_argument("--max-nodata", type=float, default=1.0, help="skip chips with a larger nodata fraction")
    p.add_argument("--skip-empty", action="store_true", help="skip chips without labels")
    p.add_argument("--val", type=float, default=0.0, help="fraction of chips (or images) for the validation split")
    p.add_argument("--split-by", default="chip", choices=["chip", "image"],
                   help="split per chip, or per raster to avoid overlap leakage between splits")
    p.add_argument("--seed", type=int, default=0, help="random seed for the split")
    p.add_argument("--preview", type=int, default=0, metavar="N", help="also write N chips with labels drawn on them")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    config = ChipConfig(
        size=a.size,
        overlap=a.overlap,
        class_field=a.class_field,
        bands=[int(b) for b in a.bands.split(",")] if a.bands else None,
        min_visible=a.min_visible,
        min_area_px=a.min_area,
        max_nodata=a.max_nodata,
        skip_empty=a.skip_empty,
        val_fraction=a.val,
        split_by=a.split_by,
        seed=a.seed,
        classes=[c.strip() for c in a.classes.split(",")] if a.classes else None,
    )
    try:
        stats = make_chips(a.rasters, a.labels, a.out, a.format, config, a.image_format, a.preview)
    except (OSError, ValueError, ImportError) as e:
        print(f"geochips: error: {e}", file=sys.stderr)
        return 1
    print(f"classes: {', '.join(stats['classes']) or '(none)'}")
    for split in ("train", "val"):
        if stats["chips"][split]:
            print(f"{split}: {stats['chips'][split]} chips, {stats['labels'][split]} labels")
    print(f"written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
