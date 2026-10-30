"""Cut GeoTIFFs and vector labels into ML training chips (YOLO / COCO)."""

from geochips.core import Chip, ChipConfig, Label, iter_chips, iter_windows, load_labels, make_chips

__all__ = ["Chip", "ChipConfig", "Label", "iter_chips", "iter_windows", "load_labels", "make_chips"]
__version__ = "0.1.0"
