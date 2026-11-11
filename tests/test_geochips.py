import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping

from geochips import ChipConfig, iter_windows, load_labels, make_chips
from geochips.cli import main

CRS = "EPSG:32633"
TRANSFORM = from_origin(500000, 4600000, 10, 10)  # 10 m pixels
W, H = 1000, 700

# objects in pixel coordinates (col0, row0, col1, row1)
OBJECTS = {
    "inside": ("building", (100, 100, 140, 120)),  # fully inside the first 512 chip
    "edge": ("building", (500, 200, 540, 240)),  # straddles the column-512 boundary
    "field": ("field", (700, 400, 900, 650)),
}


def pixel_box_to_lonlat(col0, row0, col1, row1):
    x0, y0 = TRANSFORM @ (col0, row0)
    x1, y1 = TRANSFORM @ (col1, row1)
    return transform_geom(CRS, "EPSG:4326", mapping(box(x0, y1, x1, y0)))


@pytest.fixture
def scene(tmp_path):
    raster = tmp_path / "scene.tif"
    data = np.zeros((3, H, W), dtype="uint16")
    data += np.linspace(100, 4000, W, dtype="uint16")[None, None, :]
    data[:, :, 900:] = 0  # nodata strip on the right
    with rasterio.open(raster, "w", driver="GTiff", width=W, height=H, count=3, dtype="uint16",
                       crs=CRS, transform=TRANSFORM, nodata=0) as dst:
        dst.write(data)
    features = [
        {"type": "Feature", "properties": {"class": cls, "name": name}, "geometry": pixel_box_to_lonlat(*px)}
        for name, (cls, px) in OBJECTS.items()
    ]
    labels = tmp_path / "labels.geojson"
    labels.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return raster, labels


def read_yolo(out, split="train"):
    return {p.stem: [list(map(float, line.split())) for line in p.read_text().splitlines()]
            for p in sorted((out / "labels" / split).glob("*.txt"))}


def test_windows_cover_image_and_end_on_edges():
    wins = list(iter_windows(1000, 700, 512, 0))
    assert [(w.col_off, w.row_off) for w in wins] == [(0, 0), (488, 0), (0, 188), (488, 188)]
    assert all(w.width == 512 and w.height == 512 for w in wins)
    wins = list(iter_windows(1000, 700, 512, 256))
    assert sorted({w.col_off for w in wins}) == [0, 256, 488]
    assert [(w.width, w.height) for w in iter_windows(300, 200, 512)] == [(300, 200)]
    with pytest.raises(ValueError):
        list(iter_windows(100, 100, 64, 64))


def test_labels_reprojected_to_pixels(scene):
    from geochips.core import labels_to_pixels

    raster, labels_path = scene
    labels, crs = load_labels(labels_path)
    assert crs == rasterio.crs.CRS.from_epsg(4326)
    with rasterio.open(raster) as src:
        px = labels_to_pixels(labels, crs, src)
    for lab, (_, expected) in zip(px, OBJECTS.values()):
        assert np.allclose(lab.geometry.bounds, expected, atol=1e-3)


def test_yolo_boxes(scene, tmp_path):
    raster, labels = scene
    out = tmp_path / "out"
    stats = make_chips(raster, labels, out, "yolo", ChipConfig(size=512))
    assert stats["classes"] == ["building", "field"]
    assert stats["chips"]["train"] == 4
    rows = read_yolo(out)
    # "inside" in chip (0, 0): box 100..140 x 100..120 of 512
    # and the left 12 px of "edge", clipped at column 512
    assert rows["scene_0_0"] == [
        pytest.approx([0, 120 / 512, 110 / 512, 40 / 512, 20 / 512], abs=1e-5),
        pytest.approx([0, 506 / 512, 220 / 512, 12 / 512, 40 / 512], abs=1e-5),
    ]
    # the whole of "edge" is inside chip (0, 488)
    edge = [r for r in rows["scene_0_488"] if r[0] == 0]
    assert edge == [pytest.approx([0, (520 - 488) / 512, 220 / 512, 40 / 512, 40 / 512], abs=1e-5)]
    data_yaml = (out / "data.yaml").read_text()
    assert '0: "building"' in data_yaml and "val: images/train" in data_yaml


def test_min_visible_drops_cut_objects(scene, tmp_path):
    raster, labels = scene
    out = tmp_path / "out"
    make_chips(raster, labels, out, "yolo", ChipConfig(size=512, min_visible=0.9))
    rows = read_yolo(out)
    # "inside" plus the 12/40 of "edge" that falls left of column 512, which is now dropped
    assert len(rows["scene_0_0"]) == 1


def test_yolo_seg_polygons(scene, tmp_path):
    raster, labels = scene
    out = tmp_path / "out"
    make_chips(raster, labels, out, "yolo-seg", ChipConfig(size=512))
    line = read_yolo(out)["scene_0_0"][0]
    assert line[0] == 0 and len(line) == 1 + 2 * 4
    xy = np.array(line[1:]).reshape(-1, 2) * 512
    assert np.allclose(sorted(set(np.round(xy[:, 0], 3))), [100, 140], atol=1e-2)


def test_coco(scene, tmp_path):
    raster, labels = scene
    out = tmp_path / "out"
    make_chips(raster, labels, out, "coco", ChipConfig(size=512, skip_empty=True))
    coco = json.loads((out / "annotations" / "instances_train.json").read_text())
    assert [c["name"] for c in coco["categories"]] == ["building", "field"]
    assert len({a["id"] for a in coco["annotations"]}) == len(coco["annotations"])
    first = next(a for a in coco["annotations"] if a["image_id"] == 1 and a["area"] == 800)
    assert first["bbox"] == pytest.approx([100, 100, 40, 20], abs=1e-2)
    for img in coco["images"]:
        assert (out / "images" / "train" / img["file_name"]).exists()


def test_nodata_and_empty_filters(scene, tmp_path):
    raster, labels = scene
    # 512-wide chips at column 488 contain 100 nodata columns (~20 %)
    stats = make_chips(raster, labels, tmp_path / "a", "yolo", ChipConfig(size=512, max_nodata=0.1))
    assert stats["chips"]["train"] == 2
    stats = make_chips(raster, labels, tmp_path / "b", "yolo", ChipConfig(size=256, skip_empty=True))
    assert stats["chips"]["train"] == sum(1 for v in read_yolo(tmp_path / "b").values() if v)


def test_uint16_is_stretched_to_uint8_png(scene, tmp_path):
    from PIL import Image

    raster, labels = scene
    make_chips(raster, labels, tmp_path / "out", "yolo", ChipConfig(size=512))
    img = np.asarray(Image.open(tmp_path / "out" / "images" / "train" / "scene_0_0.png"))
    assert img.dtype == np.uint8 and img.shape == (512, 512, 3)
    assert img[..., 0].min() == 0 and img[..., 0].max() > 100


def test_tif_chips_keep_georeferencing(scene, tmp_path):
    raster, labels = scene
    make_chips(raster, labels, tmp_path / "out", "yolo", ChipConfig(size=512, bands=[3, 2]), image_format="tif")
    with rasterio.open(tmp_path / "out" / "images" / "train" / "scene_188_488.tif") as chip:
        assert chip.crs.to_epsg() == 32633 and chip.count == 2 and chip.dtypes[0] == "uint16"
        assert chip.transform == TRANSFORM @ rasterio.Affine.translation(488, 188)


def test_split_by_image_keeps_raster_together(scene, tmp_path):
    raster, labels = scene
    stats = make_chips(raster, labels, tmp_path / "out", "yolo",
                       ChipConfig(size=256, val_fraction=1.0, split_by="image"))
    assert stats["chips"]["train"] == 0 and stats["chips"]["val"] > 0


def test_cli(scene, tmp_path, capsys):
    raster, labels = scene
    out = tmp_path / "out"
    code = main([str(raster), "-l", str(labels), "-o", str(out), "-s", "256", "--overlap", "32",
                 "--val", "0.3", "--seed", "1", "--preview", "2", "-f", "coco"])
    assert code == 0
    assert "chips" in capsys.readouterr().out
    assert len(list((out / "preview").glob("*.png"))) == 2


def read_voc(path):
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    size = tuple(int(root.find(f"size/{t}").text) for t in ("width", "height", "depth"))
    objects = [
        (o.find("name").text, int(o.find("truncated").text), tuple(int(o.find(f"bndbox/{t}").text) for t in ("xmin", "ymin", "xmax", "ymax")))
        for o in root.iter("object")
    ]
    return root.find("filename").text, size, objects


def test_voc(scene, tmp_path):
    raster, labels = scene
    out = tmp_path / "out"
    make_chips(raster, labels, out, "voc", ChipConfig(size=512, val_fraction=0.5, seed=1), image_format="jpg")
    names = {s: (out / "ImageSets" / "Main" / f"{s}.txt").read_text().split() for s in ("train", "val")}
    assert sorted(names["train"] + names["val"]) == ["scene_0_0", "scene_0_488", "scene_188_0", "scene_188_488"]
    assert names["train"] and names["val"]
    assert (out / "labels.txt").read_text().split() == ["building", "field"]
    for name in names["train"] + names["val"]:
        assert (out / "JPEGImages" / f"{name}.jpg").exists()
    filename, size, objects = read_voc(out / "Annotations" / "scene_0_0.xml")
    assert filename == "scene_0_0.jpg" and size == (512, 512, 3)
    # "inside" covers pixels 100..139 x 100..119; "edge" is cut at column 512
    assert objects == [("building", 0, (101, 101, 140, 120)), ("building", 1, (501, 201, 512, 240))]
    _, _, objects = read_voc(out / "Annotations" / "scene_0_488.xml")
    assert ("building", 0, (13, 201, 52, 240)) in objects


def test_voc_depth_follows_the_saved_image(scene, tmp_path):
    raster, labels = scene
    make_chips(raster, labels, tmp_path / "a", "voc", ChipConfig(size=512, bands=[1]))
    assert read_voc(tmp_path / "a" / "Annotations" / "scene_0_0.xml")[1] == (512, 512, 1)
    make_chips(raster, labels, tmp_path / "b", "voc", ChipConfig(size=512, bands=[3, 2]), image_format="tif")
    assert read_voc(tmp_path / "b" / "Annotations" / "scene_0_0.xml")[1] == (512, 512, 2)


def test_truncated_flag_marks_objects_cut_by_the_chip():
    from rasterio.windows import Window
    from shapely import STRtree
    from shapely.geometry import box as sbox

    from geochips.core import Label, clip_labels

    labels = [Label("a", sbox(10, 10, 20, 20)), Label("b", sbox(60, 10, 80, 20))]
    clipped = clip_labels(labels, STRtree([lab.geometry for lab in labels]), Window(0, 0, 64, 64), 0.0, 1.0)
    assert [(lab.cls, lab.truncated) for lab in clipped] == [("a", False), ("b", True)]
