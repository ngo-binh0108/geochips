r"""Build docs/demo.jpg from the demo data: the scene with its chip grid, and three labelled chips.

    python examples/download_demo.py examples/data
    geochips examples/data/utrecht_s2.tif -l examples/data/utrecht_landuse.geojson \
        -o out/demo -f yolo-seg -s 256 --overlap 32 --min-area 16 --preview 20 \
        --classes residential,industrial,forest,water
    python examples/make_figure.py
"""

from pathlib import Path

import rasterio
from PIL import Image, ImageDraw

from geochips import iter_windows
from geochips.writers import to_pil

SCENE = Path("examples/data/utrecht_s2.tif")
PREVIEW = Path("out/demo/preview")
CHIPS = ["utrecht_s2_224_448", "utrecht_s2_448_448", "utrecht_s2_448_672"]
OUT = Path("docs/demo.jpg")
SIDE = 600  # height of the scene panel
GAP = 12

with rasterio.open(SCENE) as src:
    scene = to_pil(src.read()).convert("RGB")
    windows = list(iter_windows(src.width, src.height, 256, 32))
scale = SIDE / scene.height
scene = scene.resize((round(scene.width * scale), SIDE), Image.LANCZOS)
draw = ImageDraw.Draw(scene)
for w in windows:
    x0, y0 = w.col_off * scale, w.row_off * scale
    draw.rectangle([x0, y0, x0 + w.width * scale, y0 + w.height * scale], outline=(255, 255, 255), width=1)

tile = (SIDE - 2 * GAP) // 3
canvas = Image.new("RGB", (scene.width + GAP + tile, SIDE), (255, 255, 255))
canvas.paste(scene, (0, 0))
for i, name in enumerate(CHIPS):
    chip = Image.open(PREVIEW / f"{name}.png").resize((tile, tile), Image.NEAREST)
    canvas.paste(chip, (scene.width + GAP, i * (tile + GAP)))
OUT.parent.mkdir(exist_ok=True)
canvas.save(OUT, quality=88, optimize=True)
print(f"wrote {OUT} ({canvas.width}x{canvas.height})")
