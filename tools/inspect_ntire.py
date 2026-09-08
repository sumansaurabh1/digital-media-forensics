import csv
import io
import json
import zipfile
from pathlib import Path
from collections import Counter
from PIL import Image

ROOT = Path(r"D:\ai-forensics-datasets\NTIRE-2026")
OUT = Path(r"D:\ai-forensics-datasets\manifests")
OUT.mkdir(parents=True, exist_ok=True)

summary = {"total": 0, "real": 0, "ai": 0, "shards": {}}

for zip_path in sorted(ROOT.glob("shard_*.zip")):
    print(f"Inspecting {zip_path.name}...")

    with zipfile.ZipFile(zip_path) as z:
        label_file = next(x for x in z.namelist() if x.endswith("labels.csv"))
        rows = list(csv.DictReader(
            io.TextIOWrapper(z.open(label_file), encoding="utf-8")
        ))

        counts = Counter(r["label"] for r in rows)
        widths = Counter()
        heights = Counter()
        sizes = []

        for i, row in enumerate(rows):
            image_path = f"{zip_path.stem}/images/{row['image_name']}"
            sizes.append(z.getinfo(image_path).file_size)

            if i < 1000:
                with z.open(image_path) as f:
                    with Image.open(f) as im:
                        widths[im.width] += 1
                        heights[im.height] += 1

        summary["shards"][zip_path.stem] = {
            "images": len(rows),
            "real": counts["0"],
            "ai": counts["1"],
            "sampled_for_dimensions": min(1000, len(rows)),
            "common_widths": widths.most_common(10),
            "common_heights": heights.most_common(10),
            "avg_file_size_mb": round(sum(sizes) / len(sizes) / 1024 / 1024, 3),
            "min_file_size_kb": round(min(sizes) / 1024, 2),
            "max_file_size_mb": round(max(sizes) / 1024 / 1024, 2)
        }

        summary["total"] += len(rows)
        summary["real"] += counts["0"]
        summary["ai"] += counts["1"]

print(json.dumps(summary, indent=2))

(OUT / "ntire_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print(f"\nSaved: {OUT / 'ntire_summary.json'}")
