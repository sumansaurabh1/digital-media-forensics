import sys, json, zipfile, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.detectors.ai_detector.capcheck_detector import CapCheckDetector
from backend.detectors.ai_detector.evaluation import _metrics

ZIP = Path(r"D:\ai-forensics-datasets\NTIRE-2026\shard_0.zip")
N = 100
detector = CapCheckDetector()
results = []

with zipfile.ZipFile(ZIP) as z:
    labels = {}
    for line in z.read("shard_0/labels.csv").decode("utf-8").splitlines()[1:]:
        _, name, label = line.split(",")
        labels[name] = int(label)

    names = [n for n in z.namelist() if n.lower().endswith((".jpg",".jpeg",".png",".webp"))]

    for name in names[:N]:
        image_name = Path(name).name
        data = z.read(name)
        with tempfile.NamedTemporaryFile(suffix=Path(name).suffix, delete=False) as f:
            f.write(data)
            path = Path(f.name)
        try:
            r = detector.detect(path)
            results.append((labels[image_name], float(r["ai_score"])))
        finally:
            path.unlink(missing_ok=True)

truth = [x[0] for x in results]
scores = [x[1] for x in results]
pred = [int(x >= 0.5) for x in scores]

print(json.dumps(_metrics(truth, pred, scores, 0.5), indent=2))
