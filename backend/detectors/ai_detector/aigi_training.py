from pathlib import Path
import io, csv, glob, random, zipfile, json
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch import nn
from transformers import AutoImageProcessor, AutoModelForImageClassification

AIGI = Path(r"D:\ai-forensics-datasets\AIGI-QualityParadox\data")
NTIRE = Path(r"D:\ai-forensics-datasets\NTIRE-2026")
BASE = Path("models/capcheck-unified-morph")
OUT = Path("models/capcheck-unified-aigi")
BATCH = 32
N = 24000

processor = AutoImageProcessor.from_pretrained(BASE, local_files_only=True)
model = AutoModelForImageClassification.from_pretrained(BASE, local_files_only=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for p in model.parameters():
    p.requires_grad = False

head = tuple(model.classifier.parameters())
for p in head:
    p.requires_grad = True

optimizer = torch.optim.AdamW(head, lr=5e-5)

aigi = []
for f in sorted(glob.glob(str(AIGI / "fake-*.parquet"))):
    pf = pq.ParquetFile(f)
    for batch in pf.iter_batches(batch_size=256, columns=["image"]):
        aigi.extend(batch.to_pylist())
        if len(aigi) >= N:
            break
    if len(aigi) >= N:
        break

aigi = aigi[:N]
zips = sorted(NTIRE.glob("*.zip"))
if not zips:
    raise RuntimeError("No NTIRE ZIP files found")

real = []
for zpath in zips:
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.lower().endswith((".jpg",".jpeg",".png",".webp"))]
        for name in names:
            real.append((zpath, name))
            if len(real) >= N:
                break
    if len(real) >= N:
        break

if len(aigi) < N or len(real) < N:
    raise RuntimeError(f"Need {N} each, found {len(aigi)} AIGI and {len(real)} NTIRE real")

samples = [(r, 1) for r in aigi] + [(r, 0) for r in real]
random.shuffle(samples)

model.train()
for start in range(0, len(samples), BATCH):
    batch = samples[start:start+BATCH]
    images = []

    for item, label in batch:
        if label == 1:
            images.append(Image.open(io.BytesIO(item["image"]["bytes"])).convert("RGB"))
        else:
            with zipfile.ZipFile(item[0]) as z:
                images.append(Image.open(io.BytesIO(z.read(item[1]))).convert("RGB"))

    labels = torch.tensor([y for _, y in batch], device=device)
    x = processor(images=images, return_tensors="pt")
    x = {k: v.to(device) for k, v in x.items()}

    optimizer.zero_grad()
    loss = nn.functional.cross_entropy(model(**x).logits, labels)
    loss.backward()
    optimizer.step()

    done = min(start + BATCH, len(samples))
    if done % 512 < BATCH:
        print(f"{done}/{len(samples)} | loss={loss.item():.4f}", flush=True)

OUT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUT)
processor.save_pretrained(OUT)
(OUT / "training_metadata.json").write_text(json.dumps({
    "dataset": "AIGI-QualityParadox + NTIRE-real",
    "aigi_fake": N,
    "ntire_real": N,
    "base_model": str(BASE),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
}, indent=2))

print(f"DONE: {len(samples)} samples -> {OUT}", flush=True)
