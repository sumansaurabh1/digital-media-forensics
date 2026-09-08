from pathlib import Path
import io, glob, json
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch import nn
from transformers import AutoImageProcessor, AutoModelForImageClassification
from backend.detectors.ai_detector.training import resolve_class_mapping

ROOT = Path(r"D:\ai-forensics-datasets\Modern-AI-Real\data")
BASE = Path("models/capcheck-ntire-full")
OUT = Path("models/capcheck-unified")

processor = AutoImageProcessor.from_pretrained(BASE, local_files_only=True)
model = AutoModelForImageClassification.from_pretrained(BASE, local_files_only=True)
mapping = resolve_class_mapping(model)
device = torch.device("cuda")
model.to(device)

for p in model.parameters():
    p.requires_grad = False

head = []
for m in model.modules():
    ps = tuple(m.parameters(recurse=False))
    if ps and any(p.ndim > 0 and p.shape[0] == model.config.num_labels for p in ps):
        head.extend(ps)

head = tuple(dict.fromkeys(head))
for p in head:
    p.requires_grad = True

optimizer = torch.optim.AdamW(head, lr=1e-4)
files = sorted(glob.glob(str(ROOT / "train-*.parquet")))
total = 0
model.train()

for shard, file in enumerate(files, 1):
    pf = pq.ParquetFile(file)
    for table in pf.iter_batches(batch_size=32, columns=["image", "label"]):
        rows = table.to_pylist()
        images = [Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB") for r in rows]
        labels = torch.tensor([mapping[1 - int(r["label"])] for r in rows], device=device)

        x = processor(images=images, return_tensors="pt")
        x = {k: v.to(device) for k, v in x.items()}
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(**x).logits, labels)
        loss.backward()
        optimizer.step()

        total += len(rows)
        if total % 500 < 32:
            print(f"{total}/10695 | shard {shard}/{len(files)} | loss={loss.item():.4f}", flush=True)

OUT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUT)
processor.save_pretrained(OUT)
(OUT / "training_metadata.json").write_text(json.dumps({
    "dataset": "Modern-AI-Real",
    "images": total,
    "base_model": str(BASE),
    "label_mapping": mapping,
    "device": torch.cuda.get_device_name(0)
}, indent=2))
print(f"DONE: {total} images -> {OUT}", flush=True)
