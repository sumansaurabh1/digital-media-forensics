from pathlib import Path
import json, random
import torch
from PIL import Image
from torch import nn
from transformers import AutoImageProcessor, AutoModelForImageClassification

MORPH_ROOT = Path(r"D:\ai-forensics-datasets\DiM-FRLL-Morphs\morphs")
FRLL_ROOT = Path(r"D:\ai-forensics-datasets\FRLL")
BASE = Path("models/capcheck-unified")
OUT = Path("models/capcheck-unified-morph")
BATCH = 32

morphs = list(MORPH_ROOT.rglob("*.png"))
reals = [p for p in FRLL_ROOT.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}]

if len(morphs) < 3000 or not reals:
    raise RuntimeError(f"Found {len(morphs)} morphs and {len(reals)} real images. Check FRLL path.")

processor = AutoImageProcessor.from_pretrained(BASE, local_files_only=True)
model = AutoModelForImageClassification.from_pretrained(BASE, local_files_only=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for p in model.parameters():
    p.requires_grad = False

head = [p for p in model.classifier.parameters()]
for p in head:
    p.requires_grad = True

optimizer = torch.optim.AdamW(head, lr=5e-5)
samples = [(p, 1) for p in morphs] + [(random.choice(reals), 0) for _ in morphs]
random.shuffle(samples)

model.train()
for start in range(0, len(samples), BATCH):
    batch = samples[start:start+BATCH]
    images = [Image.open(p).convert("RGB") for p, _ in batch]
    labels = torch.tensor([y for _, y in batch], device=device)
    x = processor(images=images, return_tensors="pt")
    x = {k: v.to(device) for k, v in x.items()}
    optimizer.zero_grad()
    loss = nn.functional.cross_entropy(model(**x).logits, labels)
    loss.backward()
    optimizer.step()

    done = min(start + BATCH, len(samples))
    if done % 320 < BATCH:
        print(f"{done}/{len(samples)} | loss={loss.item():.4f}", flush=True)

OUT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUT)
processor.save_pretrained(OUT)
(OUT / "training_metadata.json").write_text(json.dumps({
    "dataset": "DiM-FRLL-Morphs",
    "morph_images": len(morphs),
    "real_images_used": len(samples) - len(morphs),
    "base_model": str(BASE),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
}, indent=2))

print(f"DONE: {len(samples)} samples -> {OUT}", flush=True)
