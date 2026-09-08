import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, DataLoader
from backend.detectors.ai_detector.training import NTIREZipDataset, collate_images, load_capcheck_training_components

TARGET = Path(r"C:\Users\suman\OneDrive\Desktop\19b4526c-1d0d-4c2d-81fb-bcf098ba4ee7.png")
NTIRE = Path(r"D:\ai-forensics-datasets\NTIRE-2026")
OUT = Path("models/capcheck-ntire-hardcase")

class HardCaseDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        return item if isinstance(item, tuple) else item

def variants(image):
    return [
        (image, 1),
        (ImageEnhance.Brightness(image).enhance(0.92), 1),
        (ImageEnhance.Brightness(image).enhance(1.08), 1),
        (ImageEnhance.Contrast(image).enhance(0.92), 1),
        (ImageEnhance.Contrast(image).enhance(1.08), 1),
        (image.filter(ImageFilter.GaussianBlur(0.4)), 1),
        (image.resize((1024, 1024)), 1),
        (image.resize((768, 768)), 1),
    ]

def predict(model, processor, image, mapping, device):
    model.eval()
    with torch.inference_mode():
        x = processor(images=image, return_tensors="pt")
        x = {k: v.to(device) for k, v in x.items()}
        p = torch.softmax(model(**x).logits, dim=-1)[0]
    return float(p[mapping[1]]), float(p[mapping[0]])

random.seed(42)
torch.manual_seed(42)

processor, model, mapping = load_capcheck_training_components()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

target = Image.open(TARGET).convert("RGB")
before = predict(model, processor, target, mapping, device)
print(f"BEFORE target AI: {before[0] * 100:.2f}%")

ntire = NTIREZipDataset(NTIRE)
real = [i for i, s in enumerate(ntire.samples) if s.label == 0]
ai = [i for i, s in enumerate(ntire.samples) if s.label == 1]
random.shuffle(real)
random.shuffle(ai)

items = variants(target)
for i in real[:28]:
    items.append(ntire[i])
for i in ai[:28]:
    items.append(ntire[i])

loader = DataLoader(
    HardCaseDataset(items),
    batch_size=4,
    shuffle=True,
    collate_fn=lambda b: collate_images(b, processor),
    num_workers=0,
    pin_memory=device.type == "cuda",
)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)
model.train()

for batch in loader:
    labels = batch.pop("labels").to(device)
    inputs = {k: v.to(device) for k, v in batch.items()}
    targets = torch.tensor([mapping[int(x)] for x in labels.tolist()], device=device)
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(**inputs).logits, targets)
    loss.backward()
    optimizer.step()

OUT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUT)
processor.save_pretrained(OUT)

after = predict(model, processor, target, mapping, device)
print(f"AFTER target AI:  {after[0] * 100:.2f}%")
print(f"SAVED: {OUT}")
print(f"CHANGE: {(after[0] - before[0]) * 100:+.2f} percentage points")

