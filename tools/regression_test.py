import sys, random, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.detectors.ai_detector.training import NTIREZipDataset
from transformers import AutoImageProcessor, AutoModelForImageClassification

NTIRE = r"D:\ai-forensics-datasets\NTIRE-2026"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)

ds = NTIREZipDataset(NTIRE)
real = [i for i, s in enumerate(ds.samples) if s.label == 0]
ai = [i for i, s in enumerate(ds.samples) if s.label == 1]
random.shuffle(real)
random.shuffle(ai)
tests = [(i, 0) for i in real[:20]] + [(i, 1) for i in ai[:20]]

for ckpt in ["models/capcheck-ntire-full", "models/capcheck-ntire-hardcase"]:
    processor = AutoImageProcessor.from_pretrained(ckpt, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(ckpt, local_files_only=True).to(device).eval()
    correct = 0

    print(f"\n=== {ckpt} ===")

    for i, label in tests:
        image, _ = ds[i]
        x = processor(images=image, return_tensors="pt")
        x = {k: v.to(device) for k, v in x.items()}

        with torch.no_grad():
            pred = model(**x).logits.argmax(dim=-1).item()

        correct += pred == label

    print(f"Accuracy: {correct}/40 = {correct / 40 * 100:.2f}%")
