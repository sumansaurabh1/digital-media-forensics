@'
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODELS = [
    "models/capcheck-ntire-full",
    "models/capcheck-unified",
    "models/capcheck-unified-morph",
]

IMAGES = [
    r"C:\Users\suman\Downloads\ChatGPT Image Sep 7, 2026, 11_22_22 AM.png",
    r"C:\Users\suman\OneDrive\Desktop\ChatGPT Image Sep 7, 2026, 01_06_20 AM.png",
    r"C:\Users\suman\OneDrive\Desktop\19b4526c-1d0d-4c2d-81fb-bcf098ba4ee7.png",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model_path in MODELS:
    if not Path(model_path).exists():
        print(f"\nSKIP: {model_path}")
        continue

    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(model_path, local_files_only=True).to(device)
    model.eval()

    print(f"\n=== {model_path} ===")

    for image_path in IMAGES:
        image = Image.open(image_path).convert("RGB")
        x = processor(images=image, return_tensors="pt")
        x = {k: v.to(device) for k, v in x.items()}

        with torch.inference_mode():
            probs = torch.softmax(model(**x).logits, dim=-1)[0]

        labels = model.config.id2label
        scores = {labels[i]: round(float(probs[i]), 4) for i in range(len(probs))}
        print(Path(image_path).name, scores)
'@ | Set-Content eval_models.py