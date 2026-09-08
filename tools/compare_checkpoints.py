import torch
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

images = [
r"C:\Users\suman\OneDrive\Desktop\19b4526c-1d0d-4c2d-81fb-bcf098ba4ee7.png",
r"C:\Users\suman\OneDrive\Desktop\ChatGPT Image Sep 7, 2026, 01_06_20 AM.png"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for ckpt in ["models/capcheck-ntire-full", "models/capcheck-ntire-hardcase"]:
    processor = AutoImageProcessor.from_pretrained(ckpt, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(ckpt, local_files_only=True).to(device).eval()

    print(f"\n=== {ckpt} ===")

    for path in images:
        image = Image.open(path).convert("RGB")
        x = processor(images=image, return_tensors="pt")
        x = {k: v.to(device) for k, v in x.items()}

        with torch.no_grad():
            p = torch.softmax(model(**x).logits, dim=-1)[0]

        print(Path(path).name)
        print(f"human: {p[0].item()*100:.2f}% | AI: {p[1].item()*100:.2f}%")
