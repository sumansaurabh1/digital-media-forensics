import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.detectors.ai_detector.training import NTIREZipDataset, TrainingConfig, fine_tune_ntire, load_capcheck_training_components

dataset = NTIREZipDataset(r"D:\ai-forensics-datasets\NTIRE-2026")
processor, model, mapping = load_capcheck_training_components()

result = fine_tune_ntire(
    dataset, processor, model, mapping,
    TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-5, validation_ratio=0.2, num_workers=0, checkpoint_dir=Path("models/capcheck-ntire-full"))
)

print(result["history"])
print("SAVED:", result["checkpoint_dir"])
