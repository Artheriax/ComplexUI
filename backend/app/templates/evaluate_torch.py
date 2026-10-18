# /// script
# dependencies = ["torch>=2.2"]
# ///
import torch

from torch import nn

model: ML_MODEL
data: DATASET

features, labels = data
model.eval()
with torch.no_grad():
    logits = model(features)
    loss = float(nn.CrossEntropyLoss()(logits, labels).item())
predictions = logits.argmax(dim=1)

metrics = {
    "accuracy": float((predictions == labels).float().mean().item()),
    "loss": loss,
    "n_test_samples": int(labels.shape[0]),
}

return metrics: METRICS
