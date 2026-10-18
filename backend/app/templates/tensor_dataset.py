# /// script
# dependencies = ["torch>=2.2"]
# ///
import torch

samples: int = 600
seed: int = 7
noise: float = 0.35
device: str = "cpu"

generator = torch.Generator().manual_seed(seed)
half = samples // 2
blob_a = torch.randn(half, 2, generator=generator) + 1.6
blob_b = torch.randn(samples - half, 2, generator=generator) - 1.6
features = torch.cat([blob_a, blob_b]) + noise * torch.randn(
    samples, 2, generator=generator
)
labels = torch.cat(
    [torch.zeros(half, dtype=torch.long), torch.ones(samples - half, dtype=torch.long)]
)
data = (features.to(device), labels.to(device))

return data: DATASET
