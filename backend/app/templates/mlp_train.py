# /// script
# dependencies = ["torch>=2.2"]
# ///
import torch

from torch import nn

data: DATASET
hidden: int = 16
epochs: int = 80
lr: float = 0.08
seed: int = 7

torch.manual_seed(seed)
features, labels = data
num_features = int(features.shape[1])
num_classes = int(labels.max().item()) + 1

model = nn.Sequential(
    nn.Linear(num_features, hidden),
    nn.ReLU(),
    nn.Linear(hidden, num_classes),
)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()
losses = []
for _ in range(epochs):
    optimizer.zero_grad()
    logits = model(features)
    loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()
    losses.append(float(loss.item()))
history = torch.tensor(losses)

return model: ML_MODEL, history: TENSOR
