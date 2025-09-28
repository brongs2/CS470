import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from model import CNFModel

torch.backends.cuda.matmul.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"
dim = 28*28
model = CNFModel(dim, hidden_dims=[64,64,64], device=device).to(device)

# Optimizer
opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

# Subsample 20% of MNIST for training
full_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
subset_size = int(0.2 * len(full_dataset))
indices = np.random.choice(len(full_dataset), size=subset_size, replace=False)
train_dataset = Subset(full_dataset, indices)
train_loader = DataLoader(
    train_dataset,
    batch_size=8, shuffle=True, drop_last=True  # 더 작은 미니배치로 OOM 회피
)

# Gradient Accumulation: effective batch = 8 * 4 = 32
accum_steps = 4

print("🚀 Training start...")
model.train()
for epoch in range(3):
    running = 0.0
    opt.zero_grad(set_to_none=True)
    for i, (x, _) in enumerate(train_loader):
        x = x.to(device)
        logp = model(x)                 # log p(x)
        loss = -logp.mean() / accum_steps
        loss.backward()
        running += loss.item() * accum_steps

        if (i + 1) % accum_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        if i % 200 == 0:
            print(f"Epoch {epoch} Iter {i} Loss {(running / (i+1)):.4f}")

print("✅ Training finished!")
torch.save(model.state_dict(), "cnf_mnist.pth")
print("💾 Model saved to cnf_mnist.pth")