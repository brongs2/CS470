import argparse
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from model import CNFModel

ALPHA = 0.05  # logit alpha for dequantization

parser = argparse.ArgumentParser()
parser.add_argument("--eps", type=int, default=1, help="Number of Hutchinson epsilons")
args = parser.parse_args()
EPS = args.eps

torch.backends.cuda.matmul.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"
# Reproducibility
torch.manual_seed(0)
np.random.seed(0)

dim = 28*28
model = CNFModel(dim, hidden_dims=[64,64,64], device=device, method="euler", n_steps=64, num_eps=EPS).to(device)

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

def dequant_logit(x, alpha=ALPHA):
    # x: (B, 784) in [0,1]
    u = torch.rand_like(x)
    x = (x * 255.0 + u) / 256.0  # uniform dequantization
    x_ = alpha + (1.0 - 2.0 * alpha) * x
    y = torch.log(x_) - torch.log(1.0 - x_)
    D = x_.size(1)
    ldj_const = torch.log(torch.tensor(1.0 - 2.0 * alpha, device=x.device, dtype=x.dtype)) * D
    ldj = ldj_const - (torch.log(x_) + torch.log(1.0 - x_)).sum(dim=1, keepdim=True)
    return y, ldj

# Gradient Accumulation: effective batch = 8 * 4 = 32
accum_steps = 4

print("🚀 Training start...")
print(f"Using Hutchinson epsilons (K) = {EPS}")
model.train()
for epoch in range(10):
    running = 0.0
    opt.zero_grad(set_to_none=True)
    for i, (x, _) in enumerate(train_loader):
        x = x.to(device)
        # dequantize + logit with analytic log-det
        y, ldj = dequant_logit(x)
        logp = model(y, extra_logdet=ldj)  # log p(x) with preprocessing ldj
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
torch.save({
    "state_dict": model.state_dict(),
    "num_eps": EPS
}, "cnf_mnist.pth")
print("💾 Model saved to cnf_mnist.pth (with num_eps)")