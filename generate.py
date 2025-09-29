import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--eps", type=int, default=1, help="Number of Hutchinson epsilons")
args = parser.parse_args()
EPS = args.eps

import torch
import torch.nn as nn
import torch.optim as optim
from model import CNFModel

device = "cuda" if torch.cuda.is_available() else "cpu"
dim = 28 * 28

model = CNFModel(dim, hidden_dims=[64,64,64], device=device, method="rk4", n_steps=64, num_eps=EPS).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ... rest of the training code ...