import torch
from model import CNFModel

device = "cuda" if torch.cuda.is_available() else "cpu"
dim = 28 * 28
model = CNFModel(dim, hidden_dims=[64,64, 64], device=device).to(device)
model.load_state_dict(torch.load("cnf_mnist.pth", map_location=device))  # 저장된 모델 불러오기

import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    samples = model.sample(16).cpu().view(16, 1, 28, 28)

rows = [torch.cat(list(samples[i*4:(i+1)*4]), dim=2) for i in range(4)]
grid = torch.cat(rows, dim=1)
plt.imshow(grid.squeeze(), cmap="gray")
plt.axis("off")
plt.show()