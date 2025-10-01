import torch
from model import CNFModel

device = "cuda" if torch.cuda.is_available() else "cpu"
dim = 28 * 28
model = CNFModel(dim, hidden_dims=[128,128,128], device=device, method="rk4", n_steps=64).to(device)
state = torch.load("cnf_mnist.pth", map_location=device)  # 저장된 모델 불러오기
if isinstance(state, dict) and "state_dict" in state and "num_eps" in state:
    model.load_state_dict(state["state_dict"], strict=False)
    model.set_num_eps(state.get("num_eps", 1))
    print(f"Loaded model with num_eps = {state.get('num_eps', 1)}")
else:
    model.load_state_dict(state, strict=False)

import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    samples = model.sample(16).cpu().view(16, 1, 28, 28)

rows = [torch.cat(list(samples[i*4:(i+1)*4]), dim=2) for i in range(4)]
grid = torch.cat(rows, dim=1)
plt.imshow(grid.squeeze(), cmap="gray")
plt.axis("off")
plt.show()