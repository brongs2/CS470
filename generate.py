import torch
import matplotlib.pyplot as plt
from model import CNFModel

device = "cuda" if torch.cuda.is_available() else "cpu"
dim = 28 * 28

# 1. 모델 정의 (오리지널 버전, method 파라미터 없음)
model = CNFModel(dim, hidden_dims=[64, 64, 64], device=device).to(device)

# 2. 학습된 가중치 불러오기
state = torch.load("cnf_mnist.pth", map_location=device)
model.load_state_dict(state, strict=True)

# 3. 샘플링
model.eval()
with torch.no_grad():
    samples = model.sample(16).cpu().view(16, 1, 28, 28)

# 4. 시각화
rows = [torch.cat(list(samples[i*4:(i+1)*4]), dim=2) for i in range(4)]
grid = torch.cat(rows, dim=1)
plt.imshow(grid.squeeze(), cmap="gray")
plt.axis("off")
plt.show()