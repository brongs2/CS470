import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    samples = model.sample(16).cpu().view(16, 1, 28, 28)

rows = [torch.cat(list(samples[i*4:(i+1)*4]), dim=2) for i in range(4)]
grid = torch.cat(rows, dim=1)
plt.imshow(grid.squeeze(), cmap="gray")
plt.axis("off")
plt.show()