import torch

data = torch.zeros((1200, 2, 8192), dtype=torch.float64)

for i in range(1, 31):
    data[(i-1) * 40: i * 40] = torch.load(f"N8192_nu0.01_samples40_batch{i}.pt")
    
torch.save(data, "N8192_nu001_T1_samples1200.pt")