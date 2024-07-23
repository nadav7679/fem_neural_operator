import torch

data = torch.zeros((1200, 2, 8192), dtype=torch.float64)

data[0: 600] = torch.load(f"N4096_nu0.029_T0.1_samples600_batch1.pt")
data[600: 1200] = torch.load(f"N4096_nu0.029_T0.1_samples600_batch2.pt")
    
torch.save(data, "N4096_HER_nu0029_T01_samples1200.pt")