import torch

for T in ["0.6", "0.7", "0.8", "0.9", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5"]:
    data = torch.zeros((1200, 2, 8192), dtype=torch.float64)

    data[0: 600] = torch.load(f"N4096_nu0.029_T{T}_samples600_batch1.pt")
    data[600: 1200] = torch.load(f"N4096_nu0.029_T{T}_samples600_batch2.pt")
        
    torch.save(data, f"N4096_HER_nu0029_T{T[0] + T[-1]}_samples1200.pt")