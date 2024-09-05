from firedrake import *
import torch


def downsample(data, mesh1, mesh2, N, T):
    cg1 = FunctionSpace(mesh1, "CG", 3)
    cg2 = FunctionSpace(mesh2, "CG", 3)
    
    downsampled_data = np.zeros((1200, 2, cg2.dof_count), dtype=np.float64)
    for i, datapoint in enumerate(data):
        for j in range(2): # Once for input (j=0), once for output (j=1) 
            func_cg1 = Function(cg1, val=data[i, j, :])
            func_cg2 = Function(cg2)
            
            downsampled_data[i, j, :] = func_cg2.interpolate(func_cg1).dat.data[:]
            
    torch.save(torch.Tensor(downsampled_data), f"N{N}_CG3_nu0029_T{T}_samples1200.pt")


if __name__ == "__main__":
    with CheckpointFile(f"/home/clustor/ma/n/np923/fem_neural_operator/fem_neural_operator/data/KS/meshes/N4096.h5", "r") as f:
        mesh1 = f.load_mesh()
        
    T_domain = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"]
    for N in [64, 128, 256, 512, 1024, 2048]:
        with CheckpointFile(f"/home/clustor/ma/n/np923/fem_neural_operator/fem_neural_operator/data/KS/meshes/N{N}.h5", "r") as f:
            mesh2 = f.load_mesh()
                
        for T in T_domain:
            data = torch.load(f"/home/clustor/ma/n/np923/fem_neural_operator/fem_neural_operator/data/KS/samples/N4096_CG3_nu0029_T{T}_samples1200.pt").numpy()   
            print(f"Working on T={T}, N={N}")
            downsample(data, mesh1, mesh2, N, T)


