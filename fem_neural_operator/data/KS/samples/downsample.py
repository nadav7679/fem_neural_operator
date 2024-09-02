from firedrake import *
import torch


def downsample(data, mesh1, mesh2, N, T):
    her1 = FunctionSpace(mesh1, "HER", 3)
    her2 = FunctionSpace(mesh2, "HER", 3) 
    cg1 = FunctionSpace(mesh1, "CG", 1)
    cg2 = FunctionSpace(mesh2, "CG", 1)
    
    downsampled_data = np.zeros((1200, 2, 2 * N), dtype=np.float64) # 2*N bc HER has twice dof
    for i, datapoint in enumerate(data):
        for j in range(2): # Once for input (j=0), once for output (j=1) 
            func_her1 = Function(her1, val=data[i, j, :])
            func_her2 = Function(her2)
            func_cg2 = Function(cg2)
            func_cg1 = Function(cg1)
            
            func_cg1.interpolate(func_her1)
            func_cg2.interpolate(func_cg1)
            func_her2.project(func_cg2)
        
            downsampled_data[i, j, :] = func_her2.dat.data[:]
            
    
    torch.save(torch.Tensor(downsampled_data), f"N{N}_HER_nu0029_T{T[0] + T[-1]}_samples1200.pt")


if __name__ == "__main__":
    with CheckpointFile(f"/home/clustor2/ma/n/np923/fem_neural_operator/fem_neural_operator/data/KS/meshes/N4096.h5", "r") as f:
        mesh1 = f.load_mesh()

    for T in ["0.6", "0.7", "0.8", "0.9", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5"]:
        data = torch.load(f"/home/clustor2/ma/n/np923/fem_neural_operator/fem_neural_operator/data/KS/samples/N4096_HER_nu0029_T{T[0] + T[-1]}_samples1200.pt").numpy()
        for N in [64, 128, 256, 512, 1024, 2048]:
            with CheckpointFile(f"/home/clustor2/ma/n/np923/fem_neural_operator/fem_neural_operator/data/KS/meshes/N{N}.h5", "r") as f:
                mesh2 = f.load_mesh()
            print(f"Working on T={T}, N={N}")
            downsample(data, mesh1, mesh2, N, T)


