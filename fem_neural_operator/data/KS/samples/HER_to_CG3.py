from firedrake import *
import torch


def HER_to_CG3(data, her, cg3, T):
    cg3_data = np.zeros((1200, 2, cg3.dof_count), dtype=np.float64) # 3*N bc CG3 has thrice dof
    for i, datapoint in enumerate(data):
        for j in range(2): # Once for input (j=0), once for output (j=1) 
            func_her = Function(her, val=data[i, j, :])
            func_cg3 = Function(cg3)
            
            func_cg3.interpolate(func_her)
            
            cg3_data[i, j, :] = func_cg3.dat.data[:]
            
    
    torch.save(torch.Tensor(cg3_data), f"N4096_CG3_nu0029_T{T}_samples1200.pt")


if __name__ == "__main__":
    with CheckpointFile(f"/home/clustor/ma/n/np923/fem_neural_operator/fem_neural_operator/data/KS/meshes/N4096.h5", "r") as f:
        mesh = f.load_mesh()
    
    her = FunctionSpace(mesh, "HER", 3)
    cg3 = FunctionSpace(mesh, "CG", 3)
    
    T_domain = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"]
    for T in T_domain:
        print(f"Working on T={T}")

        data = torch.load(f"/home/clustor/ma/n/np923/fem_neural_operator/fem_neural_operator/data/KS/samples/N4096_HER_nu0029_T{T}_samples1200.pt").numpy()
        HER_to_CG3(data, her, cg3, T)


