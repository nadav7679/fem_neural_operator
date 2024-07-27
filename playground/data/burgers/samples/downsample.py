import firedrake as fd
import numpy as np
import torch

import matplotlib.pyplot as plt


def downsample(data_fs, data, N):
    print(f"../meshes/N{N}.h5")
    with fd.CheckpointFile(f"../meshes/N{N}.h5", "r") as file:
        mesh = file.load_mesh()
        fs = fd.FunctionSpace(mesh, "CG", 1)

    downsampled_data = np.zeros((1200, 2, N), dtype=np.float64)
    for i, datapoint in enumerate(data):
        a = fd.Function(data_fs, val=datapoint[0, :])
        u = fd.Function(data_fs, val=datapoint[1, :])

        downsampled_data[i, 0, :] = fd.Function(fs).interpolate(a).dat.data[:]
        downsampled_data[i, 1, :] = fd.Function(fs).interpolate(u).dat.data[:]

    torch.save(torch.Tensor(downsampled_data), f"N{N}_nu001_T1_samples1200.pt")


def check_downsample(fs1, data1, fs2, data2, index=100):
    fig1, axes1 = plt.subplots()
    fig2, axes2 = plt.subplots()

    fd.plot(fd.Function(fs1, val=data1[index, 0, :], name="FS1 - a"), axes=axes1)
    fd.plot(fd.Function(fs2, val=data2[index, 0, :], name="FS2 - a"), axes=axes1)
    fd.plot(fd.Function(fs1, val=data1[index, 1, :], name="FS1 - u"), axes=axes2)
    fd.plot(fd.Function(fs2, val=data2[index, 1, :], name="FS2 - u"), axes=axes2)

    axes1.legend()
    axes1.grid()
    axes2.legend()
    axes2.grid()

    plt.show()


if __name__ == "__main__":
    data = torch.load("N8192_nu001_T1_samples1200.pt").numpy()
    with fd.CheckpointFile(f"../meshes/N8192.h5", "r") as file:
        data_fs = fd.FunctionSpace(file.load_mesh(), "CG", 1)

    # for N in [1024, 2048]:
    #     downsample(data_fs, data, N)

    N1 = 8192
    N2 = 512

    data1 = torch.load(f"N{N1}_nu001_T1_samples1200.pt").numpy()
    data2 = torch.load(f"N{N2}_nu001_T1_samples1200.pt").numpy()

    with fd.CheckpointFile(f"../meshes/N{N1}.h5", "r") as file:
        fs1 = fd.FunctionSpace(file.load_mesh(), "CG", 1)

    with fd.CheckpointFile(f"../meshes/N{N2}.h5", "r") as file:
        fs2 = fd.FunctionSpace(file.load_mesh(), "CG", 1)

    check_downsample(fs1, data1, fs2, data2)
