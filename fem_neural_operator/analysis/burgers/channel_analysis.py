from typing import List

import firedrake as fd
import matplotlib.pyplot as plt
import torch

import numpy as np
from classes import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def average_firedrake_loss(
        models: List[BurgersModel],
        dataset: Dataset
) -> List[float]:
    """
    Calculate the average loss using Firedrake's errornorm for a list of models on a given dataset.

    Args:
        models (List[NeuralOperatorModel]): List of models to evaluate.
        dataset (Dataset): Dataset to evaluate on.

    Returns:
        List[float]: List of average losses for each model.
    """
    targets = dataset[:][1].squeeze(1).detach().cpu().numpy()

    with fd.CheckpointFile(f"data/burgers/meshes/N{models[0].N}.h5", "r") as file:
        function_space = fd.FunctionSpace(file.load_mesh(), "CG", 1)

    losses = []
    for model in models:
        predictions = model.network(dataset[:][0]).squeeze(1).detach().cpu().numpy()

        loss = 0
        for target, predict in zip(targets, predictions):
            target_func = fd.Function(function_space, val=target)
            loss += fd.errornorm(
                target_func,
                fd.Function(function_space, val=predict)
            ) / fd.norm(target_func)

        losses.append(loss / len(targets))

    return losses


def average_coefficient_loss(
        models: List[BurgersModel],
        dataset: Dataset,
) -> List[torch.Tensor]:
    """
    Calculate the average loss using coefficient approximation (i.e. only using PyTorch)
    for a list of models and corresponding datasets.

    Args:
        models (List[NeuralOperatorModel]): List of models to evaluate.
        datasets (List[Dataset]): List of datasets to evaluate on.

    Returns:
        List[torch.Tensor]: List of average losses for each model-dataset pair.
    """
    mean_rel_l2_loss = lambda x, y: torch.mean(torch.norm(x - y, 2, dim=-1) / torch.norm(y, 2, dim=-1))

    losses = []
    with torch.no_grad():
        for model in models:
            prediction = model.network(dataset[:][0])
            losses.append(mean_rel_l2_loss(dataset[:][1], prediction))

    return losses


def train_models(config, D_arr):
    """
    Train models based on the given configuration and list of D values.

    Args:
        config (dict): Configuration dictionary containing model and training parameters.
        D_arr (List[int]): List of D values for training.
    """
    global device

    for D in D_arr:
        config["D"] = D
        model = BurgersModel(config["N"],
                             config["M"],
                             config["D"],
                             config["depth"],
                             config["T"],
                             config["projection_type"],
                             device=device)
        print(f"Training D={config['D']} with param={model.param_num}")
        model.train(f"data/burgers/samples/N{config['N']}_nu001_T{config['T']}_samples1200.pt", config['epoch'],
                    lr=0.001, device=device)


def load_models(config, D_arr):
    """
    Load trained models based on the given configuration and grid resolutions.

    Args:
        config (dict): Configuration dictionary containing model and training parameters.
        D_arr (List[int]): List of channels D for loading models.

    Returns:
        Tuple[List[NeuralOperatorModel], List[Dataset]]: List of loaded models and corresponding datasets.
    """
    global device

    samples = (torch.load(f"data/burgers/samples/N{config['N']}_nu001_T{config['T']}_samples1200.pt")
                .unsqueeze(2).to(device=device, dtype=torch.float32))
    grid = torch.linspace(0, 1, config['N'], device=device)
    dataset = Dataset(samples, grid)

    models = []
    for D in D_arr:
        config["D"] = D
        filename = f"data/burgers/models/CG1/{config['projection_type']}/N{config['N']}/T{config['T']}" \
                   f"/D{config['D']}_M{config['M']}_samples{config['train_samples']}_epoch{config['epoch']}.pt"


        models.append(BurgersModel.load(filename, config["N"], config["T"], device))

    return models, dataset[config["train_samples"]:]


if __name__ == "__main__":
    D_arr = torch.arange(10, 125, 5).to(dtype=torch.int)
    config = {
        "N": 4096,
        "M": 16,
        "depth": 4,
        "T": 1,
        "projection_type": "fourier",
        "train_samples": 1000,
        "epoch": 500,
    }

    plt.figure(figsize=(8, 6))

    M_losses = [
        [np.float64(0.31618138524941863), np.float64(0.3067527238712964), np.float64(0.3049828191955372), np.float64(0.30507074451886873), np.float64(0.3055966282188277), np.float64(0.08860641887423276), np.float64(0.05534237400364196), np.float64(0.07652196382231753), np.float64(0.05617336773407237), np.float64(0.08780041562787766), np.float64(0.027200851631280085), np.float64(0.03581312634646463), np.float64(0.026539677856492618), np.float64(0.024107423789472344), np.float64(0.0371025280336425), np.float64(0.022271291961676702), np.float64(0.02275075978583878), np.float64(0.01813379984660245), np.float64(0.02328675217523544), np.float64(0.019050251343331572), np.float64(0.029248087371501582), np.float64(0.020792053214798892), np.float64(0.020211886926221855)],

        [np.float64(0.012567724347348562), np.float64(0.007997654846815593), np.float64(0.005989417740131884), np.float64(0.005554026511073632), np.float64(0.005572845004196941), np.float64(0.00521213430695857), np.float64(0.00426319147579338), np.float64(0.004394711605588534), np.float64(0.003807392669635637), np.float64(0.003925055681241042), np.float64(0.0031268538756404024), np.float64(0.002985696855720073), np.float64(0.003106027768851254), np.float64(0.0033366184243477153), np.float64(0.002846396534922225), np.float64(0.0024265291130427195), np.float64(0.002789467902607106), np.float64(0.002436996135890529), np.float64(0.00229918823295898), np.float64(0.002056356811245238), np.float64(0.0021732093534887463), np.float64(0.0019985185771563933), np.float64(0.0024511317813639888)],

        [np.float64(0.007594422007659347), np.float64(0.005047075435466557), np.float64(0.003910201845576859), np.float64(0.003199474990259935), np.float64(0.0025600695280403544), np.float64(0.0022187905887955236), np.float64(0.0019715700928706007), np.float64(0.002719010494902609), np.float64(0.001852052460160834), np.float64(0.0020178601958335744), np.float64(0.0017352601591722025), np.float64(0.001963670255814838), np.float64(0.0018372956580899147), np.float64(0.0017938161190377954), np.float64(0.0016708709315201643), np.float64(0.0015690216246878155), np.float64(0.00161413129344174), np.float64(0.0020043966429536663), np.float64(0.002393431735027797), np.float64(0.0015547769742968983), np.float64(0.001529465429318469), np.float64(0.001650695397419605), np.float64(0.0017940141845531857)],

        [np.float64(0.0064113754382022015), np.float64(0.004277912343304202), np.float64(0.004068449553912678), np.float64(0.0030095279124072224), np.float64(0.002448623739323739), np.float64(0.0026262902319002883), np.float64(0.0019941483901672976), np.float64(0.002191290194051698), np.float64(0.0023697006586252283), np.float64(0.0016832435821397818), np.float64(0.0020023355323889563), np.float64(0.0020140072934350913), np.float64(0.0023998832264514034), np.float64(0.0016049774444750885), np.float64(0.0018164036531539529), np.float64(0.0017094691178232036), np.float64(0.0014335898977479494), np.float64(0.0013809527182115458), np.float64(0.0011579846207031956), np.float64(0.001876237981709348), np.float64(0.0015217324042380528), np.float64(0.0013543253272155626), np.float64(0.0021896381161555798)],

        [np.float64(0.00558506591301818), np.float64(0.004645021238051822), np.float64(0.0032336888901894527), np.float64(0.0023516116580210144), np.float64(0.0023892816793915582), np.float64(0.002072068637454569), np.float64(0.002075934991648139), np.float64(0.0018047376096585893), np.float64(0.0018117178739733553), np.float64(0.0015962663926705797), np.float64(0.0014961053856411182), np.float64(0.0015195833547301334), np.float64(0.0019317393464846166), np.float64(0.001470163820696801), np.float64(0.0013341583506952081), np.float64(0.001590667270808142), np.float64(0.0015157303225032817), np.float64(0.0016160171819386518), np.float64(0.001407060466268008), np.float64(0.0011478650010252013), np.float64(0.001676788488845761), np.float64(0.0014542000047924129), np.float64(0.0013806502741846437)],

        [np.float64(0.005900808248306838), np.float64(0.004063283642477608), np.float64(0.0032964850244080308), np.float64(0.0027092555015251095), np.float64(0.002517520883271754), np.float64(0.002057096922662956), np.float64(0.0022319844423008012), np.float64(0.0024804666440379733), np.float64(0.0014698394092320613), np.float64(0.001543643138960979), np.float64(0.0015496635332383135), np.float64(0.0013340177355110198), np.float64(0.0014662977668546344), np.float64(0.0020777383493371863), np.float64(0.0014134039490788124), np.float64(0.0015357078959533602), np.float64(0.0013763354955045356), np.float64(0.0015626804030675393), np.float64(0.0013098035727416022), np.float64(0.001697297296547197), np.float64(0.0011481692917075256), np.float64(0.0009995441746885895), np.float64(0.00141199218327721)],

        [np.float64(0.005642008730437513), np.float64(0.0037453703088078673), np.float64(0.0029497891085204166), np.float64(0.0027885564146525525), np.float64(0.001931460806187325), np.float64(0.0025827004768087776), np.float64(0.0021225472337371627), np.float64(0.0015112615434579003), np.float64(0.0016878308283197025), np.float64(0.001429396448871636), np.float64(0.001519837414554696), np.float64(0.0020175600214931127), np.float64(0.0015011465366715675), np.float64(0.0014658942716238787), np.float64(0.0013651631212573553), np.float64(0.0013179421810274546), np.float64(0.0022025635417227195), np.float64(0.0013606235336899191), np.float64(0.0011904899227366892), np.float64(0.0014091803429901607), np.float64(0.0012203483754814042), np.float64(0.0013102859420594665), np.float64(0.0013457390245874682)],
    ]

    for losses, M in zip(M_losses, [0, 2, 4, 8, 16, 32, 64]):
        config["M"] = M
        print(f"Calculating M={M}")
        # losses = average_coefficient_loss(*load_models(config, D_arr))
        print(losses)
        
        # train_models(config, D_arr)
        # losses = average_coefficient_loss(*load_models(config, D_arr))
        plt.plot(D_arr, losses, label=f"M={config['M']}")

    # for d, loss, loss_fd, param in zip(D_arr, losses, losses_fd, parameters):
    #     print(f"d: {d:03} | Parameters: {param:06} | Average loss: {loss:.04} | Firedrake loss: {loss_fd:.04}")

    # plt.title(f"RelL2 loss vs D N={config['N']}")
    plt.xlabel("D - channels")
    plt.ylabel("RelL2")
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.savefig("channel_analysis")
    plt.show()
