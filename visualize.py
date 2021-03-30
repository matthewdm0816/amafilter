import numpy as np
from utils import process_whole, get_model, parse_config
import torch
import torch_geometric as tgnn
from dataloader import normal_noise
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import proj3d, Axes3D
from utils import load_model


def get_subplot_proj(ax, scale):
    def short_proj():
        return np.dot(Axes3D.get_proj(ax), scale)

    return short_proj


def visualize(
    ax,
    pos: np.ndarray,
    color: np.ndarray,
    subtitle: str = "Original",
    point_size: float = 0.05,
):
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(subtitle)
    ax.get_proj = get_subplot_proj(ax)
    ax.scatter(
        pos[:, 0],  # x
        pos[:, 1],  # y
        pos[:, 2],  # z
        facecolors=color / 255.0,
        s=point_size,  # height data for color
    )
    ax.view_init(90, -90)


if __name__ == "__main__":
    # parse args
    parser = ArgumentParser()
    # ...TODO
    parser.add_argument(
        "-o",
        "--optimizer",
        type=str,
        help="Specify optimizer kind",
        nargs="?",
        const="SGD",
    )
    parser.add_argument("-d", "--dataset", type=str, help="Specify dataset")
    parser.add_argument("-p", "--path", type=str, help="Specify dataset path")
    parser.add_argument("-b", "--batchsize", type=int, help="Batchsize/GPU of training")
    parser.add_argument(
        "-t", "--total", type=int, help="Total epochs", nargs="?", const=500
    )
    parser.add_argument("-m", "--model", type=str, help="Specify model type")
    parser.add_argument("--gpus", type=int, nargs="+")
    parser.add_argument(
        "-r",
        "--regularization",
        type=float,
        help="Specify graph regularization strength, 0 stands for none",
        nargs="?",
        const=0.0,
    )
    parser.add_argument(
        "-l", "--loss", type=str, help="Specify loss type", nargs="?", const="mse"
    )  # alternative: chamfer
    parser.add_argument(
        "--model_path", type=str, help="Specify model save file position"
    )
    parser.add_argument("--ply_path", type=str, help="Specify PLY file position")
    parser.add_argument(
        "--sigma",
        "-s",
        type=float,
        help="Specify noise level sigma",
        nargs="+",
        const=1.0,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Specify result save path",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Sepcify whether to show the PC plot",
    )
    args = parser.parse_args()
    (
        optimizer_type,
        dataset_type,
        gpu_ids,
        gpu_id,
        ngpu,
        parallel,
        epochs,
        model_name,
        data_path,
        regularization,
        loss_type,
        device,
        batch_size,
    ) = parse_config(args)
    ply_path, model_path, save_path = args.ply_path, args.model_path, args.save_path
    sigma = args.sigma

    # Get model
    model = get_model(
        dataset_type=dataset_type,
        device=device,
        parallel=parallel,
        gpu_ids=gpu_ids,
        gpu_id=gpu_id,
        reg=regularization,
        loss_type=loss_type,
    )

    # TODO: Load model
    # ...
    load_model(
        model=model.module if parallel else model,
        optimizer=None,
        f=args.model_path,
        optim=None,
        e=0,
        evaluate=None,
    )

    # Process PLY
    reconstructed, noisy, mesh, mse_error = process_whole(
        model,
        ply_path,
        noise_generator=normal_noise,
        sigma=sigma,
        batch_size=batch_size,
    )

    # Visualize
    color, pos = mesh.color.numpy(), mesh.pos.numpy()
    x_scale = pos[:, 0].max()
    y_scale = pos[:, 1].max()
    z_scale = pos[:, 2].max()

    scale = np.diag([x_scale, y_scale, z_scale, 1.0])
    scale = scale * (1.0 / scale.max())
    scale[3, 3] = 1.0
    print(scale)

    mpl.rcParams["legend.fontsize"] = 10
    cm = 1 / 2.54
    fig = plt.figure(figsize=(20 * cm, 60 * cm))
    fig.subplots_adjust(bottom=-0.15, top=1.2)

    # original
    ax = fig.add_subplot(131, projection="3d")

    visualize(ax, pos, color, subtitle="Original")

    # noisy
    ax = fig.add_subplot(132, projection="3d")

    visualize(ax, pos, noisy.numpy(), subtitle="Noisy")

    # reconstructed
    ax = fig.add_subplot(133, projection="3d")

    visualize(ax, pos, reconstructed.numpy(), subtitle="Reconstructed")
    plt.savefig(save_path, dpi=600)
    if args.show:
        plt.show()