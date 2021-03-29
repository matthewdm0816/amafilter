import numpy as np
from utils import process_whole, get_model, parse_config
import torch
import torch_geometric as tgnn
from dataloader import normal_noise
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == '__main__':
    # parse args
    parser = ArgumentParser()
    # ...TODO
    parser.add_argument("-o", "--optimizer", type=str, help="Specify optimizer kind", nargs="?", const="SGD")
    parser.add_argument("-d", "--dataset", type=str, help="Specify dataset")
    parser.add_argument("-p", "--path", type=str, help="Specify dataset path")
    parser.add_argument("-b", "--batchsize", type=int, help="Batchsize/GPU of training")
    parser.add_argument("-t", "--total", type=int, help="Total epochs", nargs="?", const=500)
    parser.add_argument("-m", "--model", type=str, help="Specify model type")
    parser.add_argument("--gpus", type=int, nargs="+")
    parser.add_argument("-r", "--regularization", type=float, help="Specify graph regularization strength, 0 stands for none", nargs="?", const=0.)
    parser.add_argument("-l", "--loss", type=str, help="Specify loss type", nargs="?", const="mse") # alternative: chamfer
    parser.add_argument("--model_path", type=str, help='Specify model save file position')
    parser.add_argument("--ply_path", type=str, help='Specify PLY file position')
    args = parser.parse_args()
    optimizer_type, dataset_type, gpu_ids, gpu_id, ngpu, parallel, epochs, model_name, data_path, regularization, loss_type, device, batch_size = parse_config(args)
    ply_path, model_path = args.ply_path, args.model_path

    # Get model
    model = get_model(dataset_type=dataset_type, device=device, parallel=parallel, gpu_ids=gpu_ids, gpu_id=gpu_id, reg=regularization, loss_type=loss_type)

    # TODO: Load model

    # Process PLY
    reconstructed, orig_mesh, mse_error = process_whole(model, ply_path, noise_generator=normal_noise, sigma=1.0, batch_size=batch_size)

    # Visualize
    