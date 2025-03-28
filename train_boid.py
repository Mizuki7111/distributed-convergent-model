from model import models
from data_utils import simulate_data, datasets, trainer, test_models

import os
import json
from pathlib import Path
from logging import getLogger, config
import argparse

import torch

from torchinfo import summary

from typing import Optional, Any

import numpy as np
from numpy import random as nr


def get_args():
    psr = argparse.ArgumentParser()
    psr.add_argument("log_dir", type=str)
    psr.add_argument("--type", type=str, default="swarm_wall")
    psr.add_argument("--save_all_files", type=bool, default=False)
    psr.add_argument("--n", type=int, default=30)
    psr.add_argument("--dim", type=int, default=2)
    psr.add_argument("--sim_steps", type=int, default=1000)
    psr.add_argument("--num_sim_train", type=int, default=10)
    psr.add_argument("--num_sim_validation", type=int, default=10)
    psr.add_argument("--num_sim_test", type=int, default=10)
    psr.add_argument("--dt", type=float, default=0.01)
    psr.add_argument("--dim_hid", type=int, default=64)
    psr.add_argument("--num_layers", type=int, default=4)
    psr.add_argument("--epochs", type=int, default=1000)
    psr.add_argument("--batch_size", type=int, default=500)
    psr.add_argument("--lr", type=float, default=5e-4)
    psr.add_argument("--l1reg", nargs="*", type=float, default=[0.0, 0.0])
    psr.add_argument("--l2reg", nargs="*", type=float, default=[1e-7, 1e-6])
    psr.add_argument("--num_cases", type=int, default=1)
    return psr.parse_args()


def boid_args(dim: int) -> dict:
    rectangle_barrier_force = simulate_data.RectangleInverseBarrierForce(
        np.array([[-5.0, 5.0]] * dim), 0.5
    )
    return {
        "a_align": 0.5,
        "a_cohesion": 2.0,
        "a_separation": 1.0,
        "d_align": 1.0,
        "d_cohesion": 1.0,
        "d_separation": 0.5,
        "vel_range": [0.0, 5.0],
        "forces": [rectangle_barrier_force],
    }


def simulate_boid(
    n: int,
    dim: int,
    steps: int,
    num_sim_train: int,
    num_sim_validation: int,
    num_sim_test: int,
    dt: Optional[float] = None,
    args: Any = None,
):
    if args is None:
        sim = simulate_data.BoidSimulation(**boid_args(dim))
    else:
        sim = simulate_data.BoidSimulation(**boid_args(dim), **args)
    data_train = np.zeros((num_sim_train, steps, n, dim * 2))
    data_validation = np.zeros((num_sim_validation, steps, n, dim * 2))
    data_test = np.zeros((num_sim_test, steps, n, dim * 2))
    for i in range(num_sim_train):
        while True:
            state_init = np.clip(
                nr.normal(0.0, 2.0, (n, 2 * dim)), -5.0 * 0.95, 5.0 * 0.95
            )
            state_init[:, dim:] = nr.uniform(-2.0, 2.0, (n, dim))
            data_train[i] = sim(steps, state_init, dt)
            if np.all(np.abs(data_train[i, :, :, :dim]) <= 5.0):
                break
    for i in range(num_sim_validation):
        while True:
            state_init = np.clip(
                nr.normal(0.0, 2.0, (n, 2 * dim)), -5.0 * 0.95, 5.0 * 0.95
            )
            state_init[:, dim:] = nr.uniform(-2.0, 2.0, (n, dim))
            data_validation[i] = sim(steps, state_init, dt)
            if np.all(np.abs(data_validation[i, :, :, :dim]) <= 5.0):
                break
    for i in range(num_sim_test):
        while True:
            state_init = np.clip(
                nr.normal(0.0, 2.0, (n, 2 * dim)), -5.0 * 0.95, 5.0 * 0.95
            )
            state_init[:, dim:] = nr.uniform(-2.0, 2.0, (n, dim))
            data_test[i] = sim(steps, state_init, dt)
            if np.all(np.abs(data_test[i, :, :, :dim]) <= 5.0):
                break
    return data_train, data_validation, data_test


def learning_models(args, save_dir):
    # Create save directory
    os.makedirs(f"{save_dir}/projection", exist_ok=True)
    os.makedirs(f"{save_dir}/hamiltonian", exist_ok=True)
    os.makedirs(f"{save_dir}/singlempnn", exist_ok=True)
    logger.info("Start simulation")
    # Simulate data
    data_train, data_validation, data_test = simulate_boid(
        n=args.n,
        dim=args.dim,
        steps=args.sim_steps,
        num_sim_train=args.num_sim_train,
        num_sim_validation=args.num_sim_validation,
        num_sim_test=args.num_sim_test,
        dt=args.dt,
    )
    if args.save_all_files:
        np.save(f"{save_dir}/data_train.npy", data_train)
        np.save(f"{save_dir}/data_validation.npy", data_validation)
    np.save(f"{save_dir}/data_test.npy", data_test)
    data_test = data_test[:, 0]
    # Transform data to dataset
    model_dim = 2 * args.dim
    model_dist = 1.0
    graph_generator = datasets.DistanceGraphTorch(model_dist, args.dim)
    data_train = datasets.StateBasedGraphDataset(data_train, graph_generator, args.dt)
    data_validation = datasets.StateBasedGraphDataset(
        data_validation, graph_generator, args.dt
    )
    # Setup trainer
    reg = []
    if sum(args.l1reg) > 0.0:
        reg.append(trainer.L1Reguralizer(*args.l1reg))
    if sum(args.l2reg) > 0.0:
        reg.append(trainer.L2Reguralizer(*args.l2reg))
    trainer_ = trainer.Trainer(torch.nn.MSELoss(), regularizers=reg)
    # Start process for projection model
    logger.info("Start training projection model")
    # Setup projection model
    v_pro = models.Vfunc(
        [
            models.PairwiseFunction(
                model_dim,
                args.dim_hid,
                1,
                args.num_layers,
                cofficient_func=models.DistanceGraphCofficient(
                    model_dist, args.dim, 0.01
                ),
            )
        ],
        [
            models.MLP(model_dim, args.dim_hid, 1, args.num_layers),
            models.QuadraticFunction(),
        ],
    )
    fhat = models.SingleMPNN(
        model_dim,
        args.dim_hid,
        model_dim,
        args.num_layers * 2,
        args.num_layers * 2,
        activation=torch.nn.ReLU(),
    )
    logger.debug("Args of projection model")
    logger.debug(
        f"v_pairwise->PairwiseFunction: {model_dim}, {args.dim_hid}, 1, {args.num_layers}, 'models.DistanceGraphCofficient({model_dist}, {args.dim})'"
    )
    logger.debug(f"v_each -> MLP: {model_dim}, {args.dim_hid}, 1, {args.num_layers}")
    logger.debug(f"v_each -> QuadraticFunction")
    logger.debug(
        f"fhat -> SingleMPNN: {model_dim}, {args.dim_hid}, {model_dim}, {args.num_layers * 2}, {args.num_layers * 2}"
    )
    projection_model = models.ProjectionModel(fhat, v_pro)
    logger.debug(summary(projection_model, verbose=0))
    # Setup optimizer
    projection_optimizer = torch.optim.Adam(projection_model.parameters(), lr=args.lr)
    # Train projection model
    _ = trainer_.train(
        model=projection_model,
        dataset=data_train,
        optimizer=projection_optimizer,
        batch_size=args.batch_size,
        epochs=args.epochs,
        varidation_dataset=data_validation,
        log_path=f"{save_dir}/projection",
    )
    logger.info("Start prediction with projection model")
    # Test projection model by long term prediction
    test_result = test_models.long_term_prediction(
        projection_model,
        data_test,
        args.sim_steps,
        args.dt,
        graph_generator=graph_generator,
    )
    np.save(f"{save_dir}/projection/final_prediction.npy", test_result)
    projection_model.load_state_dict(
        torch.load(f"{save_dir}/projection/best_model.pth")
    )
    test_result = test_models.long_term_prediction(
        projection_model,
        data_test,
        args.sim_steps,
        args.dt,
        graph_generator=graph_generator,
    )
    np.save(f"{save_dir}/projection/best_prediction.npy", test_result)
    del projection_model, projection_optimizer, v_pro, fhat, test_result
    # Start process for Hamiltonian model
    logger.info("Start training Hamiltonian model")
    # Setup Hamiltonian model
    v_ham = models.Vfunc(
        [
            models.PairwiseFunction(
                model_dim,
                args.dim_hid,
                1,
                args.num_layers,
                cofficient_func=models.DistanceGraphCofficient(
                    model_dist, args.dim, 0.01
                ),
            )
        ],
        [
            models.MLP(model_dim, args.dim_hid, 1, args.num_layers),
            models.QuadraticFunction(),
        ],
    )
    j_mat = models.SingleMPNN(
        model_dim,
        args.dim_hid,
        model_dim**2,
        args.num_layers,
        args.num_layers,
        activation=torch.nn.ReLU(),
    )
    r_mat = models.SingleMPNN(
        model_dim,
        args.dim_hid,
        model_dim**2,
        args.num_layers,
        args.num_layers,
        activation=torch.nn.ReLU(),
    )
    hamiltonian_model = models.HamiltonianModel(
        j_mat=j_mat, r_mat=r_mat, v=v_ham, dim=model_dim
    )
    logger.debug(summary(hamiltonian_model, verbose=0))
    logger.debug("Args of Hamiltonian model")
    logger.debug(
        f"v_pairwise->PairwiseFunction: {model_dim}, {args.dim_hid}, 1, {args.num_layers}, 'models.DistanceGraphCofficient({model_dist}, {args.dim})'"
    )
    logger.debug(f"v_each -> MLP: {model_dim}, {args.dim_hid}, 1, {args.num_layers}")
    logger.debug(f"v_each -> QuadraticFunction")
    logger.debug(
        f"j_mat -> SingleMPNN: {model_dim}, {args.dim_hid}, {model_dim**2}, {args.num_layers}, {args.num_layers}"
    )
    logger.debug(
        f"r_mat -> SingleMPNN: {model_dim}, {args.dim_hid}, {model_dim**2}, {args.num_layers}, {args.num_layers}"
    )
    # Setup optimizer
    hamiltonian_optimizer = torch.optim.Adam(hamiltonian_model.parameters(), lr=args.lr)
    # Train Hamiltonian model
    _ = trainer_.train(
        model=hamiltonian_model,
        dataset=data_train,
        optimizer=hamiltonian_optimizer,
        batch_size=args.batch_size,
        epochs=args.epochs,
        varidation_dataset=data_validation,
        log_path=f"{save_dir}/hamiltonian",
    )
    # Test Hamiltonian model by long term prediction
    test_result = test_models.long_term_prediction(
        hamiltonian_model,
        data_test,
        args.sim_steps,
        args.dt,
        graph_generator=graph_generator,
    )
    np.save(f"{save_dir}/hamiltonian/final_prediction.npy", test_result)
    hamiltonian_model.load_state_dict(
        torch.load(f"{save_dir}/hamiltonian/best_model.pth")
    )
    test_result = test_models.long_term_prediction(
        hamiltonian_model,
        data_test,
        args.sim_steps,
        args.dt,
        graph_generator=graph_generator,
    )
    np.save(f"{save_dir}/hamiltonian/best_prediction.npy", test_result)
    del hamiltonian_model, hamiltonian_optimizer, v_ham, j_mat, r_mat, test_result
    # Start process for SingleMPNN model
    logger.info("Start training SingleMPNN model")
    # Setup SingleMPNN model
    singlempnn = models.SingleMPNN(
        model_dim,
        args.dim_hid,
        model_dim,
        int(args.num_layers * 2.5),
        int(args.num_layers * 2.5),
        activation=torch.nn.ReLU(),
    )
    logger.debug(summary(singlempnn, verbose=0))
    logger.debug("Args of SingleMPNN model")
    logger.debug(
        f"SingleMPNN: {model_dim}, {args.dim_hid}, {model_dim}, {int(args.num_layers * 2.5)}, {int(args.num_layers * 2.5)}"
    )
    # Setup optimizer
    singlempnn_optimizer = torch.optim.Adam(singlempnn.parameters(), lr=args.lr)
    # Train SingleMPNN model
    _ = trainer_.train(
        model=singlempnn,
        dataset=data_train,
        optimizer=singlempnn_optimizer,
        batch_size=args.batch_size,
        epochs=args.epochs,
        varidation_dataset=data_validation,
        require_grad=False,
        log_path=f"{save_dir}/singlempnn",
    )
    # Test SingleMPNN model by long term prediction
    test_result = test_models.long_term_prediction(
        singlempnn,
        data_test,
        args.sim_steps,
        args.dt,
        graph_generator=graph_generator,
        require_grad=False,
    )
    np.save(f"{save_dir}/singlempnn/final_prediction.npy", test_result)
    singlempnn.load_state_dict(torch.load(f"{save_dir}/singlempnn/best_model.pth"))
    test_result = test_models.long_term_prediction(
        singlempnn,
        data_test,
        args.sim_steps,
        args.dt,
        graph_generator=graph_generator,
        require_grad=False,
    )
    np.save(f"{save_dir}/singlempnn/best_prediction.npy", test_result)
    del singlempnn, singlempnn_optimizer


if __name__ == "__main__":
    args = get_args()
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if len(os.listdir(args.log_dir)) > 0:
        for i in range(1, 100):
            if not os.path.exists(f"{args.log_dir}_{i}"):
                args.log_dir = f"{args.log_dir}_{i}"
                os.makedirs(args.log_dir)
                break
        else:
            print("Please make an empty directory")
            exit()
    with open(Path(__file__).resolve().parent.joinpath("log_config.json"), "r") as f:
        log_config = json.load(f)
    log_config["handlers"]["file_handler"]["filename"] = f"{args.log_dir}/boid.log"
    config.dictConfig(log_config)
    logger = getLogger(__name__)
    logger.debug(args)
    if args.num_cases > 1:
        for i in range(args.num_cases):
            logger.info(f"Start test {i+1}")
            save_dir = f"{args.log_dir}/test_{i+1}"
            learning_models(args, save_dir)
            logger.info(f"End test {i+1}")
        exit()
    else:
        save_dir = args.log_dir
        learning_models(args, save_dir)
        logger.info("End")
        exit()
