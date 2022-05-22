import flwr as fl
from flwr.common.typing import Scalar
from time import process_time

import argparse
import numpy as np
import os
import shutil
from pathlib import Path
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import random
from pathlib import Path
from typing import Dict

from utils import get_num_total_clients, get_eval_fn, get_model, load_trainset, load_testset
from flwr.client.dp_client import DPClient
from flwr.client.numpy_client import NumPyClientWrapper
from flwr.server.strategy.dp_adaptive_clip_strategy import DPAdaptiveClipStrategy
from flwr.common import weights_to_parameters
class EmnistRayClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.cid = cid
        self.model = get_model()

    def get_parameters(self):
        return self.model.get_weights()

    def get_properties(self, ins):
        return self.properties

    def fit(self, parameters, config):
            # load partition dataset
            trainset = load_trainset(self.cid)

            # send model to device
            self.model.set_weights(parameters)
            start_time = process_time()
            self.model.fit(trainset, epochs=1, verbose=0)
            end_time = process_time()
            print("elapsed time 1", end_time - start_time)

            # return local model and statistics
            
            return self.model.get_weights(), 1, {}

    def evaluate(self, parameters, config):
        # load test  data for this client 
    
        testset = load_testset()
        # evaluate
        loss, accuracy = self.model.evaluate(testset)

        return loss, 1, {"accuracy": accuracy}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--beta",
        type=float,
        default=0.9
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="exp7_results"
    )
    parser.add_argument(
        "--run_num",
        type=int,
        default=0,
        help="Run number for the particular config."
    )
    print("Starting")
    # Seeding
    args = parser.parse_args()
    
    total_num_clients = get_num_total_clients()
    
    client_resources = {
        "num_cpus": 1,
    }
    gpus =  tf.config.list_physical_devices('GPU')
    if len(gpus):
        client_resources["num_gpus"] = 0.5
    
    init_model = get_model()


    strategy = fl.server.strategy.FedAvgM(
        fraction_fit= 50/total_num_clients, 
        fraction_eval = 0,
        min_fit_clients=10,
        min_available_clients=total_num_clients,  # All clients should be available
        eval_fn=get_eval_fn(), 
        server_learning_rate = 1.0, 
        server_momentum = args.beta,
        initial_parameters = weights_to_parameters(init_model.get_weights())# centralised testset evaluation of global model
    )
    dp_strategy = DPAdaptiveClipStrategy(strategy, total_num_clients, 1)

    def client_fn(cid: str):
        # create a single client instance
        return DPClient(NumPyClientWrapper(EmnistRayClient(cid)), adaptive_clip_enabled=True)

    # (optional) specify ray config
    ray_config = {"include_dashboard": False}
    print("Starting training")
    # start simulation
    results_dir = Path(args.results_dir)
    path_to_save_metrics = results_dir / "beta_{}_run_{}".format(args.beta, args.run_num)

    if  not path_to_save_metrics.exists():
        Path.mkdir(path_to_save_metrics, parents=True)
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=total_num_clients,
        num_rounds=100,
        strategy=dp_strategy,
        ray_init_args=ray_config,
        client_resources = client_resources,
        path_to_save_metrics=path_to_save_metrics
    )