import flwr as fl
from flwr.common.typing import Scalar

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
    seed = 0
    def __init__(self, cid):
        self.__set_seed()
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.cid = cid
        self.model = get_model()

    def __set_seed(self):
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    def get_parameters(self):
        return self.model.get_weights()

    def get_properties(self, ins):
        return self.properties

    def fit(self, parameters, config):
            # load partition dataset
            trainset = load_trainset(self.cid)

            # send model to device
            self.model.set_weights(parameters)
            
            self.model.fit(trainset, epochs=1, verbose=0)

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
        "--num_rounds",
        type=int,
        default=100,
        help="Number of rounds of federated learning (default: 1)",
    )
    parser.add_argument(
        "--num_clients_per_round",
        type=int,
        default=50,
        help="Number of available clients used for fit (default: 50)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Initial seed (default: 0)"
    )

    parser.add_argument(
        "--noise_multiplier",
        type=float,
        default=1,
        help="Noise multiplier (default: 1)"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results"
    )
    print("Starting")
    # Seeding
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    EmnistRayClient.seed = args.seed
    
    total_num_clients = get_num_total_clients()
    
    client_resources = {
        "num_cpus": 1,
    }
    gpus =  tf.config.list_physical_devices('GPU')
    if len(gpus):
        client_resources["num_gpus"] = 0.5
    
    init_model = get_model()


    strategy = fl.server.strategy.FedAvgM(
        fraction_fit= args.num_clients_per_round/total_num_clients, 
        fraction_eval = 0,
        min_fit_clients=10,
        min_available_clients=total_num_clients,  # All clients should be available
        eval_fn=get_eval_fn(), 
        server_learning_rate = 1.0, 
        server_momentum = 0.9,
        initial_parameters = weights_to_parameters(init_model.get_weights())# centralised testset evaluation of global model
    )
    dp_strategy = DPAdaptiveClipStrategy(strategy, total_num_clients, args.noise_multiplier)

    def client_fn(cid: str):
        # create a single client instance
        return DPClient(NumPyClientWrapper(EmnistRayClient(cid)), adaptive_clip_enabled=True)
        # return NumPyClientWrapper(EmnistRayClient(cid))

    # (optional) specify ray config
    ray_config = {"include_dashboard": False}
    print("Starting training")
    # start simulation
    results_dir = Path(args.results_dir)
    path_to_save_metrics = results_dir / "clients_{}_seed_{}_z_{}".format(args.num_clients_per_round, args.seed, args.noise_multiplier)
    if  path_to_save_metrics.exists():
        shutil.rmtree(path_to_save_metrics)
    Path.mkdir(path_to_save_metrics, parents=True)
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=total_num_clients,
        num_rounds=args.num_rounds,
        strategy=dp_strategy,
        ray_init_args=ray_config,
        client_resources = client_resources,
        path_to_save_metrics=path_to_save_metrics
    )