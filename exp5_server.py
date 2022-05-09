# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Minimal example on how to start a simple Flower server."""


import argparse
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torchvision
from pathlib import Path
import exp5_utils


from flwr.server.strategy.dp_adaptive_clip_strategy import DPAdaptiveClipStrategy

# pylint: disable=no-member
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--server_address",
    type=str,
    required=True,
    help=f"gRPC server address",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=1,
    help="Number of rounds of federated learning (default: 1)",
)
parser.add_argument(
    "--sample_fraction",
    type=float,
    default=1.0,
    help="Fraction of available clients used for fit/evaluate (default: 1.0)",
)

parser.add_argument(
    "--total_num_clients",
    type=int,
    default=1,
    help="Number of total clients started",
)

parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="number of workers for dataset reading",
)
parser.add_argument(
    "--noise_multiplier",
    type=float,
    default=0.1,
    help="noise multiplier",
)
parser.add_argument(
        "--results_dir",
        type=str,
        default="exp5_results"
)
parser.add_argument(
        "--run_num",
        type=int,
        default=0,
        help="Run number for the particular config."
)
parser.add_argument(
    "--model",
    type=str,
    default="Net",
    choices=["Net"],
    help="model to train",
)

parser.add_argument("--pin_memory", action="store_true")
args = parser.parse_args()

def main() -> None:
    """Start server and train five rounds."""

    print(args)
    
    # Load evaluation data
    _, testset = exp5_utils.load_cifar(download=True)

    # Create client_manager, strategy, and server
    client_manager = fl.server.SimpleClientManager()
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.sample_fraction,
        min_fit_clients=args.total_num_clients,
        min_available_clients=args.total_num_clients,
        eval_fn=get_eval_fn(testset),
        on_fit_config_fn=fit_config,
    )
    dp_strategy = DPAdaptiveClipStrategy(strategy, args.total_num_clients, args.noise_multiplier  )

    results_dir = Path(args.results_dir)
    path_to_save_metrics = results_dir / "z_{}_rounds_{}_run_{}".format(args.noise_multiplier, args.rounds, args.run_num)
    if  not path_to_save_metrics.exists():
        Path.mkdir(path_to_save_metrics, parents=True)
    server = fl.server.Server(client_manager=client_manager, strategy=dp_strategy, path_to_save_metrics=path_to_save_metrics)
    # Run server
    fl.server.start_server(
        server_address=args.server_address,
        server=server,
        config={"num_rounds": args.rounds},
    )


def fit_config(rnd: int) -> Dict[str, fl.common.Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
        "epochs": str(1),
        "batch_size": str(32),
        "num_workers": str(args.num_workers),
        "pin_memory": str(args.pin_memory),
    }
    return config


def set_weights(model: torch.nn.ModuleList, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)


def get_eval_fn(
    testset: torchvision.datasets.CIFAR10,
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        model = exp5_utils.load_model(args.model)
        set_weights(model, weights)
        model.to(DEVICE)

        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        loss, accuracy = exp5_utils.test(model, testloader, device=DEVICE)
        return loss, {"accuracy": accuracy}

    return evaluate


if __name__ == "__main__":
    main()