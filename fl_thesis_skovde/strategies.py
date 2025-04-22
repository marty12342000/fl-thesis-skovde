from .task import Net, set_weights
import torch
import wandb
from datetime import datetime
from logging import INFO, WARNING
from typing import Callable, Optional, Union, Dict, Tuple, List
import toml

from flwr.common import (
    EvaluateIns, # Ajout de EvaluateIns pour la complétude
    EvaluateRes,
    FitRes, 
    MetricsAggregationFn, 
    NDArrays, 
    Parameters, 
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters # Ajout pour la complétude si nécessaire
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager # Ajout pour la complétude si nécessaire
from flwr.server.strategy import FedAvg

    


class EarlyStoppingAMBS(FedAvg):
    """
    Strategy for federated learning with early stopping.

    Args:
        patience (int): Number of rounds without improvement before stopping.
        gamma (float): Learning rate for the momentum.
        epsilon (float): Tolerance for the momentum.
        **kwargs: Additional arguments for the FedAvg strategy.


    """
    def __init__(
        self,
        *,          
        gamma: float = 0.5,
        momentum_threshold: float = 0.001,
        alpha_partition: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs) # Passe les arguments pertinents à FedAvg


        file_path = "./pyproject.toml"
        with open(file_path, "r") as f:
            data = toml.load(f)

        options = data["tool"]["flwr"]["federations"]["local-simulation"]["options"]
        nb_clients = options["num-supernodes"] 

        with open(file_path, "w") as f:
            toml.dump(data, f)

        name = datetime.now().strftime("%m-%d")
        wandb.init(project="fl-thesis-skovde", name=f"{name}-clients:{nb_clients}-alpha:{alpha_partition}-gamma:{gamma}")



        self.gamma = gamma
        self.momentum_threshold = momentum_threshold

        self.alpha_partition = alpha_partition


        self._stop_requested = False
        self._previous_loss = 0.0
        self._previous_momentum = 0.0


        log(INFO, f"AMBS strategy initialized with gamma={self.gamma} and momentum_threshold={self.momentum_threshold}")

    def should_stop(self) -> bool:
        """
        Indicate if the training should stop according to the stopping criteria.

        """
        return self._stop_requested


    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Evaluate the parameters (potentially on the server side) and log with wandb.
        
        """
        # Note: This method is for server-side evaluation (if evaluate_fn is provided).
        # Federated evaluation (clients) goes through aggregate_evaluate.
        # Adapt according to whether you use evaluate_fn or federated evaluation.

        loss, metrics = super().evaluate(server_round, parameters)


        # Adaptive momentum-based stopping strategy
        print(f"Server round: {server_round}")

        if server_round == 0:
            self._previous_loss = loss

        delta_loss = loss - self._previous_loss
        self._previous_loss = loss

        momentum = self.gamma * self._previous_momentum + (1 - self.gamma) * delta_loss

        #if server_round == 0:
        #    pass
        #else:
        #    if momentum < self.momentum_threshold:
        #        self._stop_requested = True


        self._previous_momentum = momentum
        self._previous_loss = loss


        my_results = {"server_eval_loss": loss, "server_momentum": momentum, **{f"server_{k}": v for k,v in metrics.items()}}

        try:
            if wandb.run:
                wandb.log(my_results, step=server_round)
        except Exception as e:
            log(WARNING, f"Wandb logging in evaluate failed: {e}")


        return loss, metrics


    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        print(f"Server round: {server_round}")

        print(f"Aggregated loss: {aggregated_loss}")
        print(f"Aggregated metrics: {aggregated_metrics}")

        return aggregated_loss, aggregated_metrics

        

        


    

    





