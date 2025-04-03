from flwr.common import FitRes, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from .task import Net, set_weights
import torch
import json
import wandb
from datetime import datetime



class CustomStrategy(FedAvg):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.init(project="fl-thesis-skovde", name=f"custom-strategy-{name}")


    def aggregate_fit(self, 
                      server_round: int, 
                      results: list[tuple[ClientProxy, FitRes]], 
                      failures: list[tuple[ClientProxy, FitRes] | BaseException]
                      ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        # convert parameters to ndarrays
        ndarrays = parameters_to_ndarrays(parameters_aggregated)

        # instantiate model
        model = Net()
        set_weights(model, ndarrays)

        # save model
        torch.save(model.state_dict(), f"global_model_round_{server_round}.pth")


        return parameters_aggregated, metrics_aggregated
    

    def evaluate(self, 
                 server_round: int, 
                 parameters: Parameters
                 ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        
        loss, metrics = super().evaluate(server_round, parameters)

        my_results = {"loss": loss, **metrics}

        wandb.log(my_results, step=server_round)


        return loss, metrics
    

    

    





