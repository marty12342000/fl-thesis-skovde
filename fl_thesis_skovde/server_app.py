"""fl-thesis-skovde: A Flower / PyTorch app."""

from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerConfig
from flwr.server.serverapp_components import ServerAppComponents
from flwr.server.server_app import ServerApp
from flwr.server.strategy import FedAvg
from fl_thesis_skovde.task import Net, get_weights, set_weights, test
from datasets import load_dataset
from fl_thesis_skovde.strategies import EarlyStoppingAMBS
from torch.utils.data import DataLoader
from fl_thesis_skovde.task import get_transforms, which_dataset
import json
from .torchdatasetwrapper import TorchDatasetWrapper



def get_evaluate_fn(testloader, device):
    """Return a callback that evaluates the global model"""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model using provided centralised dataset"""

        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net, testloader, device)

        return loss, {"cen_accuracy": accuracy}
    
    return evaluate



def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregates metrics from an evaluate round."""
    # Loop trough all metrics received compute accuracies x examples
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    # Return weighted average accuracy
    return {"accuracy": sum(accuracies) / total_examples}



def on_fit_config_fn(server_round: int) -> Metrics:

    """Adjust learning rate based on current round."""

    lr = 0.01

    return {"lr": lr}

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    alpha_partition = context.run_config["alpha_partition"]


    # Initialize model parameters
    ndarrays = get_weights(Net()) # Net() is the model define in task.py 
    parameters = ndarrays_to_parameters(ndarrays)
    testset = None
    if which_dataset == "cifar10":
        testset = load_dataset("uoft-cs/cifar10")["test"]
        testset = testset.with_transform(get_transforms())

    elif which_dataset == "sentiment":
        trainset = load_dataset("mteb/tweet_sentiment_extraction")["train"]
        testset = load_dataset("mteb/tweet_sentiment_extraction")["test"]
        transform_fn = get_transforms(trainset)
        testset = TorchDatasetWrapper(testset, transform_fn)


    elif which_dataset == "mnist":
        testset = load_dataset("ylecun/mnist")["test"]
        testset = testset.with_transform(get_transforms())

    
    testloader = DataLoader(testset, batch_size=32)


    # Define strategy
    strategy = EarlyStoppingAMBS(
        dataset=which_dataset,
        alpha_partition=alpha_partition,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=on_fit_config_fn,
        evaluate_fn=get_evaluate_fn(testloader, device="cpu")
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
