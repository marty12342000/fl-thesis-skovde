import subprocess
from flwr.common import Context
import os
import toml

grid_search_params = {
    "nb_clients": [10],
    "alpha_partition": [0.1],
    "momentum_threshold": [0.0001],
    "gamma": [0.9],
}
    # parse pyproject.toml

for nb_clients in grid_search_params["nb_clients"]:
    for alpha_partition in grid_search_params["alpha_partition"]:
        for momentum_threshold in grid_search_params["momentum_threshold"]:
            for gamma in grid_search_params["gamma"]:

                file_path = ".\pyproject.toml"
                with open(file_path, "r") as f:
                    data = toml.load(f)

                # Navigate to the section you want
                config = data["tool"]["flwr"]["app"]["config"]
                config["num-supernodes"] = nb_clients
                config["alpha_partition"] = alpha_partition
                config["momentum_threshold"] = momentum_threshold
                config["gamma"] = gamma

                with open(file_path, "w") as f:
                    toml.dump(data, f)
                
                # run flower simulatio
                subprocess.run(["flwr", "run", "./"])
            
                print(f"Finished {nb_clients} clients, {alpha_partition} alpha_partition, {momentum_threshold} momentum_threshold")


