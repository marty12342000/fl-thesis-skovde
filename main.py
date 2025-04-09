import subprocess
import tomllib
from flwr.common import Context
import os

grid_search_params = {
    "nb_clients": [10],
    "alpha_partition": [0.1],
    "momentum_threshold": [0.0001],
}


def main():
    # parse pyproject.toml

    for nb_clients in grid_search_params["nb_clients"]:
        for alpha_partition in grid_search_params["alpha_partition"]:
            for momentum_threshold in grid_search_params["momentum_threshold"]:
                # update pyproject.toml
                # Get the current directory (where the script is running)
                current_dir = os.path.dirname(os.path.abspath(__file__))

                # List all files in that directory
                files = os.listdir(current_dir)
                print(files)
                with open("pyproject.toml", "rb") as f:
                    config = tomllib.load(f)

                    config["tool.flwr.app.config"]["num-supernodes"] = nb_clients
                    config["tool.flwr.app.config"]["alpha_partition"] = alpha_partition
                    config["tool.flwr.app.config"]["momentum_threshold"] = momentum_threshold
                
                # run flower simulation
                subprocess.run(["flwr", "run", "fl_thesis_skovde"])
            
                print(f"Finished {nb_clients} clients, {alpha_partition} alpha_partition, {momentum_threshold} momentum_threshold")


if __name__ == "__main__":
    main()
