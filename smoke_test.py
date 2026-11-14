# smoke_test.py
import sys
from config import Config
from main import main

# Override config for a fast 4-cell smoke test
Config.datasets = ["CIFAR10", "CIFAR100", "PACS", "OFFICE_CALTECH10"]
Config.seeds = [3042]
Config.num_rounds = 1
Config.local_epochs = 1
Config.attacks = ["none"]
Config.methods = ["FedAvg"]
Config.max_runs = 4
Config.results_dir = "results_smoke"
Config.save_csv = False
Config.print_every = 1

if __name__ == "__main__":
    main()