FHHD-R medium screening v2

Replace only main.py, config.py, and aggregators.py. Keep attacks.py, data_utils.py, and models.py unchanged.
Run: python main.py

Medium settings: CIFAR-10, 10 clients, 30% Byzantine, seeds 3042/3043/3044, attacks none/sign_flip/ipm, 20 rounds, 2 local epochs. Methods: FedAvg, Krum, TrimmedMean, FHHD-Plus, FHHD-R, FHHD-R+. Total 54 cells. Terminal summaries enabled; CSV writing disabled by default.

FHHD-R+ is a testable candidate: it blends FHHD-R's score-weighted clipped update with a coordinate-wise trimmed mean of clipped updates (blend=0.25). It may help or hurt depending on the attack; no result is guaranteed. Syntax checked, experiment not run here.
