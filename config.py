"""Configuration for the full FHHD-R sweep.

Sweep = datasets x attacks x methods x seeds.
Attacks and model architectures are unchanged from the original project.
"""

class Config:
    # ---------------- Reproducibility ----------------
    seed = 3042
    seeds = [3042] #, 3043, 3044]
    deterministic = True

    # ---------------- Federated setup ----------------
    num_clients = 10
    num_rounds = 20
    local_epochs = 2
    batch_size = 64
    num_workers = 0

    local_lr = 0.05
    base_server_lr = 1.0

    # ---------------- Data ----------------
    # Every dataset is attempted; unavailable ones are skipped with a warning.
    datasets = ["CIFAR10", "CIFAR100", "PACS", "OFFICE_CALTECH10"]
    data_root = "./data"
    download = False             # local-first; set True only with reliable network
    skip_missing_datasets = True

    # All RGB datasets are resized to 32x32 so the existing SimpleCNN is reused
    # without any architectural change. MNIST keeps 28x28 -> SimpleMLP(784).
    image_size = 32

    # Folder-dataset subset protocol (report these in the paper)
    subset_num_classes = 20
    subset_max_per_class = 150
    subset_test_fraction = 0.2
    subset_seed = 3042

    domainnet_domains = None      # None = all six domains
    inat_num_classes = 100
    inat_max_per_class = 200
    inat_preserve_longtail = True

    longtail_imbalance = None     # e.g. 100.0 -> CIFAR-100-LT

    # ---------------- Partitioning ----------------
    # dirichlet | domain | domain_dirichlet
    partition_mode = "dirichlet"
    dirichlet_alphas = [0.1]

    # ---------------- Threat model ----------------
    byzantine_fractions = [0.30]
    attacks = ["none", "sign_flip", "gaussian", "model_replacement", "ipm"]

    methods = [
        "FedAvg",
        "Krum",
        "TrimmedMean",
        "Spatial Temporal",
        "BOBA-inspired",
        "FHHD-Plus",
        "FHHD-R",
        "FHHD-R+",
    ]

    # Attack settings consumed by the UNCHANGED attacks.py
    gaussian_scale = 0.5
    model_replacement_scale = 5.0
    ipm_scale = 1.0

    # ---------------- Shared aggregation settings ----------------
    norm_cap_factor = 2.0
    center_temperature = 1.5
    direction_temperature = 5.0
    direction_floor = 0.05
    score_floor = 0.01
    weight_floor = 0.02

    server_step_cap_factor = 2.0
    server_step_cap_min = 1e-12

    # ---------------- FHHD-R ----------------
    agreement_temperature = 8.0
    agreement_gate = 0.05
    temporal_momentum = 0.80
    temporal_warmup = 2
    temporal_threshold = 1.5
    temporal_strength = 0.5
    fhhd_r_trimmed_blend = 0.25

    # FHHD-Plus compatibility
    temporal_threshold_plus = 1.5
    temporal_strength_plus = 0.7
    temporal_warmup_plus = 2
    ipm_guard_strength = 0.8

    # ---------------- Spatial Temporal baseline ----------------
    st_spatial_temperature = 5.0
    st_temporal_temperature = 5.0
    st_temporal_momentum = 0.80
    st_temporal_warmup = 1

    # ---------------- BOBA-inspired baseline ----------------
    boba_rank = None              # None -> max(1, n - 2f)
    boba_residual_temperature = 0.5
    boba_direction_temperature = 3.0

    # ---------------- Output ----------------
    results_dir = "results_full_sweep"
    save_csv = True
    print_every = 5
    max_runs = None               # integer for a smoke test