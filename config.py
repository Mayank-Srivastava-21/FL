class Config:
    seed = 3042
    seeds = [3042] #, 3043, 3044]
    deterministic = True
    num_clients = 10
    num_rounds = 20
    local_epochs = 2
    batch_size = 64
    num_workers = 0

    local_lr = 0.05
    base_server_lr = 1.0

    datasets = ["CIFAR10", "CIFAR100", "PACS", "OFFICE_CALTECH10"]
    data_root = "./data"
    download = False            
    skip_missing_datasets = True

    image_size = 32

    subset_num_classes = 20
    subset_max_per_class = 150
    subset_test_fraction = 0.2
    subset_seed = 3042

    domainnet_domains = None      
    inat_num_classes = 100
    inat_max_per_class = 200
    inat_preserve_longtail = True

    longtail_imbalance = None     

    partition_mode = "dirichlet"
    dirichlet_alphas = [0.1]

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

    gaussian_scale = 0.5
    model_replacement_scale = 5.0
    ipm_scale = 1.0

    norm_cap_factor = 2.0
    center_temperature = 1.5
    direction_temperature = 5.0
    direction_floor = 0.05
    score_floor = 0.01
    weight_floor = 0.02

    server_step_cap_factor = 2.0
    server_step_cap_min = 1e-12

    agreement_temperature = 8.0
    agreement_gate = 0.05
    temporal_momentum = 0.80
    temporal_warmup = 2
    temporal_threshold = 1.5
    temporal_strength = 0.5
    fhhd_r_trimmed_blend = 0.25

    temporal_threshold_plus = 1.5
    temporal_strength_plus = 0.7
    temporal_warmup_plus = 2
    ipm_guard_strength = 0.8

    st_spatial_temperature = 5.0
    st_temporal_temperature = 5.0
    st_temporal_momentum = 0.80
    st_temporal_warmup = 1

    boba_rank = None              
    boba_residual_temperature = 0.5
    boba_direction_temperature = 3.0

    # ---------------- Output ----------------
    results_dir = "results_full_sweep"
    save_csv = True
    print_every = 5
    max_runs = None               # integer for a smoke test
