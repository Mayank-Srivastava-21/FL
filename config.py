"""Configuration for the FHHD-R controlled screening harness.

This is a research prototype, not a guarantee of Byzantine robustness. Start with
small screening runs, inspect diagnostics, then increase seeds/rounds for reporting.
"""

class Config:
    # Reproducibility and compute
    seed = 3042
    seeds = [3042, 3043, 3044]
    deterministic = True
    num_clients = 10
    # Medium-scale screening: enough optimization to reveal trends, but not paper scale.
    num_rounds = 20
    local_epochs = 2
    batch_size = 64
    dataset = "CIFAR10"
    local_lr = 0.05
    base_server_lr = 1.0

    # Initial controlled sweep. Extend only after the diagnostics look sensible.
    byzantine_fractions = [0.30]
    dirichlet_alphas = [0.1]
    attacks = ["none", "sign_flip", "ipm"]
    methods = ["FedAvg", "Krum", "TrimmedMean", "FHHD-Plus", "FHHD-R", "FHHD-R+"]

    # Shared attack settings expected by attacks.py
    gaussian_scale = 0.5
    model_replacement_scale = 5.0
    ipm_scale = 1.0

    # FHHD-R aggregation settings
    norm_cap_factor = 2.0
    center_temperature = 1.5
    direction_temperature = 5.0
    direction_floor = 0.05
    score_floor = 0.01
    weight_floor = 0.0
    temporal_momentum = 0.80
    temporal_warmup = 2
    temporal_threshold = 1.5
    temporal_strength = 0.5
    server_step_cap_factor = 2.0
    server_step_cap_min = 1e-12
    # Experimental blend coefficient for FHHD-R+; evaluate rather than assume a win.
    fhhd_r_trimmed_blend = 0.25

    # FHHD-Plus compatibility settings
    temporal_threshold_plus = 1.5
    temporal_strength_plus = 0.7
    temporal_warmup_plus = 2
    ipm_guard_strength = 0.8

    # Results
    results_dir = "results_fhhd_r_medium"
    save_csv = False  # Terminal summaries are the default; enable only if you want CSV backups.
    print_every = 5
    max_runs = None  # None means all cells; set an integer for a smoke test.
