"""Full matched-seed sweep: datasets x attacks x methods x seeds.

Attacks (attacks.py) and model architectures (models.py) are unchanged.
Byzantine IDs and partitions are randomized per seed block, then held fixed
across methods and attacks for paired comparisons.
"""

import copy
import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from config import Config
from data_utils import get_dataset, partition_dataset, labels_from_dataset
from models import get_model
from attacks import apply_byzantine_attack
from aggregators import (fedavg, krum, trimmed_mean, spatial_temporal,
                         boba_inspired, fhhd_plus, fhhd_r, fhhd_r_plus)


def seed_all(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        print(f"[DEVICE] {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("[DEVICE] CPU")
    return torch.device("cpu")


def evaluate(model, loader, device, num_classes):
    model.eval()
    correct = np.zeros(num_classes, dtype=np.int64)
    total = np.zeros(num_classes, dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            for c in range(num_classes):
                mask = y == c
                n_c = int(mask.sum().item())
                if n_c:
                    total[c] += n_c
                    correct[c] += int(((pred == y) & mask).sum().item())
    per_class = np.full(num_classes, np.nan, dtype=np.float64)
    valid = total > 0
    per_class[valid] = 100.0 * correct[valid] / total[valid]
    vals = per_class[valid]
    acc = 100.0 * correct.sum() / max(int(total.sum()), 1)
    wca = float(np.min(vals)) if vals.size else 0.0
    macro = float(np.mean(vals)) if vals.size else 0.0
    return acc, wca, macro, per_class, total


def flatten_delta(global_model, local_model):
    with torch.no_grad():
        return torch.cat([(lp - gp).reshape(-1) for gp, lp in
                          zip(global_model.parameters(), local_model.parameters())]).detach()


def train_client(global_model, loader, device, config):
    if len(loader.dataset) == 0:
        return torch.zeros(sum(p.numel() for p in global_model.parameters()),
                           device=device, dtype=next(global_model.parameters()).dtype)
    local = copy.deepcopy(global_model).to(device)
    local.train()
    opt = optim.SGD(local.parameters(), lr=config.local_lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(config.local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss_fn(local(x), y).backward()
            opt.step()
    return flatten_delta(global_model, local)


def apply_delta(model, delta, lr):
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            k = p.numel()
            p.add_(lr * delta[offset:offset + k].view_as(p))
            offset += k
    if offset != delta.numel():
        raise ValueError("Update vector length does not match model parameters.")


def make_loaders(train_set, client_indices, config, seed):
    loaders = []
    for cid, ids in enumerate(client_indices):
        gen = torch.Generator().manual_seed(seed + 1009 * cid)
        loaders.append(DataLoader(
            Subset(train_set, list(map(int, ids))),
            batch_size=config.batch_size, shuffle=bool(len(ids)),
            generator=gen, drop_last=False,
            num_workers=getattr(config, "num_workers", 0)))
    return loaders


def get_label_histograms(train_set, client_indices, seed, alpha, num_classes, dataset):
    labels = labels_from_dataset(train_set)
    rows = []
    for cid, ids in enumerate(client_indices):
        hist = (Counter(labels[np.asarray(ids, dtype=np.int64)].tolist())
                if len(ids) else Counter())
        for cls in range(num_classes):
            rows.append({"dataset": dataset, "seed": seed, "alpha": alpha,
                         "client_id": cid, "class_id": cls,
                         "count": int(hist.get(cls, 0)), "client_total": int(len(ids))})
    return rows


def aggregate(method, updates, config, history, f):
    if method == "FedAvg":
        return fedavg(updates)
    if method == "Krum":
        return krum(updates, f)
    if method == "TrimmedMean":
        return trimmed_mean(updates, f)
    if method == "Spatial Temporal":
        return spatial_temporal(updates, config, history, f)
    if method == "BOBA-inspired":
        return boba_inspired(updates, config, f)
    if method == "FHHD-Plus":
        return fhhd_plus(updates, config, history, f=f)
    if method == "FHHD-R":
        return fhhd_r(updates, config, history, f=f)
    if method == "FHHD-R+":
        return fhhd_r_plus(updates, config, history, f)
    raise ValueError(f"Unknown aggregation method: {method}")


def attack_update(attack, own_update, mean_honest, config):
    if attack == "none":
        return own_update
    if attack == "sign_flip":
        return apply_byzantine_attack(own_update, mean_honest, attack, config)
    return apply_byzantine_attack(mean_honest, mean_honest, attack, config)


def run_one(attack, method, initial_state, loaders, test_loader, device, config,
            byz_ids, seed, alpha, fraction, meta, dataset):
    num_classes = int(meta["num_classes"])
    model = get_model(dataset, num_classes=num_classes,
                      image_size=meta["image_size"],
                      in_channels=meta["in_channels"]).to(device)
    model.load_state_dict(copy.deepcopy(initial_state))
    history = {}
    accs, wcas, macros = [], [], []
    class_rows, diag_rows = [], []

    n = config.num_clients
    byz_set = set(byz_ids)
    honest_ids = [cid for cid in range(n) if cid not in byz_set]
    f = len(byz_ids)
    if not honest_ids:
        raise ValueError("At least one honest client is required.")

    for rnd in range(config.num_rounds):
        updates = [None] * n
        for cid in honest_ids:
            updates[cid] = train_client(model, loaders[cid], device, config)
        mean_honest = torch.stack([updates[cid] for cid in honest_ids]).mean(0)

        for cid in byz_ids:
            own = (train_client(model, loaders[cid], device, config)
                   if attack in ("none", "sign_flip") else mean_honest)
            updates[cid] = attack_update(attack, own, mean_honest, config)

        delta, energy, weights = aggregate(method, updates, config, history, f)
        update_norms = torch.stack([u.norm() for u in updates]).detach().cpu().numpy()
        apply_delta(model, delta, config.base_server_lr)

        a, w, m, per_class, counts = evaluate(model, test_loader, device, num_classes)
        accs.append(a); wcas.append(w); macros.append(m)

        for cls in range(num_classes):
            class_rows.append({
                "dataset": dataset, "seed": seed, "alpha": alpha,
                "requested_fraction": fraction, "byzantine_count": f,
                "attack": attack, "method": method, "round": rnd + 1,
                "class_id": cls,
                "accuracy": "NA" if np.isnan(per_class[cls]) else float(per_class[cls]),
                "test_count": int(counts[cls])})

        wv = None if weights is None else weights.detach().cpu().numpy()
        for cid in range(n):
            diag_rows.append({
                "dataset": dataset, "seed": seed, "alpha": alpha,
                "requested_fraction": fraction, "byzantine_count": f,
                "attack": attack, "method": method, "round": rnd + 1,
                "client_id": cid, "is_byzantine": int(cid in byz_set),
                "update_norm": float(update_norms[cid]),
                "aggregation_weight": "NA" if wv is None else float(wv[cid]),
                "aggregate_energy": float(energy),
                "aggregate_norm": float(delta.norm().detach().item())})

        if (rnd + 1) % config.print_every == 0 or rnd + 1 == config.num_rounds:
            print(f"  round={rnd+1:02d}/{config.num_rounds} attack={attack:<18} "
                  f"method={method:<17} byz={f}/{n} acc={a:.2f} WCA={w:.2f} macro={m:.2f}",
                  flush=True)

    return {"accuracy": accs[-1], "wca": wcas[-1], "macro_accuracy": macros[-1],
            "round_acc": accs, "round_wca": wcas, "round_macro": macros,
            "class_rows": class_rows, "diag_rows": diag_rows, "status": "ok"}


def write_csv(path, rows, fields=None):
    if not rows:
        return
    if fields is None:
        fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_summary(path, summary_rows):
    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[(row["dataset"], row["alpha"], row["requested_fraction"],
                 row["attack"], row["method"])].append(row)

    rows = []
    for key, values in sorted(grouped.items()):
        dataset, alpha, fraction, attack, method = key
        valid = [v for v in values if v.get("status") == "ok"]
        for metric in ("accuracy", "wca", "macro_accuracy"):
            vals = [float(v[metric]) for v in valid
                    if v.get(metric) not in ("NA", None, "")]
            mean = float(np.mean(vals)) if vals else float("nan")
            std = (float(np.std(vals, ddof=1)) if len(vals) > 1
                   else (0.0 if vals else float("nan")))
            rows.append({"dataset": dataset, "alpha": alpha,
                         "requested_fraction": fraction, "attack": attack,
                         "method": method, "metric": metric,
                         "n_seeds": len(vals), "mean": mean, "std_seed": std})

    out_path = Path(path)
    fields = ["dataset", "alpha", "requested_fraction", "attack", "method",
              "metric", "n_seeds", "mean", "std_seed"]
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lookup = {}
    for row in summary_rows:
        if row.get("status") == "ok":
            lookup[(row["dataset"], row["seed"], row["alpha"],
                    row["requested_fraction"], row["attack"], row["method"])] = row

    paired = []
    for key, r_row in lookup.items():
        dataset, seed, alpha, fraction, attack, method = key
        if method != "FHHD-R":
            continue
        for baseline in ("FedAvg", "Krum", "TrimmedMean", "Spatial Temporal",
                         "BOBA-inspired", "FHHD-Plus", "FHHD-R+"):
            b = lookup.get((dataset, seed, alpha, fraction, attack, baseline))
            if b is None:
                continue
            for metric in ("accuracy", "wca", "macro_accuracy"):
                try:
                    diff = float(r_row[metric]) - float(b[metric])
                except (TypeError, ValueError):
                    continue
                paired.append({"dataset": dataset, "seed": seed, "alpha": alpha,
                               "requested_fraction": fraction, "attack": attack,
                               "metric": metric, "candidate": "FHHD-R",
                               "baseline": baseline, "fhhd_r_minus_baseline": diff,
                               "fhhd_r_win": int(diff > 0), "tie": int(diff == 0)})

    paired_path = out_path.with_name("paired_seed_differences.csv")
    with paired_path.open("w", newline="", encoding="utf-8") as fp:
        fields = ["dataset", "seed", "alpha", "requested_fraction", "attack",
                  "metric", "candidate", "baseline", "fhhd_r_minus_baseline",
                  "fhhd_r_win", "tie"]
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paired)
    print(f"[OUTPUT] {out_path}")
    print(f"[OUTPUT] {paired_path}")


def _fmt(v):
    try:
        f = float(v)
        return "NA" if np.isnan(f) else f"{f:.2f}"
    except (TypeError, ValueError):
        return "NA"


def print_terminal_results(summary_rows, config, datasets_done):
    valid = [r for r in summary_rows if r.get("status") == "ok"]
    print("\n" + "=" * 120)
    print("FINAL RESULT FOR EVERY DATASET / SEED / ATTACK / METHOD")
    print("=" * 120)
    print(f"{'Dataset':<18} {'Seed':>6} {'Attack':<18} {'Method':<17} "
          f"{'Acc %':>9} {'WCA %':>9} {'Macro %':>9} {'Status':<10}")
    print("-" * 120)
    for r in summary_rows:
        print(f"{r.get('dataset','-'):<18} {r.get('seed','-'):>6} "
              f"{r.get('attack','-'):<18} {r.get('method','-'):<17} "
              f"{_fmt(r.get('accuracy')):>9} {_fmt(r.get('wca')):>9} "
              f"{_fmt(r.get('macro_accuracy')):>9} {str(r.get('status','-'))[:10]:<10}")

    for dataset in datasets_done:
        print("\n" + "=" * 120)
        print(f"AGGREGATE ACROSS SEEDS (mean +/- sample std) -- {dataset}")
        print("=" * 120)
        print(f"{'Attack':<18} {'Method':<17} {'Accuracy %':>20} {'WCA %':>20} {'Macro %':>20}")
        print("-" * 120)
        for attack in config.attacks:
            for method in config.methods:
                rows = [r for r in valid if r.get("dataset") == dataset
                        and r.get("attack") == attack and r.get("method") == method]
                cells = []
                for key in ("accuracy", "wca", "macro_accuracy"):
                    vals = np.asarray([float(r[key]) for r in rows
                                       if r.get(key) not in ("NA", None, "")], dtype=float)
                    if vals.size == 0:
                        cells.append("NA")
                    elif vals.size == 1:
                        cells.append(f"{vals.mean():.2f} +/- 0.00")
                    else:
                        cells.append(f"{vals.mean():.2f} +/- {vals.std(ddof=1):.2f}")
                print(f"{attack:<18} {method:<17} {cells[0]:>20} {cells[1]:>20} {cells[2]:>20}")


def main():
    config = Config()
    seed_all(config.seed, config.deterministic)
    device = get_device()
    os.makedirs(config.results_dir, exist_ok=True)

    summary_rows, round_rows, class_rows = [], [], []
    diag_rows, hist_rows, metadata_rows = [], [], []
    datasets_done, datasets_skipped = [], []

    total_planned = (len(config.datasets) * len(config.seeds) *
                     len(config.dirichlet_alphas) * len(config.byzantine_fractions) *
                     len(config.attacks) * len(config.methods))
    if config.max_runs is not None:
        total_planned = min(total_planned, int(config.max_runs))
    print(f"[PLAN] up to {total_planned} cells | datasets={config.datasets}")
    print(f"[PLAN] attacks={config.attacks}")
    print(f"[PLAN] methods={config.methods}")

    run_no = 0
    stop = False

    for dataset in config.datasets:
        if stop:
            break
        try:
            train_set, test_set, meta = get_dataset(dataset, config)
        except Exception as exc:
            msg = f"[SKIP] {dataset}: {str(exc).splitlines()[0]}"
            print(msg)
            datasets_skipped.append(dataset)
            if not getattr(config, "skip_missing_datasets", True):
                raise
            continue

        num_classes = int(meta["num_classes"])
        datasets_done.append(dataset)
        test_loader = DataLoader(test_set, batch_size=config.batch_size * 2,
                                 shuffle=False,
                                 num_workers=getattr(config, "num_workers", 0))
        print(f"\n[DATA] {dataset} | classes={num_classes} | train={meta['train_size']} "
              f"| test={meta['test_size']} | image={meta['image_size']}px "
              f"| domains={meta['has_domains']}")

        for seed in config.seeds:
            if stop:
                break
            for alpha in config.dirichlet_alphas:
                if stop:
                    break
                for fraction in config.byzantine_fractions:
                    if stop:
                        break
                    f = max(1, min(config.num_clients - 1,
                                   int(round(fraction * config.num_clients))))
                    block_seed = int(seed + round(alpha * 10000) + round(fraction * 1000))

                    seed_all(block_seed, config.deterministic)
                    indices = partition_dataset(train_set, config.num_clients, alpha,
                                                config.partition_mode)

                    seed_all(block_seed + 1, config.deterministic)
                    base_model = get_model(dataset, num_classes=num_classes,
                                           image_size=meta["image_size"],
                                           in_channels=meta["in_channels"]).to(device)
                    initial_state = copy.deepcopy(base_model.state_dict())

                    byz_rng = np.random.default_rng(block_seed + 7919)
                    byz_ids = sorted(byz_rng.choice(config.num_clients, size=f,
                                                    replace=False).tolist())

                    hist_rows.extend(get_label_histograms(train_set, indices, seed,
                                                          alpha, num_classes, dataset))
                    metadata_rows.append({
                        "dataset": dataset, "seed": seed, "block_seed": block_seed,
                        "alpha": alpha, "requested_fraction": fraction,
                        "byzantine_count": f, "byzantine_ids": json.dumps(byz_ids),
                        "partition_sizes": json.dumps([len(x) for x in indices]),
                        "num_clients": config.num_clients,
                        "num_rounds": config.num_rounds,
                        "local_epochs": config.local_epochs,
                        "num_classes": num_classes,
                        "partition_mode": config.partition_mode})

                    for attack in config.attacks:
                        if stop:
                            break
                        for method in config.methods:
                            if config.max_runs is not None and run_no >= int(config.max_runs):
                                stop = True
                                break
                            run_no += 1
                            print(f"\n[{run_no}/{total_planned}] {dataset} seed={seed} "
                                  f"alpha={alpha} frac={fraction} byz={byz_ids} "
                                  f"attack={attack} method={method}", flush=True)
                            try:
                                seed_all(block_seed + 100 + config.attacks.index(attack),
                                         config.deterministic)
                                run_loaders = make_loaders(train_set, indices, config, block_seed)
                                result = run_one(attack, method, initial_state, run_loaders,
                                                 test_loader, device, config, byz_ids,
                                                 seed, alpha, fraction, meta, dataset)
                                summary_rows.append({
                                    "dataset": dataset, "seed": seed,
                                    "block_seed": block_seed, "alpha": alpha,
                                    "requested_fraction": fraction,
                                    "byzantine_clients": f,
                                    "byzantine_ids": json.dumps(byz_ids),
                                    "num_clients": config.num_clients,
                                    "attack": attack, "method": method,
                                    "accuracy": result["accuracy"], "wca": result["wca"],
                                    "macro_accuracy": result["macro_accuracy"],
                                    "status": result["status"]})
                                for rnd, (a, w, m) in enumerate(zip(result["round_acc"],
                                                                    result["round_wca"],
                                                                    result["round_macro"]), 1):
                                    round_rows.append({
                                        "dataset": dataset, "seed": seed, "alpha": alpha,
                                        "requested_fraction": fraction, "attack": attack,
                                        "method": method, "round": rnd,
                                        "accuracy": a, "wca": w, "macro_accuracy": m})
                                class_rows.extend(result["class_rows"])
                                diag_rows.extend(result["diag_rows"])
                            except ValueError as exc:
                                if method == "Krum":
                                    summary_rows.append({
                                        "dataset": dataset, "seed": seed,
                                        "block_seed": block_seed, "alpha": alpha,
                                        "requested_fraction": fraction,
                                        "byzantine_clients": f,
                                        "byzantine_ids": json.dumps(byz_ids),
                                        "num_clients": config.num_clients,
                                        "attack": attack, "method": method,
                                        "accuracy": "NA", "wca": "NA",
                                        "macro_accuracy": "NA", "status": str(exc)})
                                else:
                                    raise

    print_terminal_results(summary_rows, config, datasets_done)
    if datasets_skipped:
        print(f"\n[SKIPPED DATASETS] {datasets_skipped}")

    if config.save_csv:
        rd = config.results_dir
        write_csv(os.path.join(rd, "sweep_summary.csv"), summary_rows)
        write_csv(os.path.join(rd, "round_metrics.csv"), round_rows)
        write_csv(os.path.join(rd, "class_accuracy.csv"), class_rows)
        write_csv(os.path.join(rd, "client_label_histograms.csv"), hist_rows)
        write_csv(os.path.join(rd, "aggregation_diagnostics.csv"), diag_rows)
        write_csv(os.path.join(rd, "run_metadata.csv"), metadata_rows)
        write_comparison_summary(os.path.join(rd, "method_comparison.csv"), summary_rows)
        print(f"[OUTPUT] CSV files written to {rd}")

    print(f"[DONE] Completed {len(summary_rows)} summary cells.")


if __name__ == "__main__":
    main()