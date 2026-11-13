"""Matched-seed screening for FHHD-R and experimental FHHD-R+.

Uses the project's existing data_utils.py, models.py and attacks.py APIs. Outputs:
- sweep_summary.csv
- round_metrics.csv
- class_accuracy.csv
- client_label_histograms.csv
- aggregation_diagnostics.csv
- run_metadata.csv

Byzantine IDs and data partitions are randomized per seed/block, then held fixed
across methods and attacks for fair comparisons. This is a research harness, not a
claim that FHHD-R is guaranteed to beat every baseline.
"""
import copy
import csv
import json
import os
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from config import Config
from data_utils import get_dataset, dirichlet_partition
from models import get_model
from attacks import apply_byzantine_attack
from aggregators import fedavg, krum, trimmed_mean, fhhd_plus, fhhd_r, fhhd_r_plus


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


def labels_from_dataset(dataset):
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels, dtype=np.int64)
    labels = []
    for i in range(len(dataset)):
        item = dataset[i]
        labels.append(int(item[1]))
    return np.asarray(labels, dtype=np.int64)


def evaluate(model, loader, device, num_classes=10):
    model.eval()
    correct = np.zeros(num_classes, dtype=np.int64)
    total = np.zeros(num_classes, dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            for c in range(num_classes):
                mask = y == c
                total[c] += int(mask.sum().item())
                correct[c] += int(((pred == y) & mask).sum().item())
    per_class = np.full(num_classes, np.nan, dtype=np.float64)
    valid = total > 0
    per_class[valid] = 100.0 * correct[valid] / total[valid]
    # Macro accuracy is the unweighted mean of per-class accuracies. CIFAR-10's
    # test split is balanced, so macro and global accuracy can legitimately match.
    # WCA is the minimum over observed classes; absent classes are explicitly NaN.
    vals = per_class[valid]
    acc = 100.0 * correct.sum() / max(int(total.sum()), 1)
    wca = float(np.min(vals)) if vals.size else 0.0
    macro = float(np.mean(vals)) if vals.size else 0.0
    return acc, wca, macro, per_class, total


def flatten_delta(global_model, local_model):
    with torch.no_grad():
        return torch.cat([(lp-gp).reshape(-1) for gp, lp in zip(global_model.parameters(), local_model.parameters())]).detach()


def train_client(global_model, loader, device, config):
    local = copy.deepcopy(global_model).to(device)
    local.train()
    opt = optim.SGD(local.parameters(), lr=config.local_lr)
    loss_fn = nn.CrossEntropyLoss()
    if len(loader.dataset) == 0:
        return torch.zeros(sum(p.numel() for p in global_model.parameters()), device=device, dtype=next(global_model.parameters()).dtype)
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
            p.add_(lr * delta[offset:offset+k].view_as(p))
            offset += k
    if offset != delta.numel():
        raise ValueError("Update vector length does not match model parameters.")


def make_loaders(train_set, client_indices, config, seed):
    loaders = []
    for cid, ids in enumerate(client_indices):
        gen = torch.Generator().manual_seed(seed + 1009 * cid)
        loaders.append(DataLoader(Subset(train_set, list(map(int, ids))), batch_size=config.batch_size,
                                  shuffle=bool(len(ids)), generator=gen, drop_last=False))
    return loaders


def get_label_histograms(train_set, client_indices, seed, alpha):
    labels = labels_from_dataset(train_set)
    rows = []
    for cid, ids in enumerate(client_indices):
        hist = Counter(labels[np.asarray(ids, dtype=np.int64)].tolist()) if len(ids) else Counter()
        for cls in range(10):
            rows.append({"seed": seed, "alpha": alpha, "client_id": cid,
                         "class_id": cls, "count": int(hist.get(cls, 0)),
                         "client_total": int(len(ids))})
    return rows


def aggregate(method, updates, config, history, f):
    if method == "FedAvg":
        return fedavg(updates)
    if method == "Krum":
        return krum(updates, f)
    if method == "TrimmedMean":
        return trimmed_mean(updates, f)
    if method == "FHHD-Plus":
        return fhhd_plus(updates, config, history)
    if method == "FHHD-R":
        return fhhd_r(updates, config, history)
    if method == "FHHD-R+":
        return fhhd_r_plus(updates, config, history, f)
    raise ValueError(f"Unknown aggregation method: {method}")


def attack_update(attack, own_update, mean_honest, config):
    if attack == "none":
        return own_update
    if attack == "sign_flip":
        return apply_byzantine_attack(own_update, mean_honest, attack, config)
    return apply_byzantine_attack(mean_honest, mean_honest, attack, config)


def run_one(attack, method, initial_state, loaders, test_loader, device, config, byz_ids, seed, alpha, fraction):
    model = get_model(config.dataset).to(device)
    model.load_state_dict(copy.deepcopy(initial_state))
    history = {}
    accs, wcas, macros = [], [], []
    class_rows, diag_rows = [], []
    n = config.num_clients
    byz_set = set(byz_ids)
    honest_ids = [cid for cid in range(n) if cid not in byz_set]
    f = len(byz_ids)

    for rnd in range(config.num_rounds):
        updates = [None] * n
        # All clients get the same current global model. Honest updates are computed first
        # to construct the attack reference; Byzantine IDs remain fixed within this block.
        for cid in honest_ids:
            updates[cid] = train_client(model, loaders[cid], device, config)
        if not honest_ids:
            raise ValueError("At least one honest client is required for this attack harness.")
        mean_honest = torch.stack([updates[cid] for cid in honest_ids]).mean(0)
        for cid in byz_ids:
            own = train_client(model, loaders[cid], device, config) if attack in ("none", "sign_flip") else mean_honest
            updates[cid] = attack_update(attack, own, mean_honest, config)

        delta, energy, weights = aggregate(method, updates, config, history, f)
        update_norms = torch.stack([u.norm() for u in updates]).detach().cpu().numpy()
        apply_delta(model, delta, config.base_server_lr)
        a, w, m, per_class, class_counts = evaluate(model, test_loader, device)
        accs.append(a); wcas.append(w); macros.append(m)
        for cls in range(10):
            class_rows.append({"seed": seed, "alpha": alpha, "requested_fraction": fraction,
                               "byzantine_count": f, "attack": attack, "method": method,
                               "round": rnd+1, "class_id": cls,
                               "accuracy": "NA" if np.isnan(per_class[cls]) else float(per_class[cls]),
                               "test_count": int(class_counts[cls])})
        weight_values = None if weights is None else weights.detach().cpu().numpy()
        for cid in range(n):
            diag_rows.append({"seed": seed, "alpha": alpha, "requested_fraction": fraction,
                              "byzantine_count": f, "attack": attack, "method": method,
                              "round": rnd+1, "client_id": cid,
                              "is_byzantine": int(cid in byz_set),
                              "update_norm": float(update_norms[cid]),
                              "aggregation_weight": "NA" if weight_values is None else float(weight_values[cid]),
                              "aggregate_energy": float(energy),
                              "aggregate_norm": float(delta.norm().detach().item())})
        if (rnd+1) % config.print_every == 0 or rnd+1 == config.num_rounds:
            print(f" round={rnd+1:02d}/{config.num_rounds} attack={attack:<12} method={method:<12} "
                  f"byz={f}/{n} alpha={alpha} acc={a:.2f} WCA={w:.2f} macro={m:.2f}", flush=True)

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
    """Summarize matched-seed final metrics without hiding seed-level variability."""
    from collections import defaultdict
    grouped = defaultdict(list)
    for row in summary_rows:
        try:
            grouped[(row["alpha"], row["requested_fraction"], row["attack"], row["method"])].append(row)
        except KeyError:
            continue

    rows = []
    for (alpha, fraction, attack, method), values in sorted(grouped.items()):
        valid = [v for v in values if v.get("status") == "ok"]
        for metric in ("accuracy", "wca", "macro_accuracy"):
            vals = [float(v[metric]) for v in valid if v.get(metric) not in ("NA", None, "")]
            mean = float(np.mean(vals)) if vals else float("nan")
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0 if vals else float("nan")
            rows.append({"alpha": alpha, "requested_fraction": fraction, "attack": attack,
                         "method": method, "metric": metric, "n_seeds": len(vals),
                         "mean": mean, "std_seed": std})

    out_path = Path(path)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        fields = ["alpha", "requested_fraction", "attack", "method", "metric", "n_seeds", "mean", "std_seed"]
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Paired per-seed differences: positive means FHHD-R scored higher than baseline.
    lookup = {}
    for row in summary_rows:
        if row.get("status") == "ok":
            key = (row["seed"], row["alpha"], row["requested_fraction"], row["attack"], row["method"])
            lookup[key] = row
    paired = []
    baselines = ["FedAvg", "Krum", "TrimmedMean", "FHHD-Plus"]
    for key, r_row in lookup.items():
        seed, alpha, fraction, attack, method = key
        if method != "FHHD-R":
            continue
        for baseline in baselines:
            b = lookup.get((seed, alpha, fraction, attack, baseline))
            if b is None:
                continue
            for metric in ("accuracy", "wca", "macro_accuracy"):
                diff = float(r_row[metric]) - float(b[metric])
                paired.append({"seed": seed, "alpha": alpha, "requested_fraction": fraction,
                               "attack": attack, "metric": metric, "candidate": "FHHD-R",
                               "baseline": baseline, "fhhd_r_minus_baseline": diff,
                               "fhhd_r_win": int(diff > 0), "tie": int(diff == 0)})
    paired_path = out_path.with_name("paired_seed_differences.csv")
    with paired_path.open("w", newline="", encoding="utf-8") as fp:
        fields = ["seed", "alpha", "requested_fraction", "attack", "metric", "candidate",
                  "baseline", "fhhd_r_minus_baseline", "fhhd_r_win", "tie"]
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paired)
    print(f"[OUTPUT] {out_path}")
    print(f"[OUTPUT] {paired_path}")



def print_terminal_results(summary_rows, class_rows, config):
    """Print every completed cell plus mean/std and paired FHHD-R comparisons."""
    valid = [r for r in summary_rows if r.get("status") == "ok"]
    print("\n" + "=" * 112)
    print("FINAL RESULT FOR EVERY SEED / ATTACK / METHOD (test accuracy is descriptive during screening)")
    print("=" * 112)
    print(f"{'Seed':>6} {'Attack':<12} {'Method':<13} {'Acc %':>9} {'WCA %':>9} {'Macro %':>10} {'Status':<8}")
    print("-" * 112)
    for r in summary_rows:
        def fmt(v):
            try: return f"{float(v):.2f}"
            except (TypeError, ValueError): return "NA"
        print(f"{r.get('seed', '-'):>6} {r.get('attack', '-'): <12} {r.get('method', '-'): <13} "
              f"{fmt(r.get('accuracy')):>9} {fmt(r.get('wca')):>9} {fmt(r.get('macro_accuracy')):>10} "
              f"{str(r.get('status', '-'))[:8]:<8}")

    print("\n" + "=" * 112)
    print("AGGREGATE RESULTS ACROSS SEEDS (mean +/- sample standard deviation)")
    print("=" * 112)
    print(f"{'Attack':<12} {'Method':<13} {'Accuracy %':>20} {'WCA %':>20} {'Macro %':>20}")
    print("-" * 112)
    attacks = list(config.attacks)
    for attack in attacks:
        for method in config.methods:
            rows = [r for r in valid if r.get('attack') == attack and r.get('method') == method]
            metrics = []
            for key in ('accuracy', 'wca', 'macro_accuracy'):
                vals = np.asarray([float(r[key]) for r in rows], dtype=float)
                if len(vals) == 0: metrics.append('NA')
                elif len(vals) == 1: metrics.append(f"{vals.mean():.2f} +/- 0.00")
                else: metrics.append(f"{vals.mean():.2f} +/- {vals.std(ddof=1):.2f}")
            print(f"{attack:<12} {method:<13} {metrics[0]:>20} {metrics[1]:>20} {metrics[2]:>20}")

    print("\n" + "=" * 112)
    print("PAIRED FHHD-R WINS AGAINST EACH BASELINE (matched seed, same attack and partition)")
    print("Positive delta means FHHD-R is better; this is not a guarantee of statistical significance.")
    print("=" * 112)
    print(f"{'Attack':<12} {'Baseline':<13} {'Acc delta mean':>16} {'Acc wins/ties':>15} "
          f"{'WCA delta mean':>16} {'WCA wins/ties':>15}")
    print("-" * 112)
    for attack in attacks:
        for baseline in [m for m in config.methods if m != 'FHHD-R']:
            pairs = []
            for r in valid:
                if r.get('attack') != attack or r.get('method') != 'FHHD-R':
                    continue
                b = next((x for x in valid if x.get('seed') == r.get('seed') and
                          x.get('alpha') == r.get('alpha') and
                          x.get('requested_fraction') == r.get('requested_fraction') and
                          x.get('attack') == attack and x.get('method') == baseline), None)
                if b is not None:
                    pairs.append((float(r['accuracy'])-float(b['accuracy']), float(r['wca'])-float(b['wca'])))
            if not pairs:
                print(f"{attack:<12} {baseline:<13} {'NA':>16} {'0/0':>15} {'NA':>16} {'0/0':>15}")
                continue
            ad = np.asarray([x[0] for x in pairs]); wd = np.asarray([x[1] for x in pairs])
            at = int(np.sum(ad == 0)); wt = int(np.sum(wd == 0))
            print(f"{attack:<12} {baseline:<13} {ad.mean():>+15.2f} "
                  f"{int(np.sum(ad > 0))}/{at:<13} {wd.mean():>+15.2f} {int(np.sum(wd > 0))}/{wt:<13}")

    # Additional matched comparisons for experimental FHHD-R+ (kept separate from FHHD-R baseline table).
    if "FHHD-R+" in config.methods:
        print("\nPAIRED FHHD-R+ ACCURACY DIFFERENCES (candidate minus baseline, percentage points)")
        for attack in attacks:
            print(f"  {attack}:")
            for baseline in [m for m in config.methods if m != "FHHD-R+"]:
                diffs = []
                for r in valid:
                    if r.get("attack") != attack or r.get("method") != "FHHD-R+":
                        continue
                    b = next((x for x in valid if x.get("seed") == r.get("seed") and
                              x.get("alpha") == r.get("alpha") and
                              x.get("requested_fraction") == r.get("requested_fraction") and
                              x.get("attack") == attack and x.get("method") == baseline), None)
                    if b is not None:
                        diffs.append(float(r["accuracy"]) - float(b["accuracy"]))
                if diffs:
                    print(f"    vs {baseline:<13}: mean={np.mean(diffs):+6.2f} pp; wins={sum(d > 0 for d in diffs)}/{len(diffs)}")

    # Last-round class-level results: mean across seed runs for each attack/method/class.
    final_classes = {}
    for row in class_rows:
        if int(row.get('round', 0)) != config.num_rounds or row.get('accuracy') in ('NA', None, ''):
            continue
        key = (row['attack'], row['method'], int(row['class_id']))
        final_classes.setdefault(key, []).append(float(row['accuracy']))
    if final_classes:
        print("\nFINAL PER-CLASS TEST ACCURACY (%), MEAN ACROSS SEEDS")
        print(f"{'Attack':<12} {'Method':<13} " + " ".join(f"C{i:02d}" for i in range(10)))
        print("-" * 112)
        for attack in attacks:
            for method in config.methods:
                vals = []
                for cls in range(10):
                    xs = final_classes.get((attack, method, cls), [])
                    vals.append(f"{np.mean(xs):5.1f}" if xs else "   NA")
                print(f"{attack:<12} {method:<13} " + " ".join(vals))
    print("\n[NOTE] CIFAR-10 test data are balanced, so global and macro accuracy may be equal.")
    print("[NOTE] Choose final paper settings only after validation; do not assume FHHD-R wins before results.")


def main():
    config = Config()
    seed_all(config.seed, config.deterministic)
    device = get_device()
    os.makedirs(config.results_dir, exist_ok=True)
    train_set, test_set = get_dataset(config.dataset)
    test_loader = DataLoader(test_set, batch_size=config.batch_size*2, shuffle=False)
    num_classes = 10

    # Initial state is built per seed block below and then shared across every
    # method/attack within that block. This makes seeds independent but comparisons paired.
    initial_state = None

    summary_rows, round_rows, class_rows, diag_rows, hist_rows, metadata_rows = [], [], [], [], [], []
    blocks = [(seed, alpha, fraction) for seed in config.seeds
              for alpha in config.dirichlet_alphas for fraction in config.byzantine_fractions]
    total = len(blocks) * len(config.attacks) * len(config.methods)
    if config.max_runs is not None:
        total = min(total, int(config.max_runs))
    run_no = 0
    print(f"[INFO] FHHD-R screening | cells={total} | seeds={config.seeds} | output={config.results_dir}")
    print("[NOTE] This is an experimental prototype; no method is guaranteed to win every metric.")

    for seed, alpha, fraction in blocks:
        f = max(1, min(config.num_clients-1, int(round(fraction*config.num_clients))))
        block_seed = int(seed + round(alpha*10000) + round(fraction*1000))
        seed_all(block_seed, config.deterministic)
        indices = dirichlet_partition(train_set, config.num_clients, alpha)
        # A different initialization per seed block, reused for every method/attack
        # in this block. Re-seeding before construction makes it reproducible.
        seed_all(block_seed + 1, config.deterministic)
        base_model = get_model(config.dataset).to(device)
        initial_state = copy.deepcopy(base_model.state_dict())
        byz_rng = np.random.default_rng(block_seed + 7919)
        byz_ids = sorted(byz_rng.choice(config.num_clients, size=f, replace=False).tolist())
        hist_rows.extend(get_label_histograms(train_set, indices, seed, alpha))
        metadata_rows.append({"seed": seed, "block_seed": block_seed, "alpha": alpha,
                              "requested_fraction": fraction, "byzantine_count": f,
                              "byzantine_ids": json.dumps(byz_ids), "partition_sizes": json.dumps([len(x) for x in indices]),
                              "num_clients": config.num_clients, "num_rounds": config.num_rounds,
                              "local_epochs": config.local_epochs, "dataset": config.dataset})

        for attack in config.attacks:
            for method in config.methods:
                if run_no >= total:
                    break
                run_no += 1
                print(f"\n[{run_no}/{total}] seed={seed} alpha={alpha} fraction={fraction} "
                      f"byz_ids={byz_ids} attack={attack} method={method}")
                try:
                    # Re-seed each cell identically within the block and rebuild loaders;
                    # this matches minibatch order and any stochastic attack randomness.
                    seed_all(block_seed + 100 + config.attacks.index(attack), config.deterministic)
                    run_loaders = make_loaders(train_set, indices, config, block_seed)
                    result = run_one(attack, method, initial_state, run_loaders, test_loader, device,
                                     config, byz_ids, seed, alpha, fraction)
                    summary_rows.append({"seed": seed, "block_seed": block_seed, "alpha": alpha,
                        "requested_fraction": fraction, "byzantine_clients": f, "byzantine_ids": json.dumps(byz_ids),
                        "num_clients": config.num_clients, "attack": attack, "method": method,
                        "accuracy": result["accuracy"], "wca": result["wca"],
                        "macro_accuracy": result["macro_accuracy"], "status": result["status"]})
                    for rnd, (a, w, m) in enumerate(zip(result["round_acc"], result["round_wca"], result["round_macro"]), 1):
                        round_rows.append({"seed": seed, "block_seed": block_seed, "alpha": alpha,
                            "requested_fraction": fraction, "byzantine_clients": f, "byzantine_ids": json.dumps(byz_ids),
                            "attack": attack, "method": method, "round": rnd,
                            "accuracy": a, "wca": w, "macro_accuracy": m})
                    class_rows.extend(result["class_rows"])
                    diag_rows.extend(result["diag_rows"])
                except ValueError as exc:
                    if method == "Krum":
                        summary_rows.append({"seed": seed, "block_seed": block_seed, "alpha": alpha,
                            "requested_fraction": fraction, "byzantine_clients": f, "byzantine_ids": json.dumps(byz_ids),
                            "num_clients": config.num_clients, "attack": attack, "method": method,
                            "accuracy": "NA", "wca": "NA", "macro_accuracy": "NA", "status": str(exc)})
                    else:
                        raise
            if run_no >= total:
                break
        if run_no >= total:
            break

    print_terminal_results(summary_rows, class_rows, config)

    if config.save_csv:
        write_csv(os.path.join(config.results_dir, "sweep_summary.csv"), summary_rows)
        write_csv(os.path.join(config.results_dir, "round_metrics.csv"), round_rows)
        write_csv(os.path.join(config.results_dir, "class_accuracy.csv"), class_rows)
        write_csv(os.path.join(config.results_dir, "client_label_histograms.csv"), hist_rows)
        write_csv(os.path.join(config.results_dir, "aggregation_diagnostics.csv"), diag_rows)
        write_csv(os.path.join(config.results_dir, "run_metadata.csv"), metadata_rows)
        write_comparison_summary(os.path.join(config.results_dir, "method_comparison.csv"), summary_rows)
        print(f"[OUTPUT] CSV files written to {config.results_dir}")
    print(f"[DONE] Completed {len(summary_rows)} summary cells.")


if __name__ == "__main__":
    main()
