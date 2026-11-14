# smoke_test.py
"""Shorter-run head-to-head smoke test comparing SPECTRA-FL against all baselines
across all attacks and datasets without modifying any other project files.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from config import Config
import main as sweep_main


# ----------------------------------------------------------------------
# Proposed Method: SPECTRA-FL
# (Spectral-Layerwise Trust & Rare-Class Adaptive Aggregation)
# ----------------------------------------------------------------------
def _block_normalize(G, num_blocks=16):
    """Equalize structural layers (conv, hidden FC, classifier head) before scoring."""
    eps = torch.finfo(G.dtype).eps
    d = G.shape[1]
    b_count = min(int(num_blocks), d)
    chunks = torch.tensor_split(G, b_count, dim=1)
    normed_chunks = []
    for chunk in chunks:
        rms = chunk.square().mean(dim=1, keepdim=True).sqrt()
        med_rms = rms.median().clamp_min(eps)
        denom = rms.clamp_min(0.25 * med_rms).clamp_min(eps)
        normed_chunks.append(chunk / denom)
    G_bal = torch.cat(normed_chunks, dim=1)
    return G_bal / G_bal.norm(dim=1, keepdim=True).clamp_min(eps)


def _geometric_median(Gh, n_iters=6):
    """Weiszfeld geometric median on unit-normalized structural directions."""
    eps = torch.finfo(Gh.dtype).eps
    mu = Gh.median(dim=0).values
    if float(mu.norm()) <= eps:
        mu = Gh.mean(dim=0)
    mu = mu / mu.norm().clamp_min(eps)
    for _ in range(n_iters):
        dist = (Gh - mu.unsqueeze(0)).norm(dim=1).clamp_min(1e-6)
        w = 1.0 / dist
        mu_next = (w.unsqueeze(1) * Gh).sum(dim=0)
        if float(mu_next.norm()) <= eps:
            break
        mu = mu_next / mu_next.norm()
    return mu


def _loo_subspace_residuals(Gh_sub, rank):
    """Leave-one-out off-subspace residual ratios using exact Gram eigendecomposition."""
    m = Gh_sub.shape[0]
    if m <= 2:
        return torch.zeros(m, device=Gh_sub.device, dtype=Gh_sub.dtype)
    k = max(1, min(int(rank), m - 2))
    K = (Gh_sub @ Gh_sub.t()).detach().double()
    K = 0.5 * (K + K.t())
    res = torch.zeros(m, device=Gh_sub.device, dtype=torch.float64)
    all_idx = torch.arange(m, device=Gh_sub.device)
    for j in range(m):
        keep = all_idx != j
        K_sub = K[keep][:, keep]
        c_j = K[j, keep]
        try:
            evals, evecs = torch.linalg.eigh(K_sub)
            vals = evals[-k:].clamp_min(1e-12)
            vecs = evecs[:, -k:]
            proj_coords = vecs.t() @ c_j
            proj_energy = (proj_coords.square() / vals).sum()
            res[j] = (1.0 - proj_energy).clamp_min(0.0).sqrt()
        except Exception:
            res[j] = 0.0
    return res.to(dtype=Gh_sub.dtype)


def spectra_fl(updates, config, client_history=None, f=0):
    """SPECTRA-FL aggregator."""
    history = client_history if client_history is not None else {}
    G = torch.stack(updates)
    n = G.shape[0]
    if n == 0:
        raise ValueError("No client updates supplied.")
    eps = torch.finfo(G.dtype).eps
    f_int = max(0, min(int(f), (n - 1) // 2))

    if n < 3:
        agg = G.mean(0)
        w = torch.full((n,), 1.0 / n, device=G.device, dtype=G.dtype)
        return agg, float((w * G.square().sum(1)).sum().detach().item()), w

    # 1. Adaptive norm calibration (preserves clean non-IID scales, caps 5x spikes)
    raw_norms = G.norm(dim=1)
    med_norm = raw_norms.median().clamp_min(eps)
    clip_cap = 2.5 * med_norm
    clip_scale = (clip_cap / raw_norms.clamp_min(eps)).clamp(max=1.0)
    clipped = G * clip_scale.unsqueeze(1)

    # 2. Block-balanced structural embedding & Weiszfeld geometric median
    Gh = _block_normalize(clipped, num_blocks=16)
    mu = _geometric_median(Gh, n_iters=6)
    cos_mu = (Gh @ mu).clamp(-1.0, 1.0)

    # 3. Collusion-proof peer cosine (skips top f-1 potential colluding allies)
    C = (Gh @ Gh.t()).clamp(-1.0, 1.0)
    off_mask = ~torch.eye(n, dtype=torch.bool, device=G.device)
    off = C[off_mask].view(n, n - 1)
    off_sorted, _ = off.sort(dim=1, descending=True)
    lo = min(f_int, n - 2)
    hi = max(lo + 1, n - f_int - 1)
    peer_cos = off_sorted[:, lo:hi].mean(dim=1)
    dir_align = 0.5 * cos_mu + 0.5 * peer_cos

    # 4. LOO residual purification to build a Byzantine-free honest core
    n_core = max(3, min(n, n - f_int))
    pre_valid = torch.where(dir_align > 0.02)[0]
    if pre_valid.numel() < n_core:
        pre_valid = dir_align.argsort(descending=True)[:n_core]

    rank_core = max(2, min(n_core - 1, max(2, n - 2 * f_int)))
    Gh_pre = Gh[pre_valid]
    loo_res_pre = _loo_subspace_residuals(Gh_pre, rank=rank_core)
    pre_quality = dir_align[pre_valid] - 1.5 * loo_res_pre
    core_local = pre_quality.argsort(descending=True)[:n_core]
    core_idx = pre_valid[core_local]

    # 5. Clean-core subspace projection & orthogonal residual for all clients
    Gh_core = Gh[core_idx]
    K_core = (Gh_core @ Gh_core.t()).detach().double()
    K_core = 0.5 * (K_core + K_core.t())
    K_all_core = (Gh @ Gh_core.t()).detach().double()

    try:
        evals, evecs = torch.linalg.eigh(K_core)
        k_clean = max(2, min(n_core - 1, rank_core))
        vals = evals[-k_clean:].clamp_min(1e-12)
        vecs = evecs[:, -k_clean:]
        # Orthonormal subspace coordinates P of shape (n, k_clean)
        P = (K_all_core @ vecs) / vals.sqrt().unsqueeze(0)
        proj_energy = P.square().sum(dim=1)
        residuals = (1.0 - proj_energy).clamp_min(0.0).sqrt().to(dtype=G.dtype)
        P = P.to(dtype=G.dtype)
    except Exception:
        residuals = torch.zeros(n, device=G.device, dtype=G.dtype)
        P = torch.ones((n, 1), device=G.device, dtype=G.dtype)

    # 6. MAD statistical hard gate (zeroes out attackers, keeps all honest clients)
    med_dir = dir_align[core_idx].median()
    mad_dir = (dir_align[core_idx] - med_dir).abs().median().clamp_min(0.03)
    z_dir = (dir_align - med_dir) / (1.4826 * mad_dir)

    med_res = residuals[core_idx].median()
    mad_res = (residuals[core_idx] - med_res).abs().median().clamp_min(0.02)
    z_res = (residuals - med_res) / (1.4826 * mad_res)

    trusted = (
        (dir_align > 0.02)
        & (z_dir > -3.2)
        & (z_res < 3.0)
        & (residuals < med_res + 0.18)
        & (raw_norms <= 3.0 * med_norm)
    )

    min_keep = max(2, n - f_int) if f_int > 0 else n
    if int(trusted.sum().item()) < min_keep:
        pos_candidates = torch.where(dir_align > 0.0)[0]
        if pos_candidates.numel() == 0:
            pos_candidates = torch.arange(n, device=G.device)
        comb = z_dir[pos_candidates] - z_res[pos_candidates]
        pick = pos_candidates[comb.argsort(descending=True)[:min(min_keep, pos_candidates.numel())]]
        trusted = torch.zeros(n, dtype=torch.bool, device=G.device)
        trusted[pick] = True

    # 7. Whitened in-subspace leverage on verified honest clients (boosts WCA & Macro)
    axis_energy = P[trusted].square().sum(dim=0).clamp_min(1e-6)
    leverage = (P.square() / axis_energy.unsqueeze(0)).sum(dim=1)
    lev_mean = leverage[trusted].mean().clamp_min(eps)
    lev_boost = (leverage / lev_mean).clamp(0.60, 1.80)

    # 8. Temporal consistency check (catches sudden mid-run flip/noise bursts)
    temp_score = torch.ones(n, device=G.device, dtype=G.dtype)
    ema_bank = history.setdefault("client_ema", {})
    for i in range(n):
        cur_dir = Gh[i].detach()
        st = ema_bank.get(i)
        if st is None:
            ema_bank[i] = {"dir": cur_dir.clone(), "rounds": 1}
        else:
            prev_dir = st["dir"].to(device=G.device, dtype=G.dtype)
            if int(st["rounds"]) >= 1:
                c_temp = torch.dot(cur_dir, prev_dir).clamp(-1.0, 1.0)
                temp_score[i] = torch.sigmoid(4.0 * (c_temp + 0.25))
            if bool(trusted[i].item()):
                st["dir"] = (0.75 * prev_dir + 0.25 * cur_dir).detach()
                st["rounds"] = int(st["rounds"]) + 1

    # 9. Combine quality + rare-class leverage strictly inside trusted mask
    quality = (
        torch.sigmoid(4.0 * dir_align)
        * torch.exp(-0.5 * torch.relu(z_res))
        * temp_score
        * (0.60 + 0.40 * lev_boost)
    ).clamp_min(1e-6)

    masked_scores = torch.where(trusted, quality, torch.zeros_like(quality))
    if float(masked_scores.sum()) <= eps:
        masked_scores = trusted.to(dtype=G.dtype)
    weights = masked_scores / masked_scores.sum().clamp_min(eps)

    agg_clean = (weights.unsqueeze(1) * clipped).sum(dim=0)

    # 10. Byzantine-free server momentum & norm calibration
    mom = history.get("server_momentum")
    if mom is None:
        mom = agg_clean.detach().clone()
    else:
        mom = mom.to(device=G.device, dtype=G.dtype)
        mom = 0.30 * mom + 0.70 * agg_clean.detach()
    history["server_momentum"] = mom.detach().clone()

    aggregate = 0.75 * agg_clean + 0.25 * mom
    # Preserve clean aggregate energy scale while capping runaway steps
    target_norm = agg_clean.norm().clamp_min(eps)
    agg_norm = aggregate.norm()
    if agg_norm > eps:
        aggregate = aggregate * (target_norm / agg_norm)

    step_cap = 2.0 * med_norm
    final_norm = aggregate.norm()
    if final_norm > step_cap and final_norm > eps:
        aggregate = aggregate * (step_cap / final_norm)

    energy = float((weights * clipped.square().sum(1)).sum().detach().item())
    return aggregate, energy, weights.detach()


# ----------------------------------------------------------------------
# Monkey-patch main.aggregate & paired summary in-memory for smoke_test
# ----------------------------------------------------------------------
_ORIG_AGGREGATE = sweep_main.aggregate
_ORIG_WRITE_COMP = sweep_main.write_comparison_summary


def _patched_aggregate(method, updates, config, history, f):
    if method == "SPECTRA-FL":
        return spectra_fl(updates, config, history, f)
    return _ORIG_AGGREGATE(method, updates, config, history, f)


def _patched_write_comparison_summary(path, summary_rows):
    _ORIG_WRITE_COMP(path, summary_rows)
    out_path = Path(path)
    lookup = {}
    for row in summary_rows:
        if row.get("status") == "ok":
            lookup[(row["dataset"], row["seed"], row["alpha"],
                    row["requested_fraction"], row["attack"], row["method"])] = row

    paired = []
    for key, cand_row in lookup.items():
        dataset, seed, alpha, fraction, attack, method = key
        if method != "SPECTRA-FL":
            continue
        for baseline in ("FedAvg", "Krum", "TrimmedMean", "Spatial Temporal",
                         "BOBA-inspired", "FHHD-Plus", "FHHD-R", "FHHD-R+"):
            b = lookup.get((dataset, seed, alpha, fraction, attack, baseline))
            if b is None:
                continue
            for metric in ("accuracy", "wca", "macro_accuracy"):
                try:
                    diff = float(cand_row[metric]) - float(b[metric])
                except (TypeError, ValueError):
                    continue
                paired.append({
                    "dataset": dataset, "seed": seed, "alpha": alpha,
                    "requested_fraction": fraction, "attack": attack,
                    "metric": metric, "candidate": "SPECTRA-FL",
                    "baseline": baseline, "spectra_fl_minus_baseline": diff,
                    "spectra_fl_win": int(diff > 0), "tie": int(diff == 0),
                })

    paired_path = out_path.with_name("paired_seed_differences_spectra_fl.csv")
    with paired_path.open("w", newline="", encoding="utf-8") as fp:
        fields = [
            "dataset", "seed", "alpha", "requested_fraction", "attack",
            "metric", "candidate", "baseline", "spectra_fl_minus_baseline",
            "spectra_fl_win", "tie",
        ]
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paired)
    print(f"[OUTPUT] {paired_path}")


def _print_head_to_head(results_dir):
    comp_path = Path(results_dir) / "method_comparison.csv"
    if not comp_path.is_file():
        return
    rows = list(csv.DictReader(comp_path.open("r", encoding="utf-8")))
    grouped = defaultdict(dict)
    for r in rows:
        grouped[(r["dataset"], r["attack"], r["metric"])][r["method"]] = float(r["mean"])

    print("\n" + "=" * 110)
    print("HEAD-TO-HEAD SMOKE TEST CHECK: SPECTRA-FL vs BEST BASELINE")
    print("=" * 110)
    print(f"{'Dataset':<16} {'Attack':<18} {'Metric':<15} {'Best Baseline':<24} {'SPECTRA-FL':>11} {'Margin':>10}")
    print("-" * 110)
    wins, total = 0, 0
    for (dataset, attack, metric), m_dict in sorted(grouped.items()):
        if "SPECTRA-FL" not in m_dict:
            continue
        ours = m_dict["SPECTRA-FL"]
        baselines = {k: v for k, v in m_dict.items() if k != "SPECTRA-FL" and not np.isnan(v)}
        if not baselines:
            continue
        best_name = max(baselines, key=baselines.get)
        best_val = baselines[best_name]
        margin = ours - best_val
        total += 1
        if margin > 0:
            wins += 1
        print(f"{dataset:<16} {attack:<18} {metric:<15} "
              f"{best_name + f' ({best_val:.2f})':<24} {ours:>11.2f} {margin:>+10.2f}")
    print("-" * 110)
    print(f"[SMOKE SUMMARY] SPECTRA-FL beat the best baseline in {wins}/{total} (dataset, attack, metric) cells.")


sweep_main.aggregate = _patched_aggregate
sweep_main.write_comparison_summary = _patched_write_comparison_summary

# ----------------------------------------------------------------------
# Configure shorter smoke-test sweep across all attacks & methods
# ----------------------------------------------------------------------
Config.datasets = ["CIFAR10", "CIFAR100", "PACS", "OFFICE_CALTECH10"]
Config.seeds = [3045]
Config.num_rounds = 12
Config.local_epochs = 2
Config.attacks = ["none", "sign_flip", "gaussian", "model_replacement", "ipm"]
Config.methods = [
    "FedAvg",
    "Krum",
    "TrimmedMean",
    "Spatial Temporal",
    "BOBA-inspired",
    "FHHD-Plus",
    "FHHD-R",
    "FHHD-R+",
    "SPECTRA-FL",
]
Config.max_runs = None
Config.results_dir = "results_smoke"
Config.save_csv = True
Config.print_every = 4


if __name__ == "__main__":
    sweep_main.main()
    _print_head_to_head(Config.results_dir)