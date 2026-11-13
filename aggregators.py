"""Aggregation rules for controlled FHHD-R experiments.

FHHD-R is a proposed heuristic research variant, not a formal Hodge decomposition
and not a certified Byzantine-robust algorithm. Scores are diagnostics, not proofs
that a client is malicious. All methods return (aggregate, energy, client_weights).
For coordinate-wise trimmed mean, per-client scalar weights are not well-defined,
so client_weights is None.
"""
import torch


def fedavg(updates):
    G = torch.stack(updates)
    weights = torch.full((len(updates),), 1.0 / len(updates), device=G.device, dtype=G.dtype)
    return G.mean(0), float((weights * G.square().sum(1)).sum().detach().item()), weights


def krum(updates, f):
    G = torch.stack(updates)
    n = len(updates)
    if n < 2 * f + 3:
        raise ValueError(f"Krum requires n >= 2f+3 (n={n}, f={f}); skipped.")
    distances = torch.cdist(G, G)
    distances.fill_diagonal_(float("inf"))
    k = n - f - 2
    scores = distances.sort(dim=1).values[:, :k].sum(dim=1)
    chosen = int(scores.argmin().item())
    weights = torch.zeros(n, device=G.device, dtype=G.dtype)
    weights[chosen] = 1.0
    return G[chosen], float(G[chosen].square().sum().detach().item()), weights


def trimmed_mean(updates, f):
    G = torch.stack(updates)
    n = G.shape[0]
    if f > 0 and n > 2 * f:
        G = G.sort(dim=0).values[f:n-f]
    aggregate = G.mean(0)
    return aggregate, float(aggregate.square().sum().detach().item()), None


def _normalize(scores, floor=0.0):
    scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0).clamp_min(1e-12)
    scores = scores / scores.sum().clamp_min(1e-12)
    floor = min(max(float(floor), 0.0), 1.0 / max(scores.numel(), 1))
    if floor > 0:
        scores = floor + (1.0 - scores.numel() * floor) * scores
    return scores / scores.sum().clamp_min(1e-12)


def _cos_rows(G, ref):
    eps = torch.finfo(G.dtype).eps
    return (G @ ref) / (G.norm(dim=1).clamp_min(eps) * ref.norm().clamp_min(eps))


def _clip_updates(G, factor):
    norms = G.norm(dim=1)
    med = norms.median().clamp_min(torch.finfo(G.dtype).eps)
    cap = max(float(factor), 1.0) * med
    clipped = G * (cap / norms.clamp_min(torch.finfo(G.dtype).eps)).clamp(max=1.0).unsqueeze(1)
    return clipped, norms, cap


def _temporal_scores(G, history, warmup, momentum, threshold, strength):
    n = G.shape[0]
    scores = torch.ones(n, device=G.device, dtype=G.dtype)
    eps = torch.finfo(G.dtype).eps
    med_norm = G.norm(dim=1).median().clamp_min(eps)
    momentum = min(max(float(momentum), 0.0), 0.99)
    for i in range(n):
        current = G[i].detach()
        state = history.get(i)
        if state is None:
            history[i] = {"ema": current.clone(), "rounds": 1}
            continue
        old = state["ema"].to(device=G.device, dtype=G.dtype)
        innovation = (current - old).norm() / old.norm().clamp_min(0.1 * med_norm).clamp_min(eps)
        if int(state.get("rounds", 1)) >= int(warmup):
            scores[i] = torch.exp(-float(strength) * torch.relu(innovation - float(threshold))).clamp(0.02, 1.0)
        state["ema"] = (momentum * old + (1.0 - momentum) * current).detach()
        state["rounds"] = int(state.get("rounds", 1)) + 1
    return scores


def _score_updates(G, config, history, method="FHHD-R"):
    """Compute interpretable multi-signal weights and diagnostics."""
    eps = torch.finfo(G.dtype).eps
    clipped, raw_norms, cap = _clip_updates(G, config.norm_cap_factor)
    center = clipped.median(dim=0).values
    centered = clipped - center
    distances = centered.norm(dim=1)
    distance_scale = distances.median().clamp_min(0.1 * clipped.norm(dim=1).median()).clamp_min(eps)
    center_score = torch.exp(-distances / (max(float(config.center_temperature), 0.1) * distance_scale))

    cosine = _cos_rows(clipped, center).clamp(-1.0, 1.0)
    direction_score = float(config.direction_floor) + (1.0 - float(config.direction_floor)) * torch.sigmoid(
        float(config.direction_temperature) * cosine
    )

    peer_score = torch.ones(len(G), device=G.device, dtype=G.dtype)
    if len(G) > 1:
        for i in range(len(G)):
            peers = torch.cat((clipped[:i], clipped[i+1:]), dim=0)
            peer_center = peers.median(dim=0).values
            peer_cos = _cos_rows(clipped[i:i+1], peer_center)[0].clamp(-1.0, 1.0)
            peer_score[i] = torch.sigmoid(float(config.direction_temperature) * peer_cos)

    temporal_score = _temporal_scores(
        clipped, history, config.temporal_warmup, config.temporal_momentum,
        config.temporal_threshold, config.temporal_strength
    )

    # FHHD-R uses all four signals; FHHD-Plus preserves the older six-component
    # implementation's intent but shares stable normalization and instrumentation.
    if method == "FHHD-Plus":
        scores = center_score * direction_score * peer_score * temporal_score
    else:
        # Geometric averaging reduces the risk that multiplying several imperfect
        # signals makes every score collapse to the floor.
        signals = torch.stack([center_score, direction_score, peer_score, temporal_score]).clamp_min(eps)
        scores = torch.exp(torch.log(signals).mean(dim=0))
    scores = scores.clamp_min(float(config.score_floor))
    weights = _normalize(scores, getattr(config, "weight_floor", 0.0))
    aggregate = (weights.unsqueeze(1) * clipped).sum(0)

    # Bound server-step magnitude relative to typical clipped client updates.
    agg_norm = aggregate.norm()
    ref_norm = clipped.norm(dim=1).median().clamp_min(float(config.server_step_cap_min))
    step_cap = max(float(config.server_step_cap_factor), 0.0) * ref_norm
    if agg_norm > step_cap and agg_norm > eps:
        aggregate = aggregate * (step_cap / agg_norm)

    diag = {
        "raw_norms": raw_norms.detach(),
        "clipped_norms": clipped.norm(dim=1).detach(),
        "center_distances": distances.detach(),
        "center_scores": center_score.detach(),
        "direction_cosines": cosine.detach(),
        "direction_scores": direction_score.detach(),
        "peer_scores": peer_score.detach(),
        "temporal_scores": temporal_score.detach(),
        "weights": weights.detach(),
        "clip_cap": float(cap.detach().item()),
        "server_step_cap": float(step_cap.detach().item()),
        "server_step_norm": float(aggregate.norm().detach().item()),
        "center_aggregate_cosine": float(_cos_rows(aggregate.unsqueeze(0), center)[0].detach().item()) if aggregate.norm() > eps and center.norm() > eps else 0.0,
    }
    energy = float((weights * clipped.square().sum(1)).sum().detach().item())
    return aggregate, energy, weights.detach(), diag


def fhhd_r(updates, config, client_history=None):
    history = client_history if client_history is not None else {}
    G = torch.stack(updates)
    if G.shape[0] == 0:
        raise ValueError("No client updates supplied.")
    aggregate, energy, weights, _ = _score_updates(G, config, history, method="FHHD-R")
    return aggregate, energy, weights


def fhhd_plus(updates, config, client_history=None, ablation=None):
    """Backward-compatible FHHD-Plus approximation with optional ablations."""
    history = client_history if client_history is not None else {}
    G = torch.stack(updates)
    if G.shape[0] == 0:
        raise ValueError("No client updates supplied.")
    # Retain the earlier ablation names for compatibility.
    ablation = ablation or "full"
    if ablation == "full":
        aggregate, energy, weights, _ = _score_updates(G, config, history, method="FHHD-Plus")
        return aggregate, energy, weights
    # Ablations use the same raw diagnostics but explicitly disable one signal.
    clipped, raw_norms, cap = _clip_updates(G, config.norm_cap_factor)
    eps = torch.finfo(G.dtype).eps
    center = clipped.median(0).values
    distances = (clipped-center).norm(dim=1)
    scale = distances.median().clamp_min(0.1*clipped.norm(dim=1).median()).clamp_min(eps)
    center_score = torch.exp(-distances/(max(float(config.center_temperature),0.1)*scale))
    cosine = _cos_rows(clipped, center).clamp(-1,1)
    direction_score = float(config.direction_floor)+(1-float(config.direction_floor))*torch.sigmoid(float(config.direction_temperature)*cosine)
    peer_score = torch.ones(len(G),device=G.device,dtype=G.dtype)
    if len(G)>1:
        for i in range(len(G)):
            peers=torch.cat((clipped[:i],clipped[i+1:]),0)
            peer_score[i]=torch.sigmoid(float(config.direction_temperature)*_cos_rows(clipped[i:i+1],peers.median(0).values)[0].clamp(-1,1))
    temporal_score = _temporal_scores(clipped, history, config.temporal_warmup, config.temporal_momentum, config.temporal_threshold, config.temporal_strength)
    signals = {"no_center": torch.ones_like(center_score), "no_direction": torch.ones_like(direction_score), "no_peer": torch.ones_like(peer_score), "no_temporal": torch.ones_like(temporal_score), "no_clip": None, "no_ipm_guard": None}
    parts = []
    parts.append(torch.ones_like(center_score) if ablation=="no_center" else center_score)
    parts.append(torch.ones_like(direction_score) if ablation=="no_direction" else direction_score)
    parts.append(torch.ones_like(peer_score) if ablation=="no_peer" else peer_score)
    parts.append(torch.ones_like(temporal_score) if ablation=="no_temporal" else temporal_score)
    if ablation == "no_clip":
        clipped = G
    score = torch.stack(parts).clamp_min(eps).prod(0).clamp_min(float(config.score_floor))
    weights = _normalize(score, getattr(config,"weight_floor",0.0))
    aggregate=(weights.unsqueeze(1)*clipped).sum(0)
    cap_step=max(float(config.server_step_cap_factor),0.0)*clipped.norm(dim=1).median().clamp_min(float(config.server_step_cap_min))
    if aggregate.norm()>cap_step and aggregate.norm()>eps:
        aggregate=aggregate*(cap_step/aggregate.norm())
    energy=float((weights*clipped.square().sum(1)).sum().detach().item())
    return aggregate,energy,weights.detach()



def fhhd_r_plus(updates, config, client_history=None, f=0):
    """Experimental FHHD-R + coordinate-trimmed hybrid; no performance guarantee."""
    history = client_history if client_history is not None else {}
    G = torch.stack(updates)
    if G.shape[0] == 0:
        raise ValueError("No client updates supplied.")
    score_aggregate, energy, weights, _ = _score_updates(G, config, history, method="FHHD-R")
    clipped, _, _ = _clip_updates(G, config.norm_cap_factor)
    n = clipped.shape[0]
    trim = max(0, min(int(f), (n - 1) // 2))
    ordered = clipped.sort(dim=0).values
    trimmed = ordered[trim:n-trim].mean(0) if trim > 0 and n > 2 * trim else clipped.mean(0)
    blend = min(max(float(getattr(config, "fhhd_r_trimmed_blend", 0.25)), 0.0), 1.0)
    aggregate = (1.0 - blend) * score_aggregate + blend * trimmed
    eps = torch.finfo(G.dtype).eps
    ref_norm = clipped.norm(dim=1).median().clamp_min(float(config.server_step_cap_min))
    cap = max(float(config.server_step_cap_factor), 0.0) * ref_norm
    norm = aggregate.norm()
    if norm > cap and norm > eps:
        aggregate = aggregate * (cap / norm)
    return aggregate, energy, weights.detach()
