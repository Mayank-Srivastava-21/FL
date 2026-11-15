import torch

def fedavg(updates):
    G = torch.stack(updates)
    n = len(updates)
    weights = torch.full((n,), 1.0 / n, device=G.device, dtype=G.dtype)
    energy = float((weights * G.square().sum(1)).sum().detach().item())
    return G.mean(0), energy, weights


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
        G = G.sort(dim=0).values[f:n - f]
    aggregate = G.mean(0)
    return aggregate, float(aggregate.square().sum().detach().item()), None

def _eps(G):
    return torch.finfo(G.dtype).eps


def _cos_rows(G, ref):
    eps = _eps(G)
    return (G @ ref) / (G.norm(dim=1).clamp_min(eps) * ref.norm().clamp_min(eps))


def _clip_updates(G, factor):
    norms = G.norm(dim=1)
    med = norms.median().clamp_min(_eps(G))
    cap = max(float(factor), 1.0) * med
    scale = (cap / norms.clamp_min(_eps(G))).clamp(max=1.0)
    return G * scale.unsqueeze(1), norms, cap


def _normalize(scores, floor=0.0):
    scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0).clamp_min(1e-12)
    scores = scores / scores.sum().clamp_min(1e-12)
    floor = min(max(float(floor), 0.0), 1.0 / max(scores.numel(), 1))
    if floor > 0:
        scores = floor + (1.0 - scores.numel() * floor) * scores
    return scores / scores.sum().clamp_min(1e-12)


def _normalize_masked(scores, mask, floor=0.0):
    eps = 1e-12
    s = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
    s = s * mask.to(s.dtype)
    if float(s.sum()) <= eps:
        s = mask.to(s.dtype)
    if float(s.sum()) <= eps:
        s = torch.ones_like(s)
    s = s / s.sum().clamp_min(eps)
    k = int(mask.sum().item())
    if k > 0 and float(floor) > 0.0:
        f = min(float(floor), 1.0 / k)
        s = torch.where(mask, f + (1.0 - k * f) * s, torch.zeros_like(s))
    return s / s.sum().clamp_min(eps)


def _pairwise_agreement(Gh):
    n = Gh.shape[0]
    C = Gh @ Gh.t()
    if n < 2:
        return torch.ones(n, device=Gh.device, dtype=Gh.dtype), C
    off_mask = ~torch.eye(n, dtype=torch.bool, device=Gh.device)
    off = C[off_mask].view(n, n - 1)
    return off.median(dim=1).values, C


def _cap_step(aggregate, clipped, config):
    eps = _eps(clipped)
    ref_norm = clipped.norm(dim=1).median().clamp_min(float(config.server_step_cap_min))
    cap = max(float(config.server_step_cap_factor), 0.0) * ref_norm
    norm = aggregate.norm()
    if norm > cap and norm > eps:
        aggregate = aggregate * (cap / norm)
    return aggregate, cap


def _temporal_scores(G, history, warmup, momentum, threshold, strength):
    n = G.shape[0]
    eps = _eps(G)
    scores = torch.ones(n, device=G.device, dtype=G.dtype)
    med_norm = G.norm(dim=1).median().clamp_min(eps)
    momentum = min(max(float(momentum), 0.0), 0.99)
    for i in range(n):
        current = G[i].detach()
        state = history.get(i)
        if state is None:
            history[i] = {"ema": current.clone(), "rounds": 1}
            continue
        old = state["ema"].to(device=G.device, dtype=G.dtype)
        denom = old.norm().clamp_min(0.1 * med_norm).clamp_min(eps)
        innovation = (current - old).norm() / denom
        if int(state.get("rounds", 1)) >= int(warmup):
            scores[i] = torch.exp(
                -float(strength) * torch.relu(innovation - float(threshold))
            ).clamp(0.02, 1.0)
        state["ema"] = (momentum * old + (1.0 - momentum) * current).detach()
        state["rounds"] = int(state.get("rounds", 1)) + 1
    return scores

def spatial_temporal(updates, config, client_history=None, f=0):
    history = client_history if client_history is not None else {}
    G = torch.stack(updates)
    if G.shape[0] == 0:
        raise ValueError("No client updates supplied.")
    eps = _eps(G)
    clipped, _, _ = _clip_updates(G, config.norm_cap_factor)
    Gh = clipped / clipped.norm(dim=1, keepdim=True).clamp_min(eps)

    agreement, _ = _pairwise_agreement(Gh)
    spatial = torch.sigmoid(float(getattr(config, "st_spatial_temperature", 5.0)) * agreement)

    n = G.shape[0]
    temporal = torch.ones(n, device=G.device, dtype=G.dtype)
    momentum = min(max(float(getattr(config, "st_temporal_momentum", 0.8)), 0.0), 0.99)
    warmup = int(getattr(config, "st_temporal_warmup", 1))
    t_temp = float(getattr(config, "st_temporal_temperature", 5.0))
    for i in range(n):
        cur = Gh[i].detach()
        state = history.get(i)
        if state is None:
            history[i] = {"dir": cur.clone(), "rounds": 1}
            continue
        prev = state["dir"].to(device=G.device, dtype=G.dtype)
        if int(state.get("rounds", 1)) >= warmup:
            cos = torch.dot(cur, prev) / (cur.norm().clamp_min(eps) * prev.norm().clamp_min(eps))
            temporal[i] = torch.sigmoid(t_temp * cos.clamp(-1.0, 1.0))
        history[i] = {"dir": (momentum * prev + (1.0 - momentum) * cur).detach(),
                      "rounds": int(state.get("rounds", 1)) + 1}

    scores = (spatial * temporal).clamp_min(float(config.score_floor))
    weights = _normalize(scores, getattr(config, "weight_floor", 0.0))
    aggregate = (weights.unsqueeze(1) * clipped).sum(0)
    aggregate, _ = _cap_step(aggregate, clipped, config)
    energy = float((weights * clipped.square().sum(1)).sum().detach().item())
    return aggregate, energy, weights.detach()

def boba_inspired(updates, config, f=0):
    G = torch.stack(updates)
    if G.shape[0] == 0:
        raise ValueError("No client updates supplied.")
    eps = _eps(G)
    n = G.shape[0]
    clipped, _, _ = _clip_updates(G, config.norm_cap_factor)

    center = clipped.median(dim=0).values
    X = clipped - center

    rank_cfg = getattr(config, "boba_rank", None)
    k = int(rank_cfg) if rank_cfg else max(1, n - 2 * int(f))
    k = max(1, min(k, n - 1))

    try:
        K = (X @ X.t()).detach().double().cpu()
        K = 0.5 * (K + K.t())
        _, evecs = torch.linalg.eigh(K)
        U = evecs[:, -k:].to(device=G.device, dtype=G.dtype)
        proj = U @ (U.t() @ X)
        residual = (X - proj).norm(dim=1)
    except Exception:
        residual = torch.zeros(n, device=G.device, dtype=G.dtype)

    base = X.norm(dim=1).clamp_min(eps)
    ratio = (residual / base).clamp(0.0, 2.0)
    tau = max(float(getattr(config, "boba_residual_temperature", 0.5)), 1e-3)
    subspace_score = torch.exp(-ratio / tau)

    cosine = _cos_rows(clipped, center).clamp(-1.0, 1.0)
    direction_score = torch.sigmoid(
        float(getattr(config, "boba_direction_temperature", 3.0)) * cosine)

    scores = (subspace_score * direction_score).clamp_min(float(config.score_floor))
    weights = _normalize(scores, getattr(config, "weight_floor", 0.0))
    aggregate = (weights.unsqueeze(1) * clipped).sum(0)
    aggregate, _ = _cap_step(aggregate, clipped, config)
    energy = float((weights * clipped.square().sum(1)).sum().detach().item())
    return aggregate, energy, weights.detach()


# ----------------------------------------------------------------------
# FHHD family
# ----------------------------------------------------------------------

def _signals(clipped, config, history):
    eps = _eps(clipped)
    Gh = clipped / clipped.norm(dim=1, keepdim=True).clamp_min(eps)

    agreement, _ = _pairwise_agreement(Gh)
    pos = agreement.clamp_min(0.0)
    if float(pos.sum()) <= eps:
        pos = torch.ones_like(agreement)
    ref = ((pos / pos.sum()).unsqueeze(1) * clipped).sum(0)

    center = clipped.median(dim=0).values
    distances = (clipped - center).norm(dim=1)
    scale = distances.median().clamp_min(
        0.1 * clipped.norm(dim=1).median()).clamp_min(eps)
    center_score = torch.exp(
        -distances / (max(float(config.center_temperature), 0.1) * scale))

    cosine = _cos_rows(clipped, ref).clamp(-1.0, 1.0)
    direction_score = float(config.direction_floor) + (
        1.0 - float(config.direction_floor)) * torch.sigmoid(
            float(config.direction_temperature) * cosine)

    thr = agreement.median()
    agreement_score = torch.sigmoid(
        float(getattr(config, "agreement_temperature", 8.0)) * (agreement - thr))

    temporal_score = _temporal_scores(
        clipped, history, config.temporal_warmup, config.temporal_momentum,
        config.temporal_threshold, config.temporal_strength)

    gate = agreement >= -float(getattr(config, "agreement_gate", 0.05))
    return dict(center_score=center_score, direction_score=direction_score,
                agreement_score=agreement_score, temporal_score=temporal_score,
                agreement=agreement, gate=gate, cosine=cosine, center=center)


def _score_updates(G, config, history, method="FHHD-R", f=None):
    eps = _eps(G)
    n = G.shape[0]
    clipped, _, _ = _clip_updates(G, config.norm_cap_factor)
    sig = _signals(clipped, config, history)

    stacked = torch.stack([sig["center_score"], sig["direction_score"],
                           sig["agreement_score"], sig["temporal_score"]]).clamp_min(eps)
    if method == "FHHD-Plus":
        scores = stacked.prod(dim=0)
    else:
        scores = torch.exp(torch.log(stacked).mean(dim=0))
    scores = scores.clamp_min(float(config.score_floor))

    gate = sig["gate"].clone()
    max_reject = int(f) if f is not None else (n - 1) // 2
    max_reject = max(0, min(max_reject, n - 1))
    if int((~gate).sum().item()) > max_reject:
        order = sig["agreement"].argsort()
        gate = torch.ones(n, dtype=torch.bool, device=G.device)
        gate[order[:max_reject]] = False

    weights = _normalize_masked(scores, gate, getattr(config, "weight_floor", 0.0))
    aggregate = (weights.unsqueeze(1) * clipped).sum(0)
    aggregate, _ = _cap_step(aggregate, clipped, config)
    energy = float((weights * clipped.square().sum(1)).sum().detach().item())
    return aggregate, energy, weights.detach()


def fhhd_r(updates, config, client_history=None, f=None):
    history = client_history if client_history is not None else {}
    G = torch.stack(updates)
    if G.shape[0] == 0:
        raise ValueError("No client updates supplied.")
    return _score_updates(G, config, history, "FHHD-R", f)


def fhhd_plus(updates, config, client_history=None, ablation=None, f=None):
    history = client_history if client_history is not None else {}
    G = torch.stack(updates)
    if G.shape[0] == 0:
        raise ValueError("No client updates supplied.")
    ablation = ablation or "full"
    if ablation == "full":
        return _score_updates(G, config, history, "FHHD-Plus", f)

    eps = _eps(G)
    base = G if ablation == "no_clip" else _clip_updates(G, config.norm_cap_factor)[0]
    sig = _signals(base, config, history)
    ones = torch.ones(G.shape[0], device=G.device, dtype=G.dtype)
    parts = [
        ones if ablation == "no_center" else sig["center_score"],
        ones if ablation == "no_direction" else sig["direction_score"],
        ones if ablation in ("no_peer", "no_agreement") else sig["agreement_score"],
        ones if ablation == "no_temporal" else sig["temporal_score"],
    ]
    scores = torch.stack(parts).clamp_min(eps).prod(0).clamp_min(float(config.score_floor))
    gate = (torch.ones(G.shape[0], dtype=torch.bool, device=G.device)
            if ablation == "no_ipm_guard" else sig["gate"])
    weights = _normalize_masked(scores, gate, getattr(config, "weight_floor", 0.0))
    aggregate = (weights.unsqueeze(1) * base).sum(0)
    aggregate, _ = _cap_step(aggregate, base, config)
    energy = float((weights * base.square().sum(1)).sum().detach().item())
    return aggregate, energy, weights.detach()


def fhhd_r_plus(updates, config, client_history=None, f=0):
    history = client_history if client_history is not None else {}
    G = torch.stack(updates)
    if G.shape[0] == 0:
        raise ValueError("No client updates supplied.")
    score_aggregate, energy, weights = _score_updates(G, config, history, "FHHD-R", f)
    clipped, _, _ = _clip_updates(G, config.norm_cap_factor)

    n = clipped.shape[0]
    trim = max(0, min(int(f), (n - 1) // 2))
    ordered = clipped.sort(dim=0).values
    trimmed = (ordered[trim:n - trim].mean(0)
               if trim > 0 and n > 2 * trim else clipped.mean(0))

    blend = min(max(float(getattr(config, "fhhd_r_trimmed_blend", 0.25)), 0.0), 1.0)
    aggregate = (1.0 - blend) * score_aggregate + blend * trimmed
    aggregate, _ = _cap_step(aggregate, clipped, config)
    return aggregate, energy, weights.detach()
