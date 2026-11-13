import torch


def apply_byzantine_attack(update, mean_honest, attack_type, config=None):
    """
    Apply a synthetic Byzantine attack to a client model-delta vector.

    `update` and `mean_honest` are model deltas (local_parameters - global_parameters),
    not raw loss gradients.

    Caveat: `model_replacement` below is a random high-magnitude target-delta proxy.
    A faithful model-replacement experiment should construct a malicious target model
    and submit its target_delta = target_parameters - global_parameters.
    """
    corrupted = update.clone()
    eps = torch.finfo(corrupted.dtype).eps

    if attack_type == "none":
        return corrupted

    if attack_type == "sign_flip":
        return -corrupted

    if attack_type == "gaussian":
        scale = mean_honest.norm().clamp(min=eps) * 0.5
        return corrupted + torch.randn_like(corrupted) * scale

    if attack_type == "model_replacement":
        # Explicitly a random high-norm target-delta proxy, not a full backdoor attack.
        scale_factor = getattr(config, "model_replacement_scale", 5.0)
        direction = torch.randn_like(corrupted)
        direction = direction / direction.norm().clamp(min=eps)
        return direction * (scale_factor * mean_honest.norm().clamp(min=eps))

    if attack_type == "ipm":
        # Inner-product manipulation proxy: move against the honest mean.
        ipm_scale = getattr(config, "ipm_scale", 0.5)
        return -ipm_scale * mean_honest

    raise ValueError(f"Unknown attack type: {attack_type}")
