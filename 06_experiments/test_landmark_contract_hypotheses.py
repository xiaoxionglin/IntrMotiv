"""Small mechanism tests motivated by the 2026-09-05 contract batch.

These tests establish the numerical failure modes independently of the full
Sample Factory runtime. Production integration remains covered by the
IntrMotiv source-tree tests and a forced-replacement Slurm preflight.
"""

import torch


def _moments(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return raw.mean(0), raw.var(0, unbiased=False)


def _normalize(
    raw: torch.Tensor, mean: torch.Tensor, variance: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    return (raw - mean) / torch.sqrt(variance + eps)


def test_post_update_statistics_match_updated_projection():
    generator = torch.Generator().manual_seed(99)
    samples, inputs, rows = 8192, 64, 16

    # Intermediate ReLU features are neither centered nor isotropic. This
    # synthetic distribution makes projection-dependent moments explicit.
    scale = torch.logspace(-1.0, 1.0, inputs)
    feature_mean = torch.linspace(-0.8, 1.2, inputs)
    features = torch.randn(samples, inputs, generator=generator) * scale + feature_mean

    old_weight = torch.randn(rows, inputs, generator=generator)
    old_weight = torch.nn.functional.normalize(old_weight, dim=1)
    old_raw = features @ old_weight.T
    old_mean, old_variance = _moments(old_raw)

    # A modest tangent update followed by the production row normalization.
    update = 0.04 * torch.randn(rows, inputs, generator=generator)
    update -= (update * old_weight).sum(1, keepdim=True) * old_weight
    new_weight = torch.nn.functional.normalize(old_weight + update, dim=1)
    new_raw = features @ new_weight.T

    lagged_z = _normalize(new_raw, old_mean, old_variance)
    new_mean, new_variance = _moments(new_raw)
    atomic_z = _normalize(new_raw, new_mean, new_variance)

    assert atomic_z.mean(0).abs().max() < 1e-5
    assert (atomic_z.var(0, unbiased=False) - 1).abs().max() < 2e-4

    assert lagged_z.mean(0).abs().max() > 0.05
    assert (lagged_z.var(0, unbiased=False) - 1).abs().max() > 0.05
    lagged_active = lagged_z > 2.43
    atomic_active = atomic_z > 2.43
    assert (lagged_active != atomic_active).float().mean() > 0.002


def test_running_stat_gradient_loses_batch_centering_effect():
    generator = torch.Generator().manual_seed(123)
    samples, inputs, rows = 16384, 64, 16
    feature_mean = torch.linspace(0.2, 1.2, inputs)
    features = (
        torch.randn(samples, inputs, generator=generator) * torch.logspace(-0.7, 0.7, inputs)
        + feature_mean
    )
    initial_weight = torch.nn.functional.normalize(
        torch.randn(rows, inputs, generator=generator), dim=1
    )
    positive_credit = 0.5 + torch.rand(samples, rows, generator=generator)

    def encoder_gradient(fixed_statistics: bool):
        weight = initial_weight.clone().requires_grad_(True)
        raw = features @ weight.T
        mean, variance = _moments(raw)
        if fixed_statistics:
            mean, variance = mean.detach(), variance.detach()
        z = _normalize(raw, mean, variance)
        activity = torch.relu(z - 2.43)
        loss = -(positive_credit * activity).mean()
        loss.backward()
        return activity.detach(), weight.grad.detach()

    batch_activity, batch_gradient = encoder_gradient(fixed_statistics=False)
    fixed_activity, fixed_gradient = encoder_gradient(fixed_statistics=True)

    # Forward values are identical. Only the normalization Jacobian differs.
    assert torch.equal(batch_activity, fixed_activity)

    mean_direction = torch.nn.functional.normalize(feature_mean, dim=0)

    def common_mode_scores(gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        directions = torch.nn.functional.normalize(gradient, dim=1)
        mean_alignment = (directions @ mean_direction).abs().mean()
        pairwise = directions @ directions.T
        off_diagonal = (pairwise.sum() - pairwise.diag().sum()) / (rows * (rows - 1))
        return mean_alignment, off_diagonal

    batch_mean_alignment, batch_pairwise = common_mode_scores(batch_gradient)
    fixed_mean_alignment, fixed_pairwise = common_mode_scores(fixed_gradient)

    # Treating moments as constants exposes every positively credited row to a
    # shared pull toward the non-zero visual-feature mean. Differentiable batch
    # moments subtract most of that unplanned common mode.
    assert fixed_mean_alignment > 0.50
    assert batch_mean_alignment < 0.25
    assert fixed_pairwise > 0.30
    assert batch_pairwise < 0.10
    assert torch.nn.functional.cosine_similarity(
        batch_gradient.flatten(), fixed_gradient.flatten(), dim=0
    ) < 0.10


def test_singleton_valid_advantage_has_nan_sample_std():
    advantages = torch.linspace(-1, 1, 2048)
    valid = torch.zeros(2048, dtype=torch.bool)
    valid[123] = True

    std, mean = torch.std_mean(advantages[valid])

    assert torch.isnan(std)
    assert torch.isfinite(mean)


def test_generation_barrier_operates_on_complete_sequences():
    rollouts, decisions = 64, 64
    stored_generation = torch.zeros(rollouts, decisions, dtype=torch.long)
    stored_generation[-6:] = 1
    current_generation = 1

    # Element masking retains some decisions but only six complete sequences.
    element_valid = stored_generation == current_generation
    assert element_valid.sum().item() == 6 * decisions

    # A rollout barrier must not treat these as a normal PPO batch.
    complete_current = element_valid.all(dim=1)
    assert complete_current.sum().item() == 6
    minimum_rollouts_for_update = 32
    assert complete_current.sum().item() < minimum_rollouts_for_update


def test_event_pair_opportunity_falls_superlinearly():
    old_per_row = 0.028
    new_per_row = 0.012
    rows = 16
    old_any = 1 - (1 - old_per_row) ** rows
    new_any = 1 - (1 - new_per_row) ** rows

    assert 0.36 < old_any < 0.37
    assert 0.17 < new_any < 0.18
    assert (new_any / old_any) ** 2 < 0.25
