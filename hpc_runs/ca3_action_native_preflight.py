"""Bounded native sensory/replay qualification from one canonical StudySpec row.

This executes no production training. An explicit local-calibration flag permits
a random-DG fixture while remote learned-source verification is unavailable.
The normal path loads the declared DG source and checks it remains frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from hpc_runs.intrmotiv_study import load_study
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.utils.attr_dict import AttrDict
from sf_working_directories.IntrMotiv.dmlab.custom_actor_critic import make_hipposlam_actor_critic
from sf_working_directories.IntrMotiv.dmlab.custom_learner import DistanceLearnerReward
from sf_working_directories.IntrMotiv.dmlab.dmlab_env import make_dmlab_env
from sf_working_directories.IntrMotiv.dmlab.train_hipposlam import parse_dmlab_args, register_dmlab_components
from sf_working_directories.IntrMotiv.evaluation.ca3_action_prediction import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--runfiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decisions", type=int, default=80)
    parser.add_argument("--local-calibrated-dg-fixture", action="store_true")
    args = parser.parse_args()
    if args.decisions < 2:
        raise ValueError("At least two actual decisions are required")
    output = args.output.resolve()
    if not args.local_calibrated_dg_fixture:
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Learned-source native qualification requires a Slurm compute job")
        from sf_working_directories.IntrMotiv.evaluation.submit_place_field_sweep import workspace_path

        output = workspace_path(output, "Native action-memory qualification")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    study = load_study(args.study)
    matches = [run for run in study.expand_runs() if run.name == args.run_name]
    if len(matches) != 1:
        raise ValueError("Select exactly one canonical run name")
    run = matches[0]
    overrides = [
        f"--experiment=native_preflight_{run.name}", f"--train_dir={output.parent / 'runtime'}",
        f"--dmlab_level_cache_path={output.parent / 'cache'}", f"--dmlab_runfiles_path={args.runfiles.resolve()}",
        "--dmlab_use_level_cache=False", "--with_wandb=False", "--device=cpu",
    ]
    cfg = parse_dmlab_args([*run.args, *overrides])
    if cfg.ca3_action_memory == "off":
        raise ValueError("Select an auxiliary-enabled row for the native predictor qualification")
    if not cfg.transfer_freeze_dg or cfg.transfer_calibrate_frozen_dg:
        raise ValueError("This qualification requires fixed DG weights and frozen BatchNorm statistics")
    source_provenance = {}
    if not args.local_calibrated_dg_fixture:
        source = Path(cfg.transfer_model_path).resolve(strict=True)
        configs = [parent / "config.json" for parent in source.parents if (parent / "config.json").is_file()]
        if not configs:
            raise ValueError("The learned DG source needs its saved configuration")
        source_cfg = json.loads(configs[0].read_text())
        fields = ("encoder_conv_architecture", "Hippo_n_feature", "DG_name", "DG_BN_intercept",
                  "with_number_instruction", "number_instruction_coef", "depth_sensor")
        mismatches = {key: (source_cfg.get(key), getattr(cfg, key))
                      for key in fields if source_cfg.get(key) != getattr(cfg, key)}
        if mismatches:
            raise ValueError(f"Source DG preprocessing/configuration differs: {mismatches}")
        source_provenance = {
            "checkpoint": str(source), "checkpoint_sha256": file_sha256(source),
            "config": str(configs[0]), "config_sha256": file_sha256(configs[0]),
            "matched_fields": {key: source_cfg[key] for key in fields},
        }
    register_dmlab_components()
    torch.manual_seed(run.seed)
    env = make_dmlab_env(cfg.env, cfg, None)
    try:
        # This runtime's reset(**kwargs) predates Gym's seed argument. Its
        # explicit seed method owns the native-engine seed stream.
        env.unwrapped.seed(run.seed)
        observation, _ = env.reset()
        native_reset_seeds = [int(env.unwrapped.last_reset_seed)]
        if int(observation["prev_action"][0]) != env.action_space.n:
            raise AssertionError("Reset previous-action sentinel differs from action count")
        observations = [observation]
        actions, dones, rewards = [], [], []
        for step in range(args.decisions):
            action = step % env.action_space.n
            observation, reward, done, truncated, _ = env.step(action)
            actions.append(action)
            dones.append(done or truncated)
            rewards.append(reward)
            if done or truncated:
                observation, _ = env.reset()
                native_reset_seeds.append(int(env.unwrapped.last_reset_seed))
            expected = env.action_space.n if done or truncated else action
            if int(observation["prev_action"][0]) != expected:
                raise AssertionError("Observed previous action differs from the actual executed action")
            observations.append(observation)
        actor = make_hipposlam_actor_critic(cfg, env.observation_space, env.action_space)
        learner = object.__new__(DistanceLearnerReward)
        learner.cfg = cfg
        learner.actor_critic = actor
        learner.device = torch.device("cpu")
        learner.timing = SimpleNamespace(add_time=lambda name: nullcontext())
        if not args.local_calibrated_dg_fixture:
            learner._initialize_transfer_weights()
        learner._apply_transfer_freeze()
        actor.train()
        projection = actor.encoder.DG_projection
        projection.calibrate_frozen_on_first_batch = args.local_calibrated_dg_fixture
        tensors = {
            key: torch.as_tensor(np.stack([observation[key] for observation in observations]))
            for key in observations[0]
        }
        normalized = prepare_and_normalize_obs(actor, tensors)
        assert "telemetry_pose" not in normalized
        with projection.running_stats_update(args.local_calibrated_dg_fixture):
            head = actor.forward_head(normalized)
        projection.calibrate_frozen_on_first_batch = False
        frozen = {name: tensor.clone() for name, tensor in actor.encoder.state_dict().items()}
        with torch.no_grad():
            visual = actor.encoder.projection_input(normalized)
            alone = projection(visual[:1])
            mixed = projection(torch.cat((visual[:1], visual[1:] * 3.0), dim=0))[:1]
            torch.testing.assert_close(alone, mixed, rtol=1e-5, atol=1e-5)
            batch_invariance_error = float((alone - mixed).abs().max())
        if not head[:, :cfg.Hippo_n_feature].gt(0).any():
            raise RuntimeError("Native panel has no active DG events, so transport gradients cannot be qualified")
        state = torch.zeros(1, actor.core.total_state_size)
        state_rows, outputs = [], []
        for step in range(len(head)):
            state_rows.append(state)
            value, state = actor.forward_core(head[step:step + 1], state)
            outputs.append(value)
            if step < len(dones) and dones[step]:
                state = torch.zeros_like(state)
        output_rows, state_rows = torch.cat(outputs), torch.cat(state_rows)
        valid = torch.ones(len(head), dtype=torch.bool)
        mb = AttrDict(
            rnn_states=state_rows, actions=torch.tensor(actions + [0]), dones=torch.tensor(dones + [False]),
            rewards=torch.tensor(rewards + [0.0]), rewards_encoder=torch.zeros(len(head), cfg.Hippo_n_feature),
            encoder_credit_activation_mask=torch.zeros(len(head), cfg.Hippo_n_feature, dtype=torch.bool),
            encoder_dominant_activation_mask=torch.zeros(len(head), cfg.Hippo_n_feature, dtype=torch.bool),
        )
        stats = {}
        loss = learner._calculate_fresh_encoder_loss(
            AttrDict(core_outputs=output_rows, head_outputs=head, minibatch_size=len(head)),
            mb, valid, 0, len(head), stats,
        )
        initial = {name: parameter.detach().clone() for name, parameter in actor.core.action_memory.named_parameters()}
        predictor_initial = {
            name: parameter.detach().clone() for name, parameter in actor.core.action_memory_prediction.named_parameters()
        }
        optimizer = torch.optim.Adam(actor.parameters(), lr=float(cfg.learning_rate))
        loss.backward()
        gradients = {
            name: float(parameter.grad.norm()) if parameter.grad is not None else 0.0
            for name, parameter in actor.core.action_memory.named_parameters()
        }
        predictor_gradients = {
            name: float(parameter.grad.norm()) if parameter.grad is not None else 0.0
            for name, parameter in actor.core.action_memory_prediction.named_parameters()
        }
        if not all(np.isfinite(value) and value > 0 for value in gradients.values()):
            raise RuntimeError(f"Native causal replay failed to train transport parameters: {gradients}")
        if not predictor_gradients or not all(np.isfinite(value) and value > 0 for value in predictor_gradients.values()):
            raise RuntimeError(f"Native prediction head lacks finite nonzero gradients: {predictor_gradients}")
        optimizer.step()
        assert all(torch.equal(value, actor.encoder.state_dict()[name]) for name, value in frozen.items())
        changed = {name: not torch.equal(initial[name], value) for name, value in actor.core.action_memory.named_parameters()}
        predictor_changed = {
            name: not torch.equal(predictor_initial[name], parameter)
            for name, parameter in actor.core.action_memory_prediction.named_parameters()
        }
        assert all(changed.values())
        assert all(predictor_changed.values())
        result = {
            **study.provenance(), "analysis_schema": "intrmotiv/action-native-preflight/v1",
            "run_name": run.name, "decisions": len(actions), "native_previous_action_alignment": True,
            "native_reset_seeds": native_reset_seeds, "source_provenance": source_provenance,
            "dg_source": "once_calibrated_random_local_fixture" if args.local_calibrated_dg_fixture else cfg.transfer_model_path,
            "declared_source_checkpoint_loaded": not args.local_calibrated_dg_fixture,
            "encoder_weight_and_buffer_equality_after_update": True,
            "frozen_bn_batch_invariance_max_error": batch_invariance_error,
            "dg_active_fraction": float(head[:, :cfg.Hippo_n_feature].gt(0).float().mean()),
            "loss": float(loss.detach()), "transport_gradient_norms": gradients,
            "transport_parameters_changed": changed,
            "predictor_gradient_norms": predictor_gradients, "predictor_parameters_changed": predictor_changed,
            "prediction_stats": {name: float(value.detach()) for name, value in stats.items()
                                 if name.startswith("ca3_action_memory_")},
            "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "qualification_scope": "native observations and one actual learner auxiliary update; no production run or control result",
        }
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2, allow_nan=False))
    finally:
        env.close()


if __name__ == "__main__":
    main()
