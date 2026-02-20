"""MaskablePPO training for Catan with BC-initialized policy-value network.

Usage:
    # Phase A: vs WeightedRandom
    python3 -m robottler.train_rl --opponent weighted --total-steps 500000 \
        --bc-model robottler/models/value_net_v2.pt

    # Phase B: vs AlphaBeta (resume from Phase A)
    python3 -m robottler.train_rl --opponent alphabeta --total-steps 2000000 \
        --resume robottler/models/rl_checkpoints/phase_a.zip

    # Full curriculum (automated Phases A→B)
    python3 -m robottler.train_rl --curriculum --bc-model robottler/models/value_net_v2.pt
"""

import argparse
import os
import time

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    CallbackList,
)

from robottler.catan_env import make_env, make_vec_env, make_mixed_vec_env
from robottler.policy_value_net import (
    CatanFeatureExtractor,
    load_bc_weights,
    load_bc_policy_weights,
)


CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "models", "rl_checkpoints")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class EvalWinRateCallback(BaseCallback):
    """Periodically evaluate win rate against reference opponents."""

    def __init__(self, bc_path, eval_every=50_000, n_eval_games=100,
                 opponents=("random", "weighted"), vps_to_win=10, verbose=1):
        super().__init__(verbose)
        self.bc_path = bc_path
        self.eval_every = eval_every
        self.n_eval_games = n_eval_games
        self.opponents = opponents
        self.vps_to_win = vps_to_win
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_every:
            return True
        self._last_eval_step = self.num_timesteps
        self._run_eval()
        return True

    def _run_eval(self):
        """Play games against each opponent and log win rates."""
        from robottler.catan_env import (
            StrategicCatanEnv, _make_enemy, _win_loss_reward,
            _preload_search_value_fn,
        )
        from catanatron.gym.envs.catanatron_env import CatanatronEnv

        for opp_kind in self.opponents:
            wins = 0
            losses = 0
            draws = 0

            # Pre-load value fn once per opponent type (avoids N torch.load calls)
            shared_vfn = _preload_search_value_fn(opp_kind, self.bc_path)

            for _ in range(self.n_eval_games):
                enemy = _make_enemy(opp_kind, bc_path=self.bc_path,
                                    shared_value_fn=shared_vfn)
                base_env = CatanatronEnv(config={
                    "enemies": [enemy],
                    "vps_to_win": self.vps_to_win,
                    "representation": "vector",
                    "reward_function": _win_loss_reward,
                })
                eval_env = StrategicCatanEnv(base_env, self.bc_path)

                obs, info = eval_env.reset()
                done = False
                while not done:
                    mask = eval_env.action_masks()
                    action, _ = self.model.predict(obs, deterministic=True,
                                                   action_masks=mask)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated

                if reward > 0:
                    wins += 1
                elif reward < 0:
                    losses += 1
                else:
                    draws += 1

            wr = wins / self.n_eval_games
            self.logger.record(f"eval/win_rate_vs_{opp_kind}", wr)
            self.logger.record(f"eval/wins_vs_{opp_kind}", wins)
            if self.verbose:
                print(f"  [Step {self.num_timesteps}] vs {opp_kind}: "
                      f"{wr*100:.1f}% ({wins}W/{losses}L/{draws}D)")


class ValueHeadWarmupCallback(BaseCallback):
    """Suppress value head loss for the first N steps so policy can bootstrap.

    The BC-initialized value head produces near-perfect value estimates from
    step 0, which flattens advantages and starves the policy gradient of
    signal. Setting vf_coef=0 temporarily lets the value estimates degrade
    slightly, creating advantage variance the policy can learn from.

    After warmup_steps, restores vf_coef so the value head can re-calibrate
    to the (now-improving) policy's actual trajectories.
    """

    def __init__(self, warmup_steps=100_000, target_vf_coef=0.5, verbose=1):
        super().__init__(verbose)
        self.warmup_steps = warmup_steps
        self.target_vf_coef = target_vf_coef
        self._activated = False

    def _on_training_start(self) -> None:
        self.model.vf_coef = 0.0
        if self.verbose:
            print(f"  [VF Warmup] vf_coef=0.0 for first {self.warmup_steps} steps")

    def _on_step(self) -> bool:
        if not self._activated and self.num_timesteps >= self.warmup_steps:
            self.model.vf_coef = self.target_vf_coef
            self._activated = True
            if self.verbose:
                print(f"  [VF Warmup] Step {self.num_timesteps}: "
                      f"restoring vf_coef={self.target_vf_coef}")
        return True


class EntropyMonitorCallback(BaseCallback):
    """Warn if policy entropy collapses."""

    def __init__(self, warn_threshold=0.1, check_every=10_000, verbose=1):
        super().__init__(verbose)
        self.warn_threshold = warn_threshold
        self.check_every = check_every
        self._last_check = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_check < self.check_every:
            return True
        self._last_check = self.num_timesteps

        # sb3 logs entropy as "train/entropy_loss" (negative entropy * ent_coef)
        # We check the logger for the latest value
        if hasattr(self.model, "logger") and self.model.logger is not None:
            ent = self.logger.name_to_value.get("train/entropy_loss")
            if ent is not None:
                # entropy_loss = -ent_coef * entropy, so entropy = -ent / ent_coef
                ent_coef = self.model.ent_coef
                if ent_coef > 0:
                    entropy = -ent / ent_coef
                    self.logger.record("monitor/policy_entropy", entropy)
                    if entropy < self.warn_threshold and self.verbose:
                        print(f"  WARNING: Policy entropy={entropy:.4f} < {self.warn_threshold}")
        return True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_model(bc_path, env, resume_path=None, ent_coef=0.05, vf_coef=0.5,
                bc_policy_path=None):
    """Build MaskablePPO, optionally resuming from a checkpoint."""
    policy_kwargs = {
        "features_extractor_class": CatanFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
        "net_arch": dict(pi=[64], vf=[64]),
    }

    if resume_path:
        print(f"Resuming from {resume_path}")
        model = MaskablePPO.load(resume_path, env=env)
        return model

    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        n_steps=256,
        batch_size=2048,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        learning_rate=3e-4,
        max_grad_norm=0.5,
        verbose=1,
    )

    if bc_policy_path:
        print("Loading BC value + policy weights...")
        load_bc_policy_weights(model, bc_path, bc_policy_path)
    else:
        print("Loading BC value weights (policy head random)...")
        load_bc_weights(model, bc_path)
    return model


def train_phase(model, total_steps, bc_path, phase_name="train",
                eval_opponents=("random", "weighted"), vps_to_win=10,
                vf_warmup_steps=100_000):
    """Train for total_steps with eval + checkpoint callbacks."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    cb_list = [
        ValueHeadWarmupCallback(
            warmup_steps=vf_warmup_steps,
            target_vf_coef=model.vf_coef,
        ),
        EvalWinRateCallback(
            bc_path=bc_path,
            eval_every=50_000,
            n_eval_games=100,
            opponents=eval_opponents,
            vps_to_win=vps_to_win,
        ),
        CheckpointCallback(
            save_freq=max(100_000 // model.n_envs, 1),
            save_path=CHECKPOINT_DIR,
            name_prefix=phase_name,
        ),
        EntropyMonitorCallback(),
    ]
    callbacks = CallbackList(cb_list)

    print(f"\nStarting training: {phase_name} for {total_steps} steps")
    start = time.time()
    model.learn(total_timesteps=total_steps, callback=callbacks)
    elapsed = time.time() - start
    print(f"Training complete in {elapsed:.0f}s")

    # Save final checkpoint
    final_path = os.path.join(CHECKPOINT_DIR, f"{phase_name}.zip")
    model.save(final_path)
    print(f"Saved to {final_path}")
    return model, final_path


def run_curriculum(bc_path, vps_to_win=10, n_envs=8, ent_coef=0.05,
                   vf_warmup_steps=100_000, bc_policy_path=None):
    """Automated curriculum: Phase A (vs weighted) → Phase B (vs value net)."""

    # Phase A: vs WeightedRandom, 500k steps
    print("=" * 60)
    print("PHASE A: vs WeightedRandom (500k steps)")
    print("=" * 60)
    env_a = make_vec_env("weighted", vps_to_win=vps_to_win, bc_path=bc_path, n_envs=n_envs)
    model = build_model(bc_path, env_a, ent_coef=ent_coef,
                        bc_policy_path=bc_policy_path)
    model, path_a = train_phase(
        model, total_steps=500_000, bc_path=bc_path,
        phase_name="phase_a",
        eval_opponents=("random", "weighted", "alphabeta"),
        vps_to_win=vps_to_win,
        vf_warmup_steps=vf_warmup_steps,
    )
    env_a.close()

    # Phase B: vs ValueNet, 2M steps (no warmup — policy is bootstrapped)
    # Uses BC value net with 1-ply lookahead (~AlphaBeta strength, ~10x faster)
    print("\n" + "=" * 60)
    print("PHASE B: vs ValueNet (2M steps)")
    print("=" * 60)
    env_b = make_vec_env("value", vps_to_win=vps_to_win, bc_path=bc_path, n_envs=n_envs)
    model.set_env(env_b)
    model, path_b = train_phase(
        model, total_steps=2_000_000, bc_path=bc_path,
        phase_name="phase_b",
        eval_opponents=("random", "weighted", "value", "alphabeta"),
        vps_to_win=vps_to_win,
        vf_warmup_steps=0,
    )
    env_b.close()

    # Copy final as "latest"
    latest_path = os.path.join(CHECKPOINT_DIR, "latest.zip")
    model.save(latest_path)
    print(f"\nCurriculum complete. Final model: {latest_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Catan RL agent with MaskablePPO")
    parser.add_argument("--opponent", type=str, default="weighted",
                        help="Opponent type: random, weighted, alphabeta, value, search, search:<depth>, or mixed (default: weighted)")
    parser.add_argument("--total-steps", type=int, default=500_000,
                        help="Total training steps (default: 500000)")
    parser.add_argument("--bc-model", type=str, default="robottler/models/value_net_v2.pt",
                        help="Path to BC v2 checkpoint for weight init + normalization")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to RL checkpoint (.zip) to resume from")
    parser.add_argument("--curriculum", action="store_true",
                        help="Run automated curriculum (Phase A→B)")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel environments (default: 8)")
    parser.add_argument("--vps", type=int, default=10,
                        help="Victory points to win (default: 10)")
    parser.add_argument("--ent-coef", type=float, default=0.05,
                        help="Entropy coefficient (default: 0.05, higher=more exploration)")
    parser.add_argument("--vf-warmup", type=int, default=100_000,
                        help="Steps with vf_coef=0 to let policy bootstrap (default: 100000, 0=disable)")
    parser.add_argument("--bc-policy-model", type=str, default=None,
                        help="Path to BC policy checkpoint for policy head init")
    parser.add_argument("--eval-opponents", type=str, nargs="+",
                        default=["random", "weighted", "alphabeta"],
                        help="Opponents to eval against (default: random weighted alphabeta)")
    parser.add_argument("--opponent-pool", type=str, nargs="+", default=None,
                        help="Opponent pool for --opponent mixed, e.g.: "
                             "weighted value rl:path/to/checkpoint.zip")
    args = parser.parse_args()

    if args.curriculum:
        run_curriculum(args.bc_model, vps_to_win=args.vps, n_envs=args.n_envs,
                       ent_coef=args.ent_coef, vf_warmup_steps=args.vf_warmup,
                       bc_policy_path=args.bc_policy_model)
        return

    # Single-phase training
    if args.opponent == "mixed":
        if not args.opponent_pool:
            raise ValueError("--opponent mixed requires --opponent-pool")
        env = make_mixed_vec_env(
            args.opponent_pool, vps_to_win=args.vps,
            bc_path=args.bc_model, n_envs=args.n_envs,
        )
    else:
        env = make_vec_env(
            args.opponent, vps_to_win=args.vps,
            bc_path=args.bc_model, n_envs=args.n_envs,
        )
    model = build_model(args.bc_model, env, resume_path=args.resume,
                        ent_coef=args.ent_coef,
                        bc_policy_path=args.bc_policy_model)

    phase_name = f"rl_{args.opponent}"
    train_phase(
        model, total_steps=args.total_steps, bc_path=args.bc_model,
        phase_name=phase_name,
        eval_opponents=tuple(args.eval_opponents),
        vps_to_win=args.vps,
        vf_warmup_steps=args.vf_warmup,
    )
    env.close()


if __name__ == "__main__":
    main()
