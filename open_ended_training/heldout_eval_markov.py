"""
Heldout evaluation for lbf-fov-3-markov.

The ego agent was trained on the markov env (obs_dim=30), but heldout agents
were trained on the base lbf-fov-3 env (obs_dim=15). This module wraps
heldout agent policies to truncate the observation to the first 15 dims
before passing it to the heldout agent.

Standalone usage:
    python -m open_ended_training.heldout_eval_markov --checkpoint <path_to_saved_train_run>
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import logging
import time
from functools import partial

import jax
import jax.numpy as jnp

from common.run_episodes import run_episodes
from common.tree_utils import tree_stack
from common.stat_utils import compute_aggregate_stat_and_ci
from evaluation.heldout_evaluator import load_heldout_set, normalize_metrics

log = logging.getLogger(__name__)

BASE_OBS_DIM = 15  # obs dim of the base lbf-fov-3 env


class ObsTruncatingPolicyWrapper:
    """Wraps a heldout policy to truncate obs from markov env (30 dims) to base env (15 dims)."""

    def __init__(self, inner_policy, base_obs_dim=BASE_OBS_DIM):
        self.inner_policy = inner_policy
        self.base_obs_dim = base_obs_dim

    def _truncate_obs(self, obs):
        return obs[..., :self.base_obs_dim]

    def _unwrap_env_state(self, env_state):
        """Unwrap BeliefWrappedState from inside LogEnvState.

        run_episodes passes LogEnvState whose .env_state is BeliefWrappedState.
        Heldout agents expect LogEnvState whose .env_state is WrappedEnvState.
        We replace the BeliefWrappedState with its inner WrappedEnvState.
        """
        if env_state is None:
            return env_state
        from envs.lbf.belief_wrapper import BeliefWrappedState
        from envs.log_wrapper import LogEnvState
        # env_state is LogEnvState; env_state.env_state may be BeliefWrappedState
        inner = env_state.env_state
        if isinstance(inner, BeliefWrappedState):
            return env_state.replace(env_state=inner.env_state)
        return env_state

    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        obs = self._truncate_obs(obs)
        env_state = self._unwrap_env_state(env_state)
        return self.inner_policy.get_action(
            params=params, obs=obs, done=done, avail_actions=avail_actions,
            hstate=hstate, rng=rng, aux_obs=aux_obs, env_state=env_state,
            test_mode=test_mode,
        )

    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        obs = self._truncate_obs(obs)
        env_state = self._unwrap_env_state(env_state)
        return self.inner_policy.get_action_value_policy(
            params=params, obs=obs, done=done, avail_actions=avail_actions,
            hstate=hstate, rng=rng, aux_obs=aux_obs, env_state=env_state,
        )

    def init_hstate(self, batch_size, aux_info=None):
        return self.inner_policy.init_hstate(batch_size, aux_info=aux_info)

    def init_params(self, rng):
        return self.inner_policy.init_params(rng)


def eval_markov_ego_vs_heldouts(config, env, rng, num_episodes, ego_policy, ego_params,
                                heldout_agent_list, ego_test_mode=False):
    """Evaluate markov ego agents against base-env heldout partners.

    Same as eval_2d_egos_vs_heldouts but wraps heldout policies to truncate obs.
    """
    num_ego_seeds, num_ego_iters = jax.tree.leaves(ego_params)[0].shape[:2]
    tot_ego_agents = num_ego_seeds * num_ego_iters
    num_partner_total = len(heldout_agent_list)

    def _eval_ego_vs_one_partner(rng_for_ego, single_ego_params, single_ego_policy,
                                 heldout_params, heldout_policy, heldout_test_mode):
        return run_episodes(rng_for_ego, env,
                            agent_0_policy=single_ego_policy, agent_0_param=single_ego_params,
                            agent_1_policy=heldout_policy, agent_1_param=heldout_params,
                            max_episode_steps=config["global_heldout_settings"]["MAX_EPISODE_STEPS"],
                            num_eps=num_episodes,
                            agent_0_test_mode=ego_test_mode,
                            agent_1_test_mode=heldout_test_mode)

    all_metrics_for_partners = []
    partner_rngs = jax.random.split(rng, num_partner_total)

    for partner_idx in range(num_partner_total):
        heldout_policy, heldout_params, heldout_test_mode, heldout_performance_bounds = heldout_agent_list[partner_idx]

        # Wrap the heldout policy to truncate obs
        wrapped_heldout_policy = ObsTruncatingPolicyWrapper(heldout_policy)

        ego_rngs = jax.random.split(partner_rngs[partner_idx], tot_ego_agents)
        ego_rngs = ego_rngs.reshape(num_ego_seeds, num_ego_iters, 2)

        func_to_vmap = partial(_eval_ego_vs_one_partner,
                               single_ego_policy=ego_policy,
                               heldout_params=heldout_params,
                               heldout_policy=wrapped_heldout_policy,
                               heldout_test_mode=heldout_test_mode)

        vmap_over_iters = jax.vmap(func_to_vmap, in_axes=(0, 0))
        vmap_over_seeds_and_iters = jax.vmap(vmap_over_iters, in_axes=(0, 0))

        results_for_this_partner = vmap_over_seeds_and_iters(ego_rngs, ego_params)

        if config["global_heldout_settings"]["NORMALIZE_RETURNS"]:
            if heldout_performance_bounds is not None:
                results_for_this_partner = normalize_metrics(results_for_this_partner, heldout_performance_bounds)

        all_metrics_for_partners.append(results_for_this_partner)

    final_metrics = tree_stack(all_metrics_for_partners)
    final_metrics = jax.tree.map(lambda x: x.transpose(1, 2, 0, 3, 4), final_metrics)
    return final_metrics


def run_markov_heldout_evaluation(config, ego_policy, ego_params, init_ego_params,
                                  ego_test_mode=False):
    """Run heldout evaluation for lbf-fov-3-markov.

    Uses the lbf-fov-3 heldout set but runs on the markov env,
    wrapping heldout agents to see only the base 15-dim obs.
    """
    from envs import make_env
    from envs.log_wrapper import LogWrapper

    log.info("Running markov heldout evaluation...")

    # Create the markov env for evaluation
    env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(config["global_heldout_settings"]["EVAL_SEED"])
    rng, heldout_init_rng, eval_rng = jax.random.split(rng, 3)

    # Load heldout agents from the base lbf-fov-3 set
    base_task = "lbf-fov-3"
    heldout_cfg = config["heldout_set"][base_task]

    # Create a base env for loading heldout agents (they need the base obs dim)
    base_env = make_env(base_task)
    base_env = LogWrapper(base_env)
    heldout_agents = load_heldout_set(heldout_cfg, base_env, base_task, {}, heldout_init_rng)
    heldout_agent_list = list(heldout_agents.values())
    heldout_names = list(heldout_agents.keys())

    num_seeds, num_oel_iters = jax.tree.leaves(ego_params)[0].shape[:2]
    ego_names = [f"ego (seed={i}, iter={j})" for i in range(num_seeds) for j in range(num_oel_iters)]

    eval_metrics = eval_markov_ego_vs_heldouts(
        config, env, eval_rng,
        config["global_heldout_settings"]["NUM_EVAL_EPISODES"],
        ego_policy, ego_params, heldout_agent_list, ego_test_mode,
    )

    return eval_metrics, ego_names, heldout_names


def eval_single_ego_vs_heldouts(env, base_env, rng, num_episodes, ego_policy, ego_params,
                                heldout_agent_list, max_episode_steps, ego_test_mode=False):
    """Evaluate a single ego agent against all heldout partners.

    Unlike eval_markov_ego_vs_heldouts, this does NOT vmap over seeds/iters —
    ego_params is a single set of params (no batch dims).
    """
    num_partner_total = len(heldout_agent_list)
    partner_rngs = jax.random.split(rng, num_partner_total)
    results = {}

    for partner_idx in range(num_partner_total):
        heldout_policy, heldout_params, heldout_test_mode, _ = heldout_agent_list[partner_idx]
        wrapped_heldout_policy = ObsTruncatingPolicyWrapper(heldout_policy)

        metrics = jax.jit(lambda r: run_episodes(
            r, env,
            agent_0_policy=ego_policy, agent_0_param=ego_params,
            agent_1_policy=wrapped_heldout_policy, agent_1_param=heldout_params,
            max_episode_steps=max_episode_steps,
            num_eps=num_episodes,
            agent_0_test_mode=ego_test_mode,
            agent_1_test_mode=heldout_test_mode,
        ))(partner_rngs[partner_idx])

        results[partner_idx] = metrics

    return results


def main():
    parser = argparse.ArgumentParser(description="Heldout evaluation for lbf-fov-3-markov")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to saved_train_run checkpoint directory")
    parser.add_argument("--num_episodes", type=int, default=1024,
                        help="Number of evaluation episodes per heldout partner")
    parser.add_argument("--max_steps", type=int, default=128,
                        help="Max episode steps")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed index to evaluate (from multi-seed training)")
    parser.add_argument("--iter", type=int, default=-1,
                        help="OEL iteration index to evaluate (-1 = last)")
    parser.add_argument("--ego_test_mode", action="store_true",
                        help="Use argmax actions for ego (deterministic)")
    parser.add_argument("--eval_seed", type=int, default=34957,
                        help="RNG seed for evaluation")
    parser.add_argument("--fc_hidden_dim", type=int, default=256,
                        help="Hidden dim for MLP ego policy")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    from agents.agent_interface import MLPActorCriticPolicy
    from common.save_load_utils import load_train_run
    from envs import make_env
    from envs.log_wrapper import LogWrapper

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    outs = load_train_run(args.checkpoint)

    # Extract ego params: outs = (partner_outs, ego_outs) with shape (num_seeds, num_iters, ...)
    _, ego_outs = outs
    all_ego_params = ego_outs["final_params"]
    # Shape: (num_seeds, num_iters, 1, ...) -> squeeze the pop dim
    all_ego_params = jax.tree_map(lambda x: x[:, :, 0], all_ego_params)

    num_seeds = jax.tree.leaves(all_ego_params)[0].shape[0]
    num_iters = jax.tree.leaves(all_ego_params)[0].shape[1]
    print(f"Checkpoint has {num_seeds} seed(s), {num_iters} iteration(s)")

    # Select which ego to evaluate
    seed_idx = args.seed
    iter_idx = args.iter if args.iter >= 0 else num_iters - 1
    print(f"Evaluating seed={seed_idx}, iter={iter_idx}")

    ego_params = jax.tree_map(lambda x: x[seed_idx, iter_idx], all_ego_params)

    # Create markov env
    env = make_env("lbf-fov-3-markov")
    env = LogWrapper(env)

    obs_dim = env.observation_space(env.agents[0]).shape[0]
    action_dim = env.action_space(env.agents[0]).n

    ego_policy = MLPActorCriticPolicy(
        action_dim=action_dim, obs_dim=obs_dim, fc_hidden_dim=args.fc_hidden_dim,
    )

    # Create base env and load heldout agents
    base_env = make_env("lbf-fov-3")
    base_env = LogWrapper(base_env)

    # Load heldout config manually (same as global_heldout_settings.yaml lbf-fov-3 section)
    import yaml
    heldout_settings_path = os.path.join(
        os.path.dirname(__file__), "..", "evaluation", "configs", "global_heldout_settings.yaml"
    )
    with open(heldout_settings_path, "r") as f:
        heldout_settings = yaml.safe_load(f)
    heldout_cfg = heldout_settings["heldout_set"]["lbf-fov-3"]

    rng = jax.random.PRNGKey(args.eval_seed)
    rng, heldout_init_rng, eval_rng = jax.random.split(rng, 3)

    heldout_agents = load_heldout_set(heldout_cfg, base_env, "lbf-fov-3", {}, heldout_init_rng)
    heldout_agent_list = list(heldout_agents.values())
    heldout_names = list(heldout_agents.keys())

    print(f"\nEvaluating against {len(heldout_names)} heldout partners:")
    for name in heldout_names:
        print(f"  - {name}")

    # Run evaluation
    print(f"\nRunning {args.num_episodes} episodes per partner...")
    start_time = time.time()

    results = eval_single_ego_vs_heldouts(
        env, base_env, eval_rng, args.num_episodes,
        ego_policy, ego_params, heldout_agent_list,
        max_episode_steps=args.max_steps,
        ego_test_mode=args.ego_test_mode,
    )

    elapsed = time.time() - start_time
    print(f"Evaluation completed in {elapsed:.1f}s\n")

    # Print results
    print(f"{'Partner':<30} {'Mean Return':>12} {'Mean % Eaten':>12}")
    print("-" * 56)
    for partner_idx, name in enumerate(heldout_names):
        metrics = results[partner_idx]
        mean_return = float(jnp.mean(metrics["returned_episode_returns"]))
        if "percent_eaten" in metrics:
            mean_eaten = float(jnp.mean(metrics["percent_eaten"]))
            print(f"{name:<30} {mean_return:>12.4f} {mean_eaten:>11.2f}%")
        else:
            print(f"{name:<30} {mean_return:>12.4f} {'N/A':>12}")


if __name__ == "__main__":
    main()
