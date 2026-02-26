"""
Profile Hanabi environment with JAX profiler using a dummy random policy.
This isolates the environment performance from the training algorithm.
"""
import jax
import jax.numpy as jnp
import time
import os

from envs import make_env
from envs.log_wrapper import LogWrapper


def profile_hanabi_env(num_envs=8, num_steps=128, trace_dir="/tmp/jax-trace-hanabi-env"):
    """Profile Hanabi environment steps with JAX profiler."""
    
    os.makedirs(trace_dir, exist_ok=True)
    
    # Use raw JaxMARL HanabiEnv directly to avoid auto-reset wrappers
    from jaxmarl.environments.hanabi.hanabi import HanabiEnv
    env = HanabiEnv(num_agents=2)
    
    num_agents = env.num_agents
    
    print("=" * 60)
    print("Hanabi Environment Profiling (with JAX Profiler)")
    print("=" * 60)
    print(f"NUM_ENVS: {num_envs}")
    print(f"NUM_AGENTS: {num_agents}")
    print(f"NUM_STEPS: {num_steps}")
    print("=" * 60)
    
    # JIT compile the env functions
    reset_jit = jax.jit(jax.vmap(env.reset))
    step_jit = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0)))
    get_avail_jit = jax.jit(jax.vmap(env.get_avail_actions))
    
    # Dummy random policy - just samples from available actions
    @jax.jit
    def dummy_policy(rng, avail_actions):
        """Sample random valid actions for all agents."""
        actions = {}
        for i, agent in enumerate(env.agents):
            agent_rng = jax.random.fold_in(rng, i)
            avail = avail_actions[agent]  # (num_envs, num_actions)
            # Sample uniformly from available actions
            logits = jnp.where(avail, 0.0, -1e9)
            actions[agent] = jax.random.categorical(agent_rng, logits, axis=-1)
        return actions
    
    # Single step function for scan (raw JaxMARL env, no auto-reset)
    def env_step_with_dummy_policy(carry, unused):
        env_state, rng = carry
        
        rng, policy_rng, step_rng = jax.random.split(rng, 3)
        
        # Get available actions directly from raw env
        with jax.named_scope("get_avail_actions"):
            avail_actions = jax.vmap(env.get_legal_moves)(env_state)
        
        # Sample random actions
        with jax.named_scope("dummy_policy"):
            actions = dummy_policy(policy_rng, avail_actions)
        
        # Step environment - raw JaxMARL step (3 args, no reset_state)
        with jax.named_scope("env_step"):
            step_rngs = jax.random.split(step_rng, num_envs)
            obs, new_state, rewards, dones, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                step_rngs, env_state, actions
            )
        
        return (new_state, rng), (rewards, dones)
    
    # Full rollout function (WITHOUT reset - separate for profiling visibility)
    @jax.jit
    def run_rollout_only(env_state, rng):
        with jax.named_scope("rollout_scan"):
            (final_state, _), (all_rewards, all_dones) = jax.lax.scan(
                env_step_with_dummy_policy,
                (env_state, rng),
                None,
                length=num_steps
            )
        return final_state, all_rewards, all_dones
    
    # Reset function (separate JIT for profiling visibility)
    @jax.jit 
    def run_reset(rng):
        with jax.named_scope("reset_envs"):
            reset_rngs = jax.random.split(rng, num_envs)
            obs, env_state = jax.vmap(env.reset)(reset_rngs)
        return obs, env_state
    
    # Combined function for comparison
    @jax.jit
    def run_rollout(rng):
        reset_rng, rollout_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, num_envs)
        obs, env_state = jax.vmap(env.reset)(reset_rngs)
        
        (final_state, _), (all_rewards, all_dones) = jax.lax.scan(
            env_step_with_dummy_policy,
            (env_state, rollout_rng),
            None,
            length=num_steps
        )
        return final_state, all_rewards, all_dones
    
    # Warmup run
    print("\nWarming up (JIT compilation)...")
    rng = jax.random.PRNGKey(0)
    st = time.time()
    # Warmup both functions separately
    reset_rng, rollout_rng = jax.random.split(rng)
    obs, env_state = run_reset(reset_rng)
    jax.tree.map(lambda x: x.block_until_ready(), obs)
    final_state, all_rewards, all_dones = run_rollout_only(env_state, rollout_rng)
    jax.tree.map(lambda x: x.block_until_ready(), all_dones)
    warmup_time = time.time() - st
    print(f"Warmup time: {warmup_time:.2f}s")
    
    # Profiled run - use configurable trace settings for cleaner analysis
    trace_envs = 4  # Number of parallel envs in trace (4 games running)
    trace_steps = 20  # Steps per env (should be enough for ~1 episode each)
    
    # Create env/functions specifically for the trace
    @jax.jit
    def run_reset_trace(rng):
        with jax.named_scope("reset_envs"):
            reset_rngs = jax.random.split(rng, trace_envs)
            obs, env_state = jax.vmap(env.reset)(reset_rngs)
        return obs, env_state
    
    def env_step_trace(carry, unused):
        env_state, rng, reset_count = carry
        rng, policy_rng, step_rng = jax.random.split(rng, 3)
        
        with jax.named_scope("get_avail_actions"):
            avail_actions = jax.vmap(env.get_legal_moves)(env_state)
        
        with jax.named_scope("dummy_policy"):
            actions = {}
            for i, agent in enumerate(env.agents):
                agent_rng = jax.random.fold_in(policy_rng, i)
                avail = avail_actions[agent]
                logits = jnp.where(avail, 0.0, -1e9)
                actions[agent] = jax.random.categorical(agent_rng, logits, axis=-1)
        
        with jax.named_scope("env_step"):
            step_rngs = jax.random.split(step_rng, trace_envs)
            obs, new_state, rewards, dones, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                step_rngs, env_state, actions
            )
        
        # Count auto-resets (when episode ends, env auto-resets)
        new_reset_count = reset_count + jnp.sum(dones["__all__"])
        
        return (new_state, rng, new_reset_count), dones
    
    @jax.jit
    def run_rollout_trace(env_state, rng):
        with jax.named_scope("rollout_scan"):
            (final_state, _, reset_count), all_dones = jax.lax.scan(
                env_step_trace,
                (env_state, rng, 0),
                None,
                length=trace_steps
            )
        return final_state, all_dones, reset_count
    
    # Warmup trace functions
    reset_rng2, rollout_rng2 = jax.random.split(jax.random.PRNGKey(99))
    obs2, env_state2 = run_reset_trace(reset_rng2)
    run_rollout_trace(env_state2, rollout_rng2)
    
    print(f"\nRunning with JAX profiler ({trace_envs} envs x {trace_steps} steps)...")
    print(f"Trace saved to: {trace_dir}")
    rng = jax.random.PRNGKey(1)
    reset_rng, rollout_rng = jax.random.split(rng)
    
    st = time.time()
    with jax.profiler.trace(trace_dir):
        obs, env_state = run_reset_trace(reset_rng)
        jax.tree.map(lambda x: x.block_until_ready(), obs)
        
        final_state, all_dones, reset_count = run_rollout_trace(env_state, rollout_rng)
        jax.tree.map(lambda x: x.block_until_ready(), all_dones)
    profiled_time = time.time() - st
    print(f"Profiled run time: {profiled_time:.2f}s")
    
    # Print episode stats from trace
    trace_episodes = int(jnp.sum(all_dones["__all__"]))
    print(f"Episodes completed in trace: {trace_episodes}")
    print(f"Auto-resets (reset_game calls) during rollout: {int(reset_count)}")
    
    # Compute average episode length from dones
    # all_dones["__all__"] has shape (num_steps, num_envs)
    episode_ends = all_dones["__all__"]  # True when episode ends
    total_episodes = jnp.sum(episode_ends)
    total_steps = num_steps * num_envs
    avg_episode_length = total_steps / jnp.maximum(total_episodes, 1)
    
    print(f"\n--- Episode Statistics ---")
    print(f"Total episodes completed: {int(total_episodes)}")
    print(f"Average episode length: {float(avg_episode_length):.1f} steps")
    print(f"Episodes per env: {float(total_episodes / num_envs):.1f}")
    
    # Multiple runs for timing
    print("\nTiming multiple runs...")
    times = []
    for i in range(10):
        rng = jax.random.PRNGKey(i + 100)
        st = time.time()
        final_state, all_rewards, all_dones = run_rollout(rng)
        all_dones["__all__"].block_until_ready()
        times.append(time.time() - st)
    
    avg_time = sum(times) / len(times)
    print(f"Average rollout time ({num_steps} steps): {avg_time*1000:.2f} ms")
    print(f"Per-step time: {avg_time/num_steps*1000:.3f} ms")
    
    print("\n" + "=" * 60)
    print(f"To view the trace, run:")
    print(f"  tensorboard --logdir={trace_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--trace_dir", type=str, default="/tmp/jax-trace-hanabi-env")
    args = parser.parse_args()
    
    profile_hanabi_env(
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        trace_dir=args.trace_dir
    )
