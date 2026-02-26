"""
Simple profiling script to measure time breakdown in IPPO training.
This profiles each phase separately to understand where time is spent.
"""
import jax
import jax.numpy as jnp
import time
import hydra
from omegaconf import DictConfig

from envs import make_env
from envs.log_wrapper import LogWrapper
from agents.initialize_agents import initialize_mlp_agent
from marl.ppo_utils import Transition, batchify, unbatchify, _create_minibatches
import optax
from flax.training.train_state import TrainState


def profile_components(config):
    """Profile individual components of IPPO training."""
    
    algorithm_config = dict(config.algorithm)
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)
    
    # Setup
    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    
    num_envs = algorithm_config["NUM_ENVS"]
    num_actors = env.num_agents * num_envs
    rollout_length = algorithm_config["ROLLOUT_LENGTH"]
    
    # Initialize policy
    rng, init_rng = jax.random.split(rng)
    policy, init_params = initialize_mlp_agent(algorithm_config, env, init_rng)
    
    tx = optax.chain(
        optax.clip_by_global_norm(algorithm_config["MAX_GRAD_NORM"]),
        optax.adam(algorithm_config["LR"], eps=1e-5)
    )
    train_state = TrainState.create(
        apply_fn=policy.network.apply,
        params=init_params,
        tx=tx,
    )
    
    # Initialize env
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, num_envs)
    
    print("=" * 60)
    print("IPPO Component Profiling")
    print("=" * 60)
    print(f"NUM_ENVS: {num_envs}")
    print(f"NUM_ACTORS: {num_actors}")
    print(f"ROLLOUT_LENGTH: {rollout_length}")
    print(f"UPDATE_EPOCHS: {algorithm_config['UPDATE_EPOCHS']}")
    print(f"NUM_MINIBATCHES: {algorithm_config['NUM_MINIBATCHES']}")
    print("=" * 60)
    
    # Profile env.reset
    reset_jit = jax.jit(jax.vmap(env.reset))
    
    # Warmup
    obsv, env_state = reset_jit(reset_rngs)
    obsv["player_0"].block_until_ready()
    
    st = time.time()
    for _ in range(10):
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, num_envs)
        obsv, env_state = reset_jit(reset_rngs)
        obsv["player_0"].block_until_ready()
    reset_time = (time.time() - st) / 10
    print(f"\n1. env.reset (x{num_envs} envs): {reset_time*1000:.3f} ms")
    
    # Profile single env.step
    done = {k: jnp.zeros((num_envs,), dtype=bool) for k in env.agents + ["__all__"]}
    hstate = policy.init_hstate(num_actors)
    
    step_jit = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0)))
    get_avail_jit = jax.jit(jax.vmap(env.get_avail_actions))
    
    # Warmup
    rng, step_rng = jax.random.split(rng)
    step_rngs = jax.random.split(step_rng, num_envs)
    avail_actions = get_avail_jit(env_state.env_state)
    dummy_actions = {agent: jnp.zeros((num_envs,), dtype=jnp.int32) for agent in env.agents}
    new_obsv, new_state, rewards, dones, info = step_jit(step_rngs, env_state, dummy_actions)
    new_obsv["player_0"].block_until_ready()
    
    st = time.time()
    for _ in range(100):
        rng, step_rng = jax.random.split(rng)
        step_rngs = jax.random.split(step_rng, num_envs)
        new_obsv, new_state, rewards, dones, info = step_jit(step_rngs, env_state, dummy_actions)
        new_obsv["player_0"].block_until_ready()
    step_time = (time.time() - st) / 100
    print(f"2. env.step (x{num_envs} envs): {step_time*1000:.3f} ms")
    
    # Profile get_avail_actions
    st = time.time()
    for _ in range(100):
        avail = get_avail_jit(env_state.env_state)
        avail["player_0"].block_until_ready()
    avail_time = (time.time() - st) / 100
    print(f"3. get_avail_actions (x{num_envs} envs): {avail_time*1000:.3f} ms")
    
    # Profile policy forward
    obs_batch = batchify(obsv, env.agents, num_actors)
    done_batch = batchify(done, env.agents, num_actors)
    avail_batch = batchify(avail_actions, env.agents, num_actors).astype(jnp.float32)
    
    @jax.jit
    def policy_forward(params, obs, done, avail, hstate, rng):
        return policy.get_action_value_policy(
            params=params,
            obs=obs.reshape(1, num_actors, -1),
            done=done.reshape(1, num_actors),
            avail_actions=avail.reshape(1, num_actors, -1),
            hstate=hstate,
            rng=rng
        )
    
    # Warmup
    rng, act_rng = jax.random.split(rng)
    action, value, pi, new_hstate = policy_forward(train_state.params, obs_batch, done_batch, avail_batch, hstate, act_rng)
    action.block_until_ready()
    
    st = time.time()
    for _ in range(100):
        rng, act_rng = jax.random.split(rng)
        action, value, pi, new_hstate = policy_forward(train_state.params, obs_batch, done_batch, avail_batch, hstate, act_rng)
        action.block_until_ready()
    policy_time = (time.time() - st) / 100
    print(f"4. policy.forward (x{num_actors} actors): {policy_time*1000:.3f} ms")
    
    # Profile full rollout (scan over env_step)
    def single_env_step(runner_state, unused):
        train_state, env_state, last_obs, last_done, last_hstate, rng = runner_state
        
        rng, act_rng = jax.random.split(rng)
        
        last_obs_batch = batchify(last_obs, env.agents, num_actors)
        last_done_batch = batchify(last_done, env.agents, num_actors)
        
        avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
        avail_actions = jax.lax.stop_gradient(batchify(avail_actions, env.agents, num_actors).astype(jnp.float32))
        
        action, value, pi, new_hstate = policy.get_action_value_policy(
            params=train_state.params,
            obs=last_obs_batch.reshape(1, num_actors, -1),
            done=last_done_batch.reshape(1, num_actors),
            avail_actions=avail_actions.reshape(1, num_actors, -1),
            hstate=last_hstate,
            rng=act_rng
        )
        log_prob = pi.log_prob(action)
        
        action = action.squeeze()
        log_prob = log_prob.squeeze()
        value = value.squeeze()
        
        env_act = unbatchify(action, env.agents, num_envs, env.num_agents)
        env_act = {k: v.flatten() for k, v in env_act.items()}
        
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, num_envs)
        
        new_obs, new_env_state, reward, new_done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
            rng_step, env_state, env_act
        )
        
        info = jax.tree.map(lambda x: x.reshape((num_actors,)), info)
        
        transition = Transition(
            batchify(new_done, env.agents, num_actors).squeeze(),
            action,
            value,
            batchify(reward, env.agents, num_actors).squeeze(),
            log_prob,
            last_obs_batch,
            info,
            avail_actions
        )
        runner_state = (train_state, new_env_state, new_obs, new_done, new_hstate, rng)
        return runner_state, transition
    
    @jax.jit
    def run_rollout(train_state, env_state, obsv, done, hstate, rng):
        runner_state = (train_state, env_state, obsv, done, hstate, rng)
        runner_state, traj_batch = jax.lax.scan(
            single_env_step, runner_state, None, rollout_length
        )
        return runner_state, traj_batch
    
    # Warmup
    rng, rollout_rng = jax.random.split(rng)
    runner_state, traj_batch = run_rollout(train_state, env_state, obsv, done, hstate, rollout_rng)
    traj_batch.done.block_until_ready()
    
    st = time.time()
    for _ in range(10):
        rng, rollout_rng = jax.random.split(rng)
        runner_state, traj_batch = run_rollout(train_state, env_state, obsv, done, hstate, rollout_rng)
        traj_batch.done.block_until_ready()
    rollout_time = (time.time() - st) / 10
    print(f"\n5. ROLLOUT ({rollout_length} steps): {rollout_time*1000:.3f} ms")
    print(f"   - Per step: {rollout_time/rollout_length*1000:.3f} ms")
    
    # Profile PPO update (simplified - just the gradient computation)
    @jax.jit
    def ppo_loss_and_grad(params, obs, actions, old_log_probs, advantages, targets, avail, done, hstate):
        def loss_fn(params):
            _, value, pi, _ = policy.get_action_value_policy(
                params=params,
                obs=obs,
                done=done,
                avail_actions=avail,
                hstate=hstate,
                rng=jax.random.PRNGKey(0)
            )
            log_prob = pi.log_prob(actions)
            
            # Value loss
            value_loss = jnp.square(value - targets).mean()
            
            # Actor loss
            ratio = jnp.exp(log_prob - old_log_probs)
            gae = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            loss_actor1 = ratio * gae
            loss_actor2 = jnp.clip(ratio, 0.9, 1.1) * gae
            actor_loss = -jnp.minimum(loss_actor1, loss_actor2).mean()
            
            entropy = pi.entropy().mean()
            
            return actor_loss + 0.5 * value_loss - 0.01 * entropy
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        return loss, grads
    
    # Prepare minibatch data
    mb_obs = traj_batch.obs.reshape(-1, traj_batch.obs.shape[-1])
    mb_actions = traj_batch.action.reshape(-1)
    mb_log_probs = traj_batch.log_prob.reshape(-1)
    mb_values = traj_batch.value.reshape(-1)
    mb_avail = traj_batch.avail_actions.reshape(-1, traj_batch.avail_actions.shape[-1])
    mb_done = traj_batch.done.reshape(-1)
    
    # Fake advantages/targets
    mb_advantages = jnp.zeros_like(mb_values)
    mb_targets = jnp.zeros_like(mb_values)
    
    # Take a minibatch
    mb_size = num_actors * rollout_length // algorithm_config["NUM_MINIBATCHES"]
    mb_obs_small = mb_obs[:mb_size]
    mb_actions_small = mb_actions[:mb_size]
    mb_log_probs_small = mb_log_probs[:mb_size]
    mb_advantages_small = mb_advantages[:mb_size]
    mb_targets_small = mb_targets[:mb_size]
    mb_avail_small = mb_avail[:mb_size]
    mb_done_small = mb_done[:mb_size]
    
    # Warmup
    loss, grads = ppo_loss_and_grad(
        train_state.params, 
        mb_obs_small.reshape(1, mb_size, -1),
        mb_actions_small.reshape(1, mb_size),
        mb_log_probs_small.reshape(1, mb_size),
        mb_advantages_small.reshape(1, mb_size),
        mb_targets_small.reshape(1, mb_size),
        mb_avail_small.reshape(1, mb_size, -1),
        mb_done_small.reshape(1, mb_size),
        hstate
    )
    loss.block_until_ready()
    
    st = time.time()
    for _ in range(100):
        loss, grads = ppo_loss_and_grad(
            train_state.params,
            mb_obs_small.reshape(1, mb_size, -1),
            mb_actions_small.reshape(1, mb_size),
            mb_log_probs_small.reshape(1, mb_size),
            mb_advantages_small.reshape(1, mb_size),
            mb_targets_small.reshape(1, mb_size),
            mb_avail_small.reshape(1, mb_size, -1),
            mb_done_small.reshape(1, mb_size),
            hstate
        )
        loss.block_until_ready()
    grad_time = (time.time() - st) / 100
    
    num_grad_updates = algorithm_config["UPDATE_EPOCHS"] * algorithm_config["NUM_MINIBATCHES"]
    print(f"\n6. PPO gradient (1 minibatch): {grad_time*1000:.3f} ms")
    print(f"   - Total updates per iter ({num_grad_updates}x): {grad_time*num_grad_updates*1000:.3f} ms")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (per training iteration)")
    print("=" * 60)
    total_per_iter = rollout_time + (grad_time * num_grad_updates)
    print(f"Rollout:     {rollout_time*1000:.3f} ms ({rollout_time/total_per_iter*100:.1f}%)")
    print(f"PPO Update:  {grad_time*num_grad_updates*1000:.3f} ms ({grad_time*num_grad_updates/total_per_iter*100:.1f}%)")
    print(f"Total:       {total_per_iter*1000:.3f} ms")
    print("=" * 60)


@hydra.main(version_base=None, config_path="configs", config_name="base_config_marl")
def main(config: DictConfig):
    profile_components(config)


if __name__ == "__main__":
    main()
