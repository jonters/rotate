"""
Based on PureJaxRL Implementation of PPO
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
import functools
import time
from tqdm import tqdm

from agents.initialize_agents import initialize_s5_agent, initialize_mlp_agent, \
    initialize_rnn_agent, initialize_pseudo_actor_with_double_critic, initialize_pseudo_actor_with_conditional_critic
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.ppo_utils import Transition, batchify, unbatchify, _create_minibatches

from jax.sharding import Mesh, PartitionSpec as PS, NamedSharding

# Number of GPUs to shard across (set to 1 to disable sharding)
NUM_DEVICES = 4 # Set to 1 to disable sharding

# TODO: sharding -- how do we aggregate metrics
# TODO: how to ctrl-c?

def get_sharding():
    """
    Create sharding specs for seed-parallel training.
    Each seed runs entirely independently on one device.
    """
    devices = NUM_DEVICES
    print(f"Number of devices to use: {devices}")

    # Select subset of devices if NUM_DEVICES < total available
    all_devices = jax.devices()
    selected_devices = all_devices[:devices]
    
    device_mesh = Mesh(np.array(selected_devices).reshape((devices,)), axis_names=["data"])
    data_sharding = NamedSharding(device_mesh, PS("data")) # shard seeds across devices

    return data_sharding

def initialize_agent(actor_type, config, env, init_rng):
    if actor_type == "s5":
        policy, init_params = initialize_s5_agent(config, env, init_rng)
    elif actor_type == "mlp":
        policy, init_params = initialize_mlp_agent(config, env, init_rng)
    elif actor_type == "rnn":
        policy, init_params = initialize_rnn_agent(config, env, init_rng)
    elif actor_type == "pseudo_actor_with_double_critic":
        policy, init_params = initialize_pseudo_actor_with_double_critic(config, env, init_rng)
    elif actor_type == "pseudo_actor_with_conditional_critic":
        policy, init_params = initialize_pseudo_actor_with_conditional_critic(config, env, init_rng)
    return policy, init_params

def make_train(config):
    env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["TOTAL_NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"]
    )
    config["NUM_UPDATES"] = config["TOTAL_NUM_UPDATES"] // config.get("TRAIN_CHUNKS", 1)
    config["MINIBATCH_SIZE"] = (
            config["NUM_ACTORS"] * config["ROLLOUT_LENGTH"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env)

    # Accumulator for multi-seed io_callback logging.
    # Under vmap (ordered=True), the callback fires once per seed sequentially.
    # We buffer each seed's value and log the mean once all NUM_SEEDS have reported.
    _metric_buffer = {"returns": []}

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["TOTAL_NUM_UPDATES"]
        return config["LR"] * frac

    def init_env(rng):
        """Initialize environment state - kept separate from train for sharding."""
        reset_rng = jax.random.split(rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        return obsv, env_state

    def train(rng, init_obsv, init_env_state, initial=True, update_runner_state=None):

        # INIT AGENT
        rng, init_rng = jax.random.split(rng)
        policy, init_params = initialize_agent(config["ACTOR_TYPE"], config, env, init_rng)

        # Toggle between precomputed reset buffer vs on-the-fly reset
        USE_RESET_BUFFER = False  # Set to False for original behavior (reset computed each step)
        NUM_RESETS = config.get("NUM_RESETS", 20)  # buffer size (only used if USE_RESET_BUFFER=True)

        if initial:
            if config["ANNEAL_LR"]:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=linear_schedule, eps=1e-5),
                )
            else:
                tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
            train_state = TrainState.create(
                apply_fn=policy.network.apply,
                params=init_params,
                tx=tx,
            )

            # Use pre-initialized env state
            obsv, env_state = init_obsv, init_env_state

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state
            
            train_state, env_state_inner, last_obs, last_done, last_hstate, reset_idx, rng = runner_state
            
            if USE_RESET_BUFFER:
                # Generate fresh reset buffer for this update step
                rng, reset_buffer_rng = jax.random.split(rng)
                reset_rngs = jax.random.split(reset_buffer_rng, config["NUM_ENVS"] * NUM_RESETS)
                reset_rngs = reset_rngs.reshape(config["NUM_ENVS"], NUM_RESETS, -1)
                reset_buffer = jax.vmap(jax.vmap(env.reset, in_axes=(0,)), in_axes=(0,))(reset_rngs)
                # Also reset the indices to 0 for the fresh buffer
                reset_idx = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.int32)
            else:
                reset_buffer = None
                
            runner_state = (train_state, env_state_inner, last_obs, last_done, last_hstate, reset_idx, rng)

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, last_hstate, reset_idx, rng = runner_state

                rng, act_rng = jax.random.split(rng)

                last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                last_done_batch = batchify(last_done, env.agents, config["NUM_ACTORS"])

                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(batchify(avail_actions, 
                    env.agents, config["NUM_ACTORS"]).astype(jnp.float32))

                action, value, pi, new_hstate = policy.get_action_value_policy(
                    params=train_state.params,
                    obs=last_obs_batch.reshape(1, config["NUM_ACTORS"], -1),
                    done=last_done_batch.reshape(1, config["NUM_ACTORS"]),
                    avail_actions=avail_actions.reshape(1, config["NUM_ACTORS"], -1),
                    hstate=last_hstate,
                    rng=act_rng
                )
                log_prob = pi.log_prob(action)

                action = action.squeeze()
                log_prob = log_prob.squeeze()
                value = value.squeeze()

                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = {k:v.flatten() for k,v in env_act.items()}

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                if USE_RESET_BUFFER:
                    # Use precomputed reset buffer
                    new_obs, new_env_state, reward, new_done, info, new_reset_idx = jax.vmap(
                        env.step, in_axes=(0, 0, 0, 0, 0, None)
                    )(rng_step, env_state, env_act, reset_buffer, reset_idx, NUM_RESETS)
                else:
                    # Original behavior: pass None, reset computed on-the-fly
                    new_obs, new_env_state, reward, new_done, info, new_reset_idx = jax.vmap(
                        env.step, in_axes=(0, 0, 0, None, None, None)
                    )(rng_step, env_state, env_act, None, None, None)
                
                # note that num_actors = num_envs * num_agents
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)

                transition = Transition(
                    batchify(new_done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    last_obs_batch,
                    info,
                    avail_actions
                )
                runner_state = (train_state, new_env_state, new_obs, new_done, new_hstate, new_reset_idx, rng)
                return runner_state, transition
            

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["ROLLOUT_LENGTH"]
            )

            # Get final value estimate for completed trajectory
            train_state, env_state, last_obs, last_done, last_hstate, reset_idx, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            last_obs_batch = last_obs_batch.reshape(1, config["NUM_ACTORS"], -1)
            last_done_batch = batchify(last_done, env.agents, config["NUM_ACTORS"])
            last_done_batch = last_done_batch.reshape(1, config["NUM_ACTORS"])
            last_avail_batch = jax.vmap(env.get_avail_actions)(env_state.env_state)
            last_avail_batch = jax.lax.stop_gradient(batchify(last_avail_batch, 
                env.agents, config["NUM_ACTORS"]).astype(jnp.float32))
            
            _, last_val, _, _ = policy.get_action_value_policy(
                params=train_state.params,
                obs=last_obs_batch,
                done=last_done_batch,
                avail_actions=last_avail_batch,
                hstate=last_hstate,
                rng=jax.random.PRNGKey(0)  # Dummy key since we're just extracting the value
            )
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info
                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, value, pi, _ = policy.get_action_value_policy(
                            params=params,
                            obs=traj_batch.obs,
                            done=traj_batch.done,
                            avail_actions=traj_batch.avail_actions,
                            hstate=init_hstate,
                            rng=jax.random.PRNGKey(0) # only used for action sampling, which is unused here
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                minibatches = _create_minibatches(traj_batch, advantages, targets, init_hstate, 
                                                  config["NUM_ACTORS"], config["NUM_MINIBATCHES"], perm_rng)

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
                return update_state, total_loss
            init_hstate = policy.init_hstate(config["NUM_ACTORS"])
            update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            def callback(metric):
                wandb.log(
                    {
                        "returns": metric["returned_episode_returns"][-1, :].mean(),
                        "env_step": metric["update_steps"]
                        * config["NUM_ENVS"]
                        * config["ROLLOUT_LENGTH"],
                    }
                )
                # ret = float(metric["returned_episode_returns"][-1, :].mean())
                # _metric_buffer["returns"].append(ret)
                # if len(_metric_buffer["returns"]) == config["NUM_SEEDS"]:
                #     mean_return = np.mean(_metric_buffer["returns"])
                #     env_step = int(metric["update_steps"]) * config["NUM_ENVS"] * config["ROLLOUT_LENGTH"]
                #     wandb.log({"returns": mean_return, "env_step": env_step})
                #     _metric_buffer["returns"] = []

            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metric)
            update_steps += 1
            runner_state = (train_state, env_state, last_obs, last_done, last_hstate, reset_idx, rng)
            return (runner_state, update_steps), None

        if initial:
            rng, _rng = jax.random.split(rng)
            init_hstate = policy.init_hstate(config["NUM_ACTORS"])
            init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
            init_reset_idx = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.int32)
            runner_state = (train_state, env_state, obsv, init_done, init_hstate, init_reset_idx, _rng)
            update_runner_state = (runner_state, 0)

        update_runner_state, _ = jax.lax.scan(
            _update_step, 
            update_runner_state, 
            xs=None,
            length=config["NUM_UPDATES"],
        )
        return {"update_runner_state": update_runner_state}

    return init_env, train

def main():
    config = {
        "LR": 0.0005,
        "NUM_ENVS": 1024,
        "ROLLOUT_LENGTH": 128,
        "NUM_SEEDS": 4,
        "TOTAL_TIMESTEPS": int(2e9),
        "TRAIN_CHUNKS": 500,  # Set > 1 to chunk training (reduces JIT memory, enables progress tracking)
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ENV_NAME": "hanabi",
        "ENV_KWARGS": {},
        "ANNEAL_LR": False,
        "ACTOR_TYPE": "mlp", # mlp
        "ACTIVATION": "relu",
        "FC_HIDDEN_DIM": 512,
        # S5-specific hyperparameters (used when ACTOR_TYPE is "s5")
        "S5_D_MODEL": 128,
        "S5_SSM_SIZE": 128,
        "S5_N_LAYERS": 2,
        "S5_BLOCKS": 1,
        "S5_ACTOR_CRITIC_HIDDEN_DIM": 1024,
        "FC_N_LAYERS": 3,
        "S5_ACTIVATION": "full_glu",
        "S5_DO_NORM": True,
        "S5_PRENORM": True,
        "S5_DO_GTRXL_NORM": True,
        "WANDB_MODE": "online", # options: online, offline, disabled
        "ENTITY": "",
        "PROJECT": "",
    }

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "FF", config["ENV_NAME"]],
        config=config,
        mode=config["WANDB_MODE"],
    )

    data_sharding = get_sharding()

    assert config["NUM_SEEDS"] % NUM_DEVICES == 0, "NUM_SEEDS must be divisible by NUM_DEVICES for sharding"

    rng = jax.random.PRNGKey(67)
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    init_env_fn, train_fn = make_train(config)

    # Split each seed's rng into an init_rng and a train_rng
    init_rngs, train_rngs = jax.vmap(lambda r: jax.random.split(r))(rngs).transpose((1, 0, 2))

    # JIT-compile with vmap over seeds, sharded across devices
    init_env_jit = jax.jit(jax.vmap(init_env_fn),
                           in_shardings=data_sharding,
                           out_shardings=data_sharding)
    train_jit = jax.jit(jax.vmap(train_fn),
                        in_shardings=(data_sharding, data_sharding, data_sharding),
                        out_shardings=data_sharding)

    def train_continue(rng, init_obsv, init_env_state, update_runner_state):
        return train_fn(rng, init_obsv, init_env_state, initial=False, update_runner_state=update_runner_state)

    train_continue_jit = jax.jit(jax.vmap(train_continue),
                                 in_shardings=(data_sharding, data_sharding, data_sharding, data_sharding),
                                 out_shardings=data_sharding)

    # Place seed rngs on devices
    init_rngs = jax.device_put(init_rngs, data_sharding)
    train_rngs = jax.device_put(train_rngs, data_sharding)

    print("starting training...")
    import time
    from tqdm import tqdm
    st = time.time()

    train_chunks = config.get("TRAIN_CHUNKS", 1)
    init_obsv, init_env_state = init_env_jit(init_rngs)

    pbar = tqdm(range(train_chunks), desc="Training Progress")
    for iteration in pbar:
        if iteration == 0:
            out = train_jit(train_rngs, init_obsv, init_env_state)
        else:
            update_runner_state = out["update_runner_state"]
            out = train_continue_jit(train_rngs, init_obsv, init_env_state, update_runner_state)
        jax.tree.map(lambda x: x.block_until_ready(), out)

    print("training time:", time.time() - st)
    # results = out["returned_episode_returns"].mean(-1).reshape(-1)
    # jnp.save('hanabi_results', results)
    # plt.plot(results)
    # plt.xlabel("Update Step")
    # plt.ylabel("Return")
    # plt.savefig(f'IPPO_{config["ENV_NAME"]}.png')


if __name__ == "__main__":
    main()

    #CUDA_VISIBLE_DEVICES=2 python marl/run.py algorithm=ippo/hanabi task=hanabi