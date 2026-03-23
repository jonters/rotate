'''Implementation of the ROTATE algorithm (Wang et al. 2025)
https://arxiv.org/abs/2505.23686

Command to run ROTATE only on LBF: 
python open_ended_training/run.py algorithm=rotate/lbf task=lbf label=test_rotate

Suggested debug command: 
python open_ended_training/run.py algorithm=rotate/lbf task=lbf label=test_rotate logger.mode=offline algorithm.NUM_OPEN_ENDED_ITERS=1 algorithm.TIMESTEPS_PER_ITER_PARTNER=1e5 algorithm.TIMESTEPS_PER_ITER_EGO=1e5
'''
import copy
from functools import partial
import logging
import time

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Timing utilities using jax.debug.callback
# ---------------------------------------------------------------------------
_timing_state = {}


def _make_timer_start(phase_name):
    """Return a callback that records the start time for a phase."""
    def fn(iter_idx):
        _timing_state[(phase_name, int(iter_idx))] = time.time()
    return fn


def _make_timer_end(phase_name, num_updates, num_timesteps):
    """Return a callback that logs elapsed time, updates/sec, and timesteps/sec."""
    def fn(iter_idx, _dep):
        key = (phase_name, int(iter_idx))
        start = _timing_state.pop(key, None)
        if start is None:
            return
        elapsed = time.time() - start
        ups = num_updates / elapsed if elapsed > 0 else float('inf')
        tps = num_timesteps / elapsed if elapsed > 0 else float('inf')
        log.info(
            f"[OEL iter {int(iter_idx)}] {phase_name}: "
            f"{num_updates} updates in {elapsed:.2f}s ({ups:.1f} updates/sec) | "
            f"{num_timesteps} timesteps ({tps:.1f} timesteps/sec)"
        )
    return fn

from agents.agent_interface import ActorWithDoubleCriticPolicy, MLPActorCriticPolicy, S5ActorCriticPolicy, S5ActorWithDoubleCriticPolicy
from agents.population_buffer import BufferedPopulation, add_partners_to_buffer, get_final_buffer
from agents.initialize_agents import initialize_s5_agent, initialize_actor_with_double_critic
from common.plot_utils import get_metric_names
from envs import make_env
from envs.log_wrapper import LogWrapper
from open_ended_training.ppo_ego_with_buffer import train_ppo_ego_agent_with_buffer
from open_ended_training.rotate_without_pop import train_regret_maximizing_partners, log_metrics as log_metrics_without_pop
from open_ended_training.rotate_with_mixed_play import train_regret_maximizing_partners as train_partners_with_mp, log_metrics as log_mp_metrics
from marl.ippo import make_train as make_ppo_train

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_ippo_partners(config, partner_rng, env):
    '''
    Train a pool IPPO agents w/parameter sharing. 
    Returns out, a dictionary of the model checkpoints, final parameters, and metrics.
    '''
    pretrain_config = config["PRETRAIN_ARGS"]
    config["TOTAL_TIMESTEPS"] = pretrain_config["TOTAL_TIMESTEPS"] // pretrain_config["NUM_AGENTS"]
    config["NUM_CHECKPOINTS"] = pretrain_config["NUM_CHECKPOINTS"]
    config["ACTOR_TYPE"] = "pseudo_actor_with_double_critic"

    rngs = jax.random.split(partner_rng, pretrain_config["NUM_AGENTS"])
    train_jit = jax.jit(jax.vmap(make_ppo_train(config, env)))
    out = train_jit(rngs)
    return out

# def linear_schedule_regret(iter_idx, config):
#     '''Computes the upper and lower regret thresholds based on the iteration index. 
#     Updates the config with the next regret thresholds.'''
#     frac = iter_idx / config["NUM_OPEN_ENDED_ITERS"]
#     config["LOWER_REGRET_THRESHOLD"] = config["LOWER_REGRET_THRESHOLD_START"] + (config["LOWER_REGRET_THRESHOLD_END"] - config["LOWER_REGRET_THRESHOLD_START"]) * frac
#     config["UPPER_REGRET_THRESHOLD"] = config["UPPER_REGRET_THRESHOLD_START"] + (config["UPPER_REGRET_THRESHOLD_END"] - config["UPPER_REGRET_THRESHOLD_START"]) * frac
#     return config

def persistent_open_ended_training_step(carry, ego_policy, conf_policy, br_policy, 
                                        partner_population, config, ego_config, env):
    '''
    Train the ego agent against a growing population of regret-maximizing partners.
    Unlike the original implementation, the partner population persists across iterations.
    '''
    prev_ego_params, prev_conf_params, prev_br_params, population_buffer, rng, oel_iter_idx = carry
    rng, partner_rng, ego_rng, conf_init_rng, br_init_rng = jax.random.split(rng, 5)

    # -- Timing: full OEL iteration --
    total_num_updates = (
        int(config["TIMESTEPS_PER_ITER_PARTNER"]) // (
            int(config["ROLLOUT_LENGTH"]) * 4 * int(config["NUM_ENVS"]) * int(config["PARTNER_POP_SIZE"])
        )
        + int(ego_config["TOTAL_TIMESTEPS"]) // (
            int(ego_config["ROLLOUT_LENGTH"]) * int(ego_config["NUM_ENVS"])
        )
    )
    total_num_timesteps = int(config["TIMESTEPS_PER_ITER_PARTNER"]) + int(ego_config["TOTAL_TIMESTEPS"])
    jax.debug.callback(_make_timer_start("oel_iteration"), oel_iter_idx, ordered=True)

    # Initialize or reuse confederate parameters based on config
    if config["REINIT_CONF"]:
        init_rngs = jax.random.split(conf_init_rng, config["PARTNER_POP_SIZE"])
        conf_params = jax.vmap(conf_policy.init_params)(init_rngs)
    else:
        conf_params = prev_conf_params

    # Initialize or reuse best response parameters based on config
    if config["REINIT_BR_TO_BR"]:
        init_rngs = jax.random.split(br_init_rng, config["PARTNER_POP_SIZE"])
        br_params = jax.vmap(br_policy.init_params)(init_rngs)
    elif config["REINIT_BR_TO_EGO"]:
        br_params = jax.tree.map(lambda x: x[jnp.newaxis, ...].repeat(config["PARTNER_POP_SIZE"], axis=0), prev_ego_params)
    else:
        br_params = prev_br_params
    
    # Train partner agents with ego_policy
    if config["PARTNER_ALGO"] == "rotate_without_pop":
        train_partners_fn = train_regret_maximizing_partners
    elif config["PARTNER_ALGO"] == "rotate_with_mixed_play":
        train_partners_fn = train_partners_with_mp
    else:
        raise ValueError(f"Invalid PARTNER_ALGO value: {config['PARTNER_ALGO']}")

    # -- Timing: partner training --
    partner_num_updates = int(config["TIMESTEPS_PER_ITER_PARTNER"]) // (
        int(config["ROLLOUT_LENGTH"]) * 4 * int(config["NUM_ENVS"]) * int(config["PARTNER_POP_SIZE"])
    )
    partner_num_timesteps = int(config["TIMESTEPS_PER_ITER_PARTNER"])
    jax.debug.callback(_make_timer_start("partner_training"), oel_iter_idx, ordered=True)

    train_out = train_partners_fn(config, env,
                                  ego_params=prev_ego_params, ego_policy=ego_policy,
                                  conf_params=conf_params, conf_policy=conf_policy,
                                  br_params=br_params, br_policy=br_policy,
                                  partner_rng=partner_rng)

    # End partner timer (pass a leaf of train_out to enforce data dependency)
    jax.debug.callback(
        _make_timer_end("partner_training", partner_num_updates, partner_num_timesteps),
        oel_iter_idx, jax.tree.leaves(train_out)[0], ordered=True,
    )
        
    if config["EGO_TEAMMATE"] == "final":
        all_conf_params = train_out["final_params_conf"]
        
    elif config["EGO_TEAMMATE"] == "all":
        n_ckpts = config["PARTNER_POP_SIZE"] * config["NUM_CHECKPOINTS"]
        conf_ckpt_params= jax.tree.map(
            lambda x: x.reshape((n_ckpts,) + x.shape[2:]), 
            train_out["checkpoints_conf"]
        )
        all_conf_params = jax.tree.map(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            conf_ckpt_params,
            train_out["final_params_conf"]
        )
    
    # Add all checkpoints and final parameters of all partners to the buffer
    updated_buffer = add_partners_to_buffer(partner_population, population_buffer, all_conf_params)

    # -- Timing: ego training --
    ego_num_updates = int(ego_config["TOTAL_TIMESTEPS"]) // (
        int(ego_config["ROLLOUT_LENGTH"]) * int(ego_config["NUM_ENVS"])
    )
    ego_num_timesteps = int(ego_config["TOTAL_TIMESTEPS"])
    jax.debug.callback(_make_timer_start("ego_training"), oel_iter_idx, ordered=True)

    # Train ego agent using the population buffer
    ego_out = train_ppo_ego_agent_with_buffer(
        config=ego_config,
        env=env,
        train_rng=ego_rng,
        ego_policy=ego_policy,
        init_ego_params=prev_ego_params,
        n_ego_train_seeds=1,
        partner_population=partner_population,
        population_buffer=updated_buffer  # Pass the buffer to the training function
    )

    # End ego timer (pass a leaf of ego_out to enforce data dependency)
    jax.debug.callback(
        _make_timer_end("ego_training", ego_num_updates, ego_num_timesteps),
        oel_iter_idx, jax.tree.leaves(ego_out)[0], ordered=True,
    )

    updated_ego_parameters = ego_out["final_params"]
    updated_conf_parameters = train_out["final_params_conf"]
    updated_br_parameters = train_out["final_params_br"]

    # Remove initial dimension of 1, to ensure that input and output carry have the same dimension
    updated_ego_parameters = jax.tree_map(lambda x: x.squeeze(axis=0), updated_ego_parameters)

    carry = (updated_ego_parameters, updated_conf_parameters, updated_br_parameters,
             updated_buffer, rng, oel_iter_idx + 1)

    # End full OEL iteration timer
    jax.debug.callback(
        _make_timer_end("oel_iteration", total_num_updates, total_num_timesteps),
        oel_iter_idx, jax.tree.leaves(ego_out)[0], ordered=True,
    )

    return carry, (train_out, ego_out)


def train_persistent(rng, env, algorithm_config, ego_config):
    rng, init_ego_rng, init_conf_rng1, init_conf_rng2, init_br_rng, train_rng = jax.random.split(rng, 6)

    use_mlp = algorithm_config.get("USE_MLP", False)
    action_dim = env.action_space(env.agents[0]).n
    obs_dim = env.observation_space(env.agents[0]).shape[0]
    fc_hidden_dim = ego_config.get("S5_ACTOR_CRITIC_HIDDEN_DIM", 1024)

    if use_mlp:
        # MLP policies — hidden state is ignored (returns None)
        ego_policy = MLPActorCriticPolicy(
            action_dim=action_dim, obs_dim=obs_dim, fc_hidden_dim=fc_hidden_dim,
        )
        init_ego_params = ego_policy.init_params(init_ego_rng)

        conf_policy = ActorWithDoubleCriticPolicy(
            action_dim=action_dim, obs_dim=obs_dim,
        )

        br_policy = MLPActorCriticPolicy(
            action_dim=action_dim, obs_dim=obs_dim, fc_hidden_dim=fc_hidden_dim,
        )
    else:
        # S5 policies (original behavior)
        ego_policy, init_ego_params = initialize_s5_agent(ego_config, env, init_ego_rng)

        conf_policy = S5ActorWithDoubleCriticPolicy(
            action_dim=action_dim, obs_dim=obs_dim,
            d_model=ego_config.get("S5_D_MODEL", 128),
            ssm_size=ego_config.get("S5_SSM_SIZE", 128),
            ssm_n_layers=ego_config.get("S5_N_LAYERS", 2),
            blocks=ego_config.get("S5_BLOCKS", 1),
            fc_hidden_dim=fc_hidden_dim,
            fc_n_layers=ego_config.get("FC_N_LAYERS", 3),
            s5_activation=ego_config.get("S5_ACTIVATION", "full_glu"),
            s5_do_norm=ego_config.get("S5_DO_NORM", True),
            s5_prenorm=ego_config.get("S5_PRENORM", True),
            s5_do_gtrxl_norm=ego_config.get("S5_DO_GTRXL_NORM", True),
        )

        br_policy = S5ActorCriticPolicy(
            action_dim=action_dim, obs_dim=obs_dim,
            d_model=ego_config.get("S5_D_MODEL", 128),
            ssm_size=ego_config.get("S5_SSM_SIZE", 128),
            ssm_n_layers=ego_config.get("S5_N_LAYERS", 2),
            blocks=ego_config.get("S5_BLOCKS", 1),
            fc_hidden_dim=fc_hidden_dim,
            fc_n_layers=ego_config.get("FC_N_LAYERS", 3),
            s5_activation=ego_config.get("S5_ACTIVATION", "full_glu"),
            s5_do_norm=ego_config.get("S5_DO_NORM", True),
            s5_prenorm=ego_config.get("S5_PRENORM", True),
            s5_do_gtrxl_norm=ego_config.get("S5_DO_GTRXL_NORM", True),
        )

    init_conf_rngs = jax.random.split(init_conf_rng1, algorithm_config["PARTNER_POP_SIZE"])
    init_conf_params = jax.vmap(conf_policy.init_params)(init_conf_rngs)

    init_br_rngs = jax.random.split(init_br_rng, algorithm_config["PARTNER_POP_SIZE"])
    init_br_params = jax.vmap(br_policy.init_params)(init_br_rngs)
    
    # Create persistent partner population with BufferedPopulation
    # The max_pop_size must be large enough to hold all agents across all iterations

    if algorithm_config["EGO_TEAMMATE"] == "final":
        max_pop_size = algorithm_config["PARTNER_POP_SIZE"] * algorithm_config["NUM_OPEN_ENDED_ITERS"]
    elif algorithm_config["EGO_TEAMMATE"] == "all":
        max_pop_size = (algorithm_config["PARTNER_POP_SIZE"] * algorithm_config["NUM_CHECKPOINTS"] + 1) * \
                       algorithm_config["NUM_OPEN_ENDED_ITERS"]
    else:
        raise ValueError(f"Invalid EGO_TEAMMATE value: {algorithm_config['EGO_TEAMMATE']}")

    if algorithm_config["PRETRAIN_PPO"]:
        max_pop_size += algorithm_config["PRETRAIN_ARGS"]["NUM_AGENTS"] * algorithm_config["PRETRAIN_ARGS"]["NUM_CHECKPOINTS"]

    # Initialize the partner population's conf policy class with the right initializer shape
    if use_mlp:
        conf_policy2 = ActorWithDoubleCriticPolicy(
            action_dim=action_dim, obs_dim=obs_dim,
        )
    else:
        conf_policy2 = S5ActorWithDoubleCriticPolicy(
            action_dim=action_dim, obs_dim=obs_dim,
            d_model=ego_config.get("S5_D_MODEL", 128),
            ssm_size=ego_config.get("S5_SSM_SIZE", 128),
            ssm_n_layers=ego_config.get("S5_N_LAYERS", 2),
            blocks=ego_config.get("S5_BLOCKS", 1),
            fc_hidden_dim=fc_hidden_dim,
            fc_n_layers=ego_config.get("FC_N_LAYERS", 3),
            s5_activation=ego_config.get("S5_ACTIVATION", "full_glu"),
            s5_do_norm=ego_config.get("S5_DO_NORM", True),
            s5_prenorm=ego_config.get("S5_PRENORM", True),
            s5_do_gtrxl_norm=ego_config.get("S5_DO_GTRXL_NORM", True),
        )
    init_conf_params2 = conf_policy2.init_params(init_conf_rng2)
    
    partner_population = BufferedPopulation(
        max_pop_size=max_pop_size,
        policy_cls=conf_policy2,
        sampling_strategy=algorithm_config["SAMPLING_STRATEGY"],
        staleness_coef=algorithm_config["STALENESS_COEF"],
        temp=algorithm_config["SCORE_TEMP"],
    )
    population_buffer = partner_population.reset_buffer(init_conf_params2)
    
    if algorithm_config["PRETRAIN_PPO"]:
        log.info("Pretraining IPPO partners...")
        pretrain_out = train_ippo_partners(algorithm_config, train_rng, env)
        pretrain_checkpoints = pretrain_out["checkpoints"]
        num_ckpts = algorithm_config["PRETRAIN_ARGS"]["NUM_CHECKPOINTS"] * algorithm_config["PRETRAIN_ARGS"]["NUM_AGENTS"]
        pretrain_checkpoints = jax.tree.map(lambda x: x.reshape((num_ckpts,) + x.shape[2:]), 
                                          pretrain_checkpoints)

        log.info("Done pretraining IPPO partners.")
        population_buffer = add_partners_to_buffer(partner_population, population_buffer, pretrain_checkpoints)

    @jax.jit
    def open_ended_step_fn(carry, unused):
        return persistent_open_ended_training_step(carry, ego_policy, conf_policy, br_policy, 
                                                 partner_population, algorithm_config, ego_config, env)
    
    init_carry = (init_ego_params, init_conf_params, init_br_params, population_buffer, train_rng, 0)
    final_carry, outs = jax.lax.scan(
        open_ended_step_fn, 
        init_carry, 
        xs=None,
        length=algorithm_config["NUM_OPEN_ENDED_ITERS"]
    )

    # Save only the buffer from the last iteration of OEL, rather than all iterations
    outs[1]["final_buffer"] = get_final_buffer(outs[1]["final_buffer"])

    if algorithm_config["PRETRAIN_PPO"]:
        # add pretrain out to the teammate out
        outs[0]["pretrain_out"] = pretrain_out

    return outs

def run_rotate(config, wandb_logger):
    algorithm_config = dict(config["algorithm"])

    # Create only one environment instance
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)
    
    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rng, init_ego_rng = jax.random.split(rng)
    rngs = jax.random.split(rng, algorithm_config["NUM_SEEDS"])

    # initialize ego config
    ego_config = copy.deepcopy(algorithm_config)
    ego_config["TOTAL_TIMESTEPS"] = algorithm_config["TIMESTEPS_PER_ITER_EGO"]
    EGO_ARGS = algorithm_config.get("EGO_ARGS", {})
    ego_config.update(EGO_ARGS)

    log.info("Starting ROTATE training...")
    start_time = time.time()

    DEBUG = False
    with jax.disable_jit(DEBUG):
        train_fn = jax.jit(jax.vmap(partial(train_persistent, 
                env=env, algorithm_config=algorithm_config, ego_config=ego_config
                )
            )
        )
        outs = train_fn(rngs)
    
    end_time = time.time()
    log.info(f"ROTATE training completed in {end_time - start_time} seconds.")

    metric_names = get_metric_names(algorithm_config["ENV_NAME"])

    # Log metrics
    if algorithm_config["PARTNER_ALGO"] == "rotate_without_pop":
        log_metrics = log_metrics_without_pop
    elif algorithm_config["PARTNER_ALGO"] == "rotate_with_mixed_play":
        log_metrics = log_mp_metrics
    else:
        raise ValueError(f"Invalid PARTNER_ALGO value: {algorithm_config['PARTNER_ALGO']}")
    log_metrics(config, wandb_logger, outs, metric_names)

    # Prepare return values for heldout evaluation
    _, ego_outs = outs
    ego_params = jax.tree_map(lambda x: x[:, :, 0], ego_outs["final_params"]) # shape (num_seeds, num_open_ended_iters, 1, num_ckpts, leaf_dim)

    use_mlp = algorithm_config.get("USE_MLP", False)
    if use_mlp:
        action_dim = env.action_space(env.agents[0]).n
        obs_dim = env.observation_space(env.agents[0]).shape[0]
        fc_hidden_dim = ego_config.get("S5_ACTOR_CRITIC_HIDDEN_DIM", 1024)
        ego_policy = MLPActorCriticPolicy(
            action_dim=action_dim, obs_dim=obs_dim, fc_hidden_dim=fc_hidden_dim,
        )
        init_ego_params = ego_policy.init_params(init_ego_rng)
    else:
        ego_policy, init_ego_params = initialize_s5_agent(ego_config, env, init_ego_rng)

    return ego_policy, ego_params, init_ego_params