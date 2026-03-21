"""
Wrapper that augments LBF observations with belief model predictions.

Takes a partially-observable LBF env (e.g. fov=3) and concatenates each
agent's partial observation with the belief model's decoded prediction of
the full unmasked state, making the observation approximately Markov.

Observation layout: [partial_obs (15), belief_sample (15)] = 30 dims.
"""

import os
from typing import Any, Dict
from functools import partial

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from flax.serialization import from_bytes

from open_ended_training.belief_model import BeliefModel


@dataclass
class BeliefWrappedState:
    """Env state augmented with per-agent LSTM carries."""
    inner_state: Any          # WrappedEnvState from the base env
    belief_carries: Any       # dict-like: per-agent LSTM carries
    rng: jnp.ndarray          # RNG for belief sampling


class BeliefObsWrapper:
    """Wraps a JumanjiToJaxMARL LBF env to append belief predictions to obs."""

    def __init__(self, env, belief_params, belief_model=None):
        """
        Args:
            env: JumanjiToJaxMARL environment (e.g. lbf-fov-3)
            belief_params: frozen params dict for the BeliefModel
            belief_model: BeliefModel instance (uses default if None)
        """
        self.env = env
        self.num_agents = env.num_agents
        self.agents = env.agents
        self.name = env.name
        self.belief_params = belief_params

        if belief_model is None:
            belief_model = BeliefModel()
        self.belief_model = belief_model

        # Precompute the jitted apply functions
        self._step_encode = jax.jit(
            partial(belief_model.apply, belief_params, method=belief_model.step_encode)
        )
        self._decode_belief = jax.jit(
            partial(belief_model.apply, belief_params, method=belief_model.decode_belief)
        )

        # Update observation spaces to reflect concatenated obs
        from jaxmarl.environments import spaces as jaxmarl_spaces
        self.observation_spaces = {}
        for agent in self.agents:
            orig_space = env.observation_space(agent)
            new_shape = (orig_space.shape[0] + self.belief_model.num_components,)
            self.observation_spaces[agent] = jaxmarl_spaces.Box(
                low=-jnp.inf * jnp.ones(new_shape, dtype=jnp.float32),
                high=jnp.inf * jnp.ones(new_shape, dtype=jnp.float32),
                shape=new_shape,
                dtype=jnp.float32,
            )
        self.action_spaces = env.action_spaces

    def _init_belief_carries(self):
        """Create zero-initialized LSTM carries for all agents."""
        carry = self.belief_model.init_carry(batch_size=1)
        return {agent: carry for agent in self.agents}

    def _augment_obs(self, obs, inner_state, belief_carries, rng):
        """Run belief encoder step + decode for each agent, concat with obs."""
        new_obs = {}
        new_carries = {}
        for i, agent in enumerate(self.agents):
            agent_obs = obs[agent]
            carry = belief_carries[agent]

            # Encoder step: (1, obs_dim) input
            obs_batch = agent_obs[None, :]  # (1, obs_dim)
            context, new_carry = self._step_encode(obs_batch, carry)

            # Decode belief sample
            rng, sample_rng = jax.random.split(rng)
            belief_sample = self._decode_belief(context, sample_rng)  # (1, 15)
            belief_sample = belief_sample[0]  # (15,)

            new_obs[agent] = jnp.concatenate([agent_obs.astype(jnp.float32),
                                               belief_sample.astype(jnp.float32)])
            new_carries[agent] = new_carry

        return new_obs, new_carries, rng

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        key, belief_rng = jax.random.split(key)
        obs, inner_state = self.env.reset(key)
        belief_carries = self._init_belief_carries()

        obs, belief_carries, belief_rng = self._augment_obs(
            obs, inner_state, belief_carries, belief_rng
        )

        state = BeliefWrappedState(
            inner_state=inner_state,
            belief_carries=belief_carries,
            rng=belief_rng,
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state: BeliefWrappedState, actions, params=None):
        obs, inner_state, reward, done, info = self.env.step(
            key, state.inner_state, actions
        )

        # On episode reset, re-initialize belief carries
        reset_carries = self._init_belief_carries()
        belief_carries = jax.tree.map(
            lambda reset, prev: jax.lax.select(done["__all__"], reset, prev),
            reset_carries,
            state.belief_carries,
        )

        obs, belief_carries, belief_rng = self._augment_obs(
            obs, inner_state, belief_carries, state.rng
        )

        new_state = BeliefWrappedState(
            inner_state=inner_state,
            belief_carries=belief_carries,
            rng=belief_rng,
        )
        return obs, new_state, reward, done, info

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.env.action_space(agent)

    def get_avail_actions(self, state: BeliefWrappedState):
        return self.env.get_avail_actions(state.inner_state)

    def render(self, state: BeliefWrappedState):
        self.env.render(state.inner_state)


def load_belief_params(checkpoint_path=None):
    """Load belief model params from checkpoint.

    Args:
        checkpoint_path: path to .msgpack file. Defaults to
            open_ended_training/belief_model_checkpoint.msgpack
    Returns:
        params: frozen params dict
    """
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            os.path.dirname(__file__), "..", "..",
            "open_ended_training", "belief_model_checkpoint.msgpack"
        )
    model = BeliefModel()
    # Initialize to get param structure
    dummy_obs = jnp.zeros((1, 1, 15), dtype=jnp.int32)
    dummy_tgt = jnp.zeros((1, 1, 15), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), dummy_obs, dummy_tgt)

    with open(checkpoint_path, "rb") as f:
        params = from_bytes(params, f.read())
    return params
