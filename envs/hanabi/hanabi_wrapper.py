from functools import partial
from typing import Dict, Tuple, Optional

import chex
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from jaxmarl.environments.hanabi.hanabi import HanabiEnv
from jaxmarl.environments.hanabi.hanabi import State as HanabiState
from jaxmarl.environments import spaces

from envs.base_env import BaseEnv
from envs.base_env import WrappedEnvState

class HanabiWrapper(BaseEnv):
    '''Wrapper for the Hanabi environment to ensure that it follows a common interface 
    with other environments provided in this library.

    Main features:
    - Randomized agent order
    - Flattened observations
    - Base return tracking
    '''
    def __init__(self, *args, **kwargs):       
        self.env = HanabiEnv(*args, **kwargs)
        self.agents = self.env.agents
        self.num_agents = len(self.agents)

        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.agents}
        self.action_spaces = {agent: self.action_space(agent) for agent in self.agents}

    def observation_space(self, agent: str):
        """Returns the observation space."""
        obs_space = self.env.observation_space(agent)
        obs_space.shape = (obs_space.n,)
        return obs_space

    def action_space(self, agent: str):
        """Returns the action space."""
        act_space = self.env.action_space(agent)
        act_space.shape = (act_space.n,)
        return act_space

    def reset(self, key: chex.PRNGKey, ) -> Tuple[Dict[str, chex.Array], WrappedEnvState]:
        obs, env_state = self.env.reset(key)
        # compute avail_actions from the raw env_state
        avail_actions = self.env.get_legal_moves(env_state)
        step = env_state.turn
        return obs, WrappedEnvState(env_state=env_state,
                                     base_return_so_far=jnp.zeros(self.num_agents),
                                     avail_actions=avail_actions,
                                     step=step)

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: WrappedEnvState) -> Dict[str, jnp.ndarray]:
        """Returns the available actions for each agent."""
        return self.env.get_legal_moves(state.env_state)
    
    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: WrappedEnvState) -> jnp.array:
        """Returns the step count for the environment."""
        return state.env_state.turn

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: WrappedEnvState,
        actions: Dict[str, chex.Array],
        reset_state: Optional[Tuple[Dict[str, chex.Array], WrappedEnvState]] = None,
        reset_idx: Optional[int] = None,
        reset_states_length: Optional[int] = None,
    ) -> Tuple[Dict[str, chex.Array], WrappedEnvState, Dict[str, float], Dict[str, bool], Dict]:
        '''Wrapped step function with auto-reset handling.
        
        Args:
            key: Random key
            state: Current wrapped state
            actions: Agent actions
            reset_state: Optional precomputed (obs, state) tuple from self.reset().
                        If provided, uses this for auto-reset instead of calling reset().
                        This avoids redundant reset computation on GPU where jax.lax.cond
                        executes both branches when vmapped.
        
        The base return is tracked in the info dictionary, so that the return can be 
        obtained from the final info.
        '''
        key, key_reset = jax.random.split(key)
        
        # Run the actual environment step (step_env doesn't auto-reset)
        obs_st, env_state_st, rewards, dones, infos = self.env.step_env(key, state.env_state, actions)
        
        # Auto-reset based on done flag
        done_all = dones['__all__']

        # Get reset state - either precomputed or computed now
        if reset_state is not None:
            obs_re = jax.tree.map(lambda x: x[reset_idx], reset_state[0])
            wrapped_state_re = jax.tree.map(lambda x: x[reset_idx], reset_state[1])
            env_state_re = wrapped_state_re.env_state

            new_reset_idx = jax.lax.select(done_all, (reset_idx + 1) % reset_states_length, reset_idx)
        else:
            obs_re, env_state_re = self.env.reset(key_reset)
            new_reset_idx = jnp.array(0)  # Dummy value, won't be used
        
        env_state = jax.tree.map(
            lambda x, y: jax.lax.select(done_all, x, y), env_state_re, env_state_st
        )
        obs = jax.tree.map(
            lambda x, y: jax.lax.select(done_all, x, y), obs_re, obs_st
        )
        
        # Extract base reward (assuming rewards are the base rewards for Hanabi)
        # Convert rewards dict to array for tracking
        base_reward = jnp.array([rewards[agent] for agent in self.agents])
        base_return_so_far = base_reward + state.base_return_so_far
        new_info = {**infos, 'base_return': base_return_so_far, 'base_reward': base_reward}
        
        # handle auto-resetting the base return upon episode termination
        base_return_so_far = jax.lax.select(done_all, jnp.zeros(self.num_agents), base_return_so_far)
        
        # compute new avail_actions and step (use post-reset state)
        avail_actions = self.env.get_legal_moves(env_state)
        step = env_state.turn
        new_state = WrappedEnvState(env_state=env_state,
                                    base_return_so_far=base_return_so_far,
                                    avail_actions=avail_actions,
                                    step=step)
        return obs, new_state, rewards, dones, new_info, new_reset_idx
