import jax.numpy as jnp


def get_unmasked_obs(obs, state, agent_idx, num_food, num_agents, fov, grid_size):
    """Given a (partially) masked observation for an agent, return the full
    unmasked observation — i.e. with all (-1, -1, 0) entries filled in with
    the true (local-coordinate) values that the agent *would* see if it had
    full field of view, but still expressed in the same local coordinate frame.

    Args:
        obs: flat int32 array of shape (3 * (num_food + num_agents),)
             as produced by the JumanjiToJaxMARL wrapper.
        state: a WrappedEnvState (the full environment state).
        agent_idx: which agent this observation belongs to (0-indexed).
        num_food: number of food items.
        num_agents: number of agents.
        fov: the field-of-view used when the observation was generated.
        grid_size: the grid size.

    Returns:
        full_obs: same shape as obs, but with masked entries replaced by the
                  true local-coordinate values.
    """

    # OBS are
    # food [row, col, level]
    # me   [row, col, level]
    # teammate [row col, level]

    # in fov-X, observations for an agent are relative to an anchor, where 
    # the anchor is the top-left of a X by X grid

    env_state = state.env_state

    agent_pos = env_state.agents.position[agent_idx]  # (2,)

    # Local coordinate offset (same as Jumanji's transform_positions)
    offset = jnp.array([
        jnp.minimum(fov, agent_pos[0]),
        jnp.minimum(fov, agent_pos[1]),
    ]) - agent_pos  # add this to absolute pos to get local pos

    # --- Food entries (first num_food * 3 values) ---
    food_pos_local = env_state.food_items.position + offset  # (num_food, 2)
    food_levels = env_state.food_items.level                 # (num_food,)
    # Mark eaten food as masked
    eaten = env_state.food_items.eaten
    food_rows = jnp.where(eaten, -1, food_pos_local[:, 0])
    food_cols = jnp.where(eaten, -1, food_pos_local[:, 1])
    food_lvls = jnp.where(eaten, 0, food_levels)

    # --- Agent entries (next num_agents * 3 values, self first) ---
    all_pos_local = env_state.agents.position + offset  # (num_agents, 2)
    all_levels = env_state.agents.level                 # (num_agents,)

    # Self info
    self_info = jnp.array([
        all_pos_local[agent_idx, 0],
        all_pos_local[agent_idx, 1],
        all_levels[agent_idx],
    ])

    # Other agents (exclude self, preserve order)
    other_mask = jnp.arange(num_agents) != agent_idx
    other_rows = jnp.where(other_mask, all_pos_local[:, 0], -999)
    other_cols = jnp.where(other_mask, all_pos_local[:, 1], -999)
    other_lvls = jnp.where(other_mask, all_levels, -999)
    # Compact: remove self entry
    other_indices = jnp.where(other_mask, size=num_agents - 1)
    other_rows = other_rows[other_indices]
    other_cols = other_cols[other_indices]
    other_lvls = other_lvls[other_indices]

    # Build full observation vector
    # Food: interleaved (row, col, level) for each food
    food_flat = jnp.stack([food_rows, food_cols, food_lvls], axis=-1).ravel()
    # Self
    # Others: interleaved
    others_flat = jnp.stack([other_rows, other_cols, other_lvls], axis=-1).ravel()

    full_obs = jnp.concatenate([food_flat, self_info, others_flat])
    return full_obs
