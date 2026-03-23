import jax.numpy as jnp


def get_unmasked_obs(state, agent_idx, hand_size, num_colors, num_ranks):
    """Return the agent's own hidden hand as a flat one-hot vector.

    In Hanabi, each agent cannot see their own cards. This function extracts
    the agent's true hand from the full game state, encoded identically to
    how other players' hands appear in the observation's hands section.

    Args:
        state: WrappedEnvState from HanabiWrapper.
        agent_idx: which agent (0-indexed).
        hand_size: cards per hand (default 5).
        num_colors: number of colors (default 5).
        num_ranks: number of ranks (default 5).

    Returns:
        hand_flat: (hand_size * num_colors * num_ranks,) float32 one-hot
                   encoding of the agent's own hand. Each card is a
                   (num_colors * num_ranks,) one-hot block; empty slots are
                   all zeros.
    """
    env_state = state.env_state
    own_hand = env_state.player_hands[agent_idx]  # (hand_size, num_colors, num_ranks)
    return own_hand.ravel()  # (hand_size * num_colors * num_ranks,)


def hand_to_card_indices(hand_flat, hand_size, num_colors, num_ranks):
    """Convert a flat one-hot hand to per-card type indices.

    Useful as categorical training targets for the belief model.

    Args:
        hand_flat: (hand_size * num_colors * num_ranks,) from get_unmasked_obs.
        hand_size: cards per hand.
        num_colors: number of colors.
        num_ranks: number of ranks.

    Returns:
        indices: (hand_size,) int32, each in [0, num_colors*num_ranks-1] for
                 a real card, or num_colors*num_ranks for an empty slot.
    """
    card_dim = num_colors * num_ranks
    hand = hand_flat.reshape(hand_size, card_dim)
    has_card = hand.any(axis=-1)  # (hand_size,)
    card_idx = jnp.argmax(hand, axis=-1)  # (hand_size,)
    # Empty slots -> card_dim (special "no card" token)
    return jnp.where(has_card, card_idx, card_dim)
