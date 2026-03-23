"""
Train a grounded belief model for LBF or Hanabi.

The belief model predicts hidden state from a trajectory of partial
observations, using an auto-regressive encoder-decoder architecture.

Training loop: collect rollouts -> train for Y epochs -> report -> repeat.

Usage:
    python -m open_ended_training.train_grounded_belief
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
from flax.serialization import to_bytes, from_bytes
from functools import partial

from tqdm import tqdm

from envs import make_env
from open_ended_training.belief_model import BeliefModel


# ─── Game selection ──────────────────────────────────────────────────────────

GAME = "hanabi"  # "lbf" or "hanabi"

# ─── Config ───────────────────────────────────────────────────────────────────

NUM_AGENTS = 2
ROLLOUT_LENGTH = 128

# Training
NUM_ITERATIONS = 100
NUM_ROLLOUTS = 1000
NUM_EPOCHS = 10
BATCH_SIZE = 64
LR = 3e-4
GRAD_CLIP = 1.0
ENCODER_HIDDEN = 256
DECODER_HIDDEN = 256
EMBED_DIM = 64
SEED = 67

if GAME == "lbf":
    from envs.lbf import get_unmasked_obs as lbf_get_unmasked_obs

    ENV_NAME = "lbf-fov-3"
    NUM_FOOD = 3
    FOV = 3
    GRID_SIZE = 7
    MAX_AGENT_LEVEL = 2
    OBS_DIM = 15            # 5 objects * 3 values
    NUM_COMPONENTS = 15
    VALUE_OFFSET = 8
    VOCAB_SIZE = 26

elif GAME == "hanabi":
    from envs.hanabi import get_unmasked_obs as hanabi_get_unmasked_obs
    from envs.hanabi import hand_to_card_indices

    ENV_NAME = "hanabi"
    HAND_SIZE = 5
    NUM_COLORS = 5
    NUM_RANKS = 5
    OBS_DIM = 658
    NUM_COMPONENTS = 5      # one card index per hand slot
    VOCAB_SIZE = 26          # 25 card types + 1 empty slot
    VALUE_OFFSET = 0


# ─── Rollout collection ──────────────────────────────────────────────────────

def collect_rollouts_lbf(env, rng, num_rollouts, rollout_length):
    """Collect LBF trajectories with random actions."""
    num_actions = env.action_space("agent_0").n

    def _single_rollout(rng):
        rng, reset_rng = jax.random.split(rng)
        obs, state = env.reset(reset_rng)

        def _step(carry, _):
            obs, state, rng = carry
            rng, act_rng, step_rng = jax.random.split(rng, 3)

            actions = {
                f"agent_{i}": jax.random.randint(
                    jax.random.fold_in(act_rng, i), (), 0, num_actions
                )
                for i in range(NUM_AGENTS)
            }

            obs_0 = obs["agent_0"]
            obs_1 = obs["agent_1"]
            target_0 = lbf_get_unmasked_obs(
                obs_0, state, 0, NUM_FOOD, NUM_AGENTS, FOV, GRID_SIZE
            )
            target_1 = lbf_get_unmasked_obs(
                obs_1, state, 1, NUM_FOOD, NUM_AGENTS, FOV, GRID_SIZE
            )

            next_obs, next_state, reward, done, info = env.step(
                step_rng, state, actions
            )

            return (next_obs, next_state, rng), (obs_0, obs_1, target_0, target_1, done["__all__"])

        _, (obs0_traj, obs1_traj, tgt0_traj, tgt1_traj, done_traj) = jax.lax.scan(
            _step, (obs, state, rng), None, length=rollout_length
        )
        obs_traj = jnp.stack([obs0_traj, obs1_traj], axis=0)
        tgt_traj = jnp.stack([tgt0_traj, tgt1_traj], axis=0)
        done_traj = jnp.stack([done_traj, done_traj], axis=0)
        return obs_traj, tgt_traj, done_traj

    rngs = jax.random.split(rng, num_rollouts)
    all_obs, all_targets, all_dones = jax.vmap(_single_rollout)(rngs)
    all_obs = all_obs.reshape(-1, rollout_length, OBS_DIM)
    all_targets = all_targets.reshape(-1, rollout_length, NUM_COMPONENTS)
    all_dones = all_dones.reshape(-1, rollout_length)

    return all_obs, all_targets, all_dones


def collect_rollouts_hanabi(env, rng, num_rollouts, rollout_length):
    """Collect Hanabi trajectories with random legal actions."""

    def _sample_legal_action(rng, legal_mask):
        logits = jnp.where(legal_mask, 0.0, -1e10)
        return jax.random.categorical(rng, logits)

    def _single_rollout(rng):
        rng, reset_rng = jax.random.split(rng)
        obs, state = env.reset(reset_rng)

        def _step(carry, _):
            obs, state, rng = carry
            rng, act_rng, step_rng = jax.random.split(rng, 3)

            # Sample legal actions
            avail = env.get_avail_actions(state)
            actions = {
                f"agent_{i}": _sample_legal_action(
                    jax.random.fold_in(act_rng, i), avail[f"agent_{i}"]
                )
                for i in range(NUM_AGENTS)
            }

            obs_0 = obs["agent_0"]
            obs_1 = obs["agent_1"]

            # Extract hidden hand as card indices
            hand_0 = hanabi_get_unmasked_obs(state, 0, HAND_SIZE, NUM_COLORS, NUM_RANKS)
            hand_1 = hanabi_get_unmasked_obs(state, 1, HAND_SIZE, NUM_COLORS, NUM_RANKS)
            target_0 = hand_to_card_indices(hand_0, HAND_SIZE, NUM_COLORS, NUM_RANKS)
            target_1 = hand_to_card_indices(hand_1, HAND_SIZE, NUM_COLORS, NUM_RANKS)

            # HanabiWrapper.step returns 6 values (extra reset_idx)
            next_obs, next_state, reward, done, info, _ = env.step(
                step_rng, state, actions
            )

            return (next_obs, next_state, rng), (obs_0, obs_1, target_0, target_1, done["__all__"])

        _, (obs0_traj, obs1_traj, tgt0_traj, tgt1_traj, done_traj) = jax.lax.scan(
            _step, (obs, state, rng), None, length=rollout_length
        )
        obs_traj = jnp.stack([obs0_traj, obs1_traj], axis=0)
        tgt_traj = jnp.stack([tgt0_traj, tgt1_traj], axis=0)
        done_traj = jnp.stack([done_traj, done_traj], axis=0)
        return obs_traj, tgt_traj, done_traj

    rngs = jax.random.split(rng, num_rollouts)
    all_obs, all_targets, all_dones = jax.vmap(_single_rollout)(rngs)
    all_obs = all_obs.reshape(-1, rollout_length, OBS_DIM)
    all_targets = all_targets.reshape(-1, rollout_length, NUM_COMPONENTS)
    all_dones = all_dones.reshape(-1, rollout_length)

    return all_obs, all_targets, all_dones


if GAME == "lbf":
    collect_rollouts = collect_rollouts_lbf
elif GAME == "hanabi":
    collect_rollouts = collect_rollouts_hanabi


# ─── Training utilities ──────────────────────────────────────────────────────

def create_train_state(rng):
    model = BeliefModel(
        encoder_hidden=ENCODER_HIDDEN,
        decoder_hidden=DECODER_HIDDEN,
        obs_dim=OBS_DIM,
        num_components=NUM_COMPONENTS,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        value_offset=VALUE_OFFSET,
    )
    dummy_obs = jnp.zeros((1, ROLLOUT_LENGTH, OBS_DIM), dtype=jnp.int32)
    dummy_tgt = jnp.zeros((1, ROLLOUT_LENGTH, NUM_COMPONENTS), dtype=jnp.int32)
    params = model.init(rng, dummy_obs, dummy_tgt)

    tx = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP),
        optax.adam(LR),
    )
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )


@jax.jit
def train_step(state, obs_batch, target_batch):
    """Single gradient step."""
    def loss_fn(params):
        loss, acc = state.apply_fn(params, obs_batch, target_batch)
        return loss, acc

    (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, acc


def train_epoch(state, obs_data, target_data, rng):
    """One epoch over the dataset."""
    n = obs_data.shape[0]
    rng, perm_rng = jax.random.split(rng)
    perm = jax.random.permutation(perm_rng, n)
    obs_data = obs_data[perm]
    target_data = target_data[perm]

    num_batches = n // BATCH_SIZE
    total_loss = 0.0
    total_acc = jnp.zeros(NUM_COMPONENTS)

    for b in range(num_batches):
        start = b * BATCH_SIZE
        obs_batch = obs_data[start : start + BATCH_SIZE]
        tgt_batch = target_data[start : start + BATCH_SIZE]
        state, loss, acc = train_step(state, obs_batch, tgt_batch)
        total_loss += loss
        total_acc += acc

    mean_loss = total_loss / max(num_batches, 1)
    mean_acc = total_acc / max(num_batches, 1)
    return state, mean_loss, mean_acc, rng


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    rng = jax.random.PRNGKey(SEED)

    # Create environment
    env = make_env(ENV_NAME)
    print(f"Environment: {ENV_NAME} (game={GAME})")
    print(f"  obs_dim={OBS_DIM}, num_components={NUM_COMPONENTS}, vocab_size={VOCAB_SIZE}")

    # Initialize model
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng)
    num_params = sum(x.size for x in jax.tree.leaves(state.params))
    print(f"Model parameters: {num_params:,}")

    # JIT compile rollout collection
    print("Compiling rollout collector...")
    rng, collect_rng = jax.random.split(rng)
    collect_fn = jax.jit(
        partial(collect_rollouts, env, rollout_length=ROLLOUT_LENGTH),
        static_argnames=("num_rollouts",),
    )
    # Warmup
    _ = collect_fn(collect_rng, num_rollouts=2)
    print("Compilation done.\n")

    # Training loop
    iter_bar = tqdm(range(1, NUM_ITERATIONS + 1), desc="Iterations")
    for iteration in iter_bar:
        # Phase 1: Collect rollouts
        rng, collect_rng = jax.random.split(rng)
        all_obs, all_targets, all_dones = collect_fn(
            collect_rng, num_rollouts=NUM_ROLLOUTS
        )

        # Phase 2: Train for Y epochs
        epoch_bar = tqdm(range(1, NUM_EPOCHS + 1), desc=f"  Iter {iteration} epochs", leave=False)
        for epoch in epoch_bar:
            rng, epoch_rng = jax.random.split(rng)
            state, epoch_loss, epoch_acc, rng = train_epoch(
                state, all_obs, all_targets, epoch_rng
            )
            epoch_bar.set_postfix(loss=f"{epoch_loss:.4f}")

        # Phase 3: Report
        mean_acc = jnp.mean(epoch_acc)
        iter_bar.set_postfix(loss=f"{epoch_loss:.4f}", acc=f"{mean_acc:.4f}")
        tqdm.write(
            f"Iter {iteration:3d} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Acc: {mean_acc:.4f} | "
            f"Per-comp: [{', '.join(f'{a:.2f}' for a in epoch_acc)}]"
        )

    # Save model
    save_dir = os.path.dirname(__file__)
    save_path = os.path.join(save_dir, f"belief_model_checkpoint_{GAME}.msgpack")
    with open(save_path, "wb") as f:
        f.write(to_bytes(state.params))
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
