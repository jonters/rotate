"""
LSTM-based auto-regressive belief model (JAX/Flax).

Encoder:  LSTM that maps a sequence of observations to context vectors.
Decoder:  Auto-regressive LSTM that predicts each component of the hidden
          state one at a time, conditioned on the context and all previously
          predicted components.

All 15 target components are treated as categorical (grid coords + levels).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional


class BeliefEncoder(nn.Module):
    """LSTM encoder: observations -> context vectors."""
    hidden_size: int = 256

    @nn.compact
    def __call__(self, obs_seq, initial_state=None):
        """
        Args:
            obs_seq: (batch, seq_len, obs_dim)
            initial_state: optional (carry) for the LSTM
        Returns:
            contexts: (batch, seq_len, hidden_size)
            final_state: LSTM carry
        """
        batch_size = obs_seq.shape[0]

        cell = nn.OptimizedLSTMCell(features=self.hidden_size)
        if initial_state is None:
            initial_state = cell.initialize_carry(
                jax.random.PRNGKey(0), (batch_size, obs_seq.shape[-1])
            )

        scan_fn = nn.scan(
            nn.OptimizedLSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )(features=self.hidden_size)

        final_state, contexts = scan_fn(initial_state, obs_seq)
        return contexts, final_state


class BeliefDecoder(nn.Module):
    """Auto-regressive LSTM decoder: context -> component predictions."""
    hidden_size: int = 256
    num_components: int = 15
    vocab_size: int = 26       # number of categorical bins per component
    embed_dim: int = 64
    value_offset: int = 8      # offset so that value v maps to index v + offset

    @nn.compact
    def __call__(self, context, targets=None, rng=None):
        """
        Args:
            context: (batch, hidden_size) — encoder output at one timestep
            targets: (batch, num_components) int — ground truth for teacher forcing.
                     If None, samples autoregressively (inference mode).
            rng: PRNGKey, required for sampling at inference time.
        Returns:
            logits: (batch, num_components, vocab_size)
            samples: (batch, num_components) — sampled or teacher-forced values
        """
        batch_size = context.shape[0]

        # Component embedding table
        embed_table = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim)
        # Learned start token
        start_embed = self.param(
            "start_embed", nn.initializers.normal(0.02), (self.embed_dim,)
        )
        # Project context to initialize decoder hidden state
        init_proj = nn.Dense(self.hidden_size)
        cell = nn.OptimizedLSTMCell(features=self.hidden_size)
        output_head = nn.Dense(self.vocab_size)

        # Initialize decoder state from context
        h0 = nn.tanh(init_proj(context))
        c0 = jnp.zeros_like(h0)
        carry = (c0, h0)

        # First input: start token broadcast to batch
        prev_embed = jnp.broadcast_to(start_embed, (batch_size, self.embed_dim))

        all_logits = []
        all_samples = []

        for i in range(self.num_components):
            # Concatenate context with previous component embedding
            decoder_input = jnp.concatenate([context, prev_embed], axis=-1)
            carry, h = cell(carry, decoder_input)
            logits_i = output_head(h)  # (batch, vocab_size)
            all_logits.append(logits_i)

            if targets is not None:
                # Teacher forcing: use ground truth
                token_i = targets[:, i] + self.value_offset  # shift to index
                token_i = jnp.clip(token_i, 0, self.vocab_size - 1)
            else:
                # Sample from distribution
                assert rng is not None, "Need rng for inference sampling"
                rng, sub_rng = jax.random.split(rng)
                token_i = jax.random.categorical(sub_rng, logits_i, axis=-1)

            all_samples.append(token_i)
            prev_embed = embed_table(token_i)

        logits = jnp.stack(all_logits, axis=1)      # (batch, num_components, vocab_size)
        samples = jnp.stack(all_samples, axis=1)     # (batch, num_components)
        return logits, samples


class BeliefModel(nn.Module):
    """Full belief model: encoder + auto-regressive decoder."""
    encoder_hidden: int = 256
    decoder_hidden: int = 256
    obs_dim: int = 15
    num_components: int = 15
    vocab_size: int = 26
    embed_dim: int = 64
    value_offset: int = 8

    def setup(self):
        self.obs_embed = nn.Dense(self.encoder_hidden)
        self.encoder = BeliefEncoder(hidden_size=self.encoder_hidden)
        self.decoder = BeliefDecoder(
            hidden_size=self.decoder_hidden,
            num_components=self.num_components,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            value_offset=self.value_offset,
        )

    def __call__(self, obs_seq, targets, rng=None):
        """
        Training forward pass with teacher forcing.

        Args:
            obs_seq: (batch, seq_len, obs_dim) — observation trajectories
            targets: (batch, seq_len, num_components) — ground truth hidden state
            rng: unused during training
        Returns:
            loss: scalar, mean cross-entropy NLL
            per_component_acc: (num_components,) accuracy per component
        """
        batch, seq_len, _ = obs_seq.shape

        # Embed observations and encode
        embedded = nn.relu(self.obs_embed(obs_seq.astype(jnp.float32)))
        contexts, _ = self.encoder(embedded)  # (batch, seq_len, encoder_hidden)

        # Flatten time into batch for decoding
        ctx_flat = contexts.reshape(batch * seq_len, -1)
        tgt_flat = targets.reshape(batch * seq_len, self.num_components)

        logits, _ = self.decoder(ctx_flat, targets=tgt_flat)
        # logits: (batch*seq_len, num_components, vocab_size)

        # Compute cross-entropy loss
        tgt_indices = tgt_flat + self.value_offset
        tgt_indices = jnp.clip(tgt_indices, 0, self.vocab_size - 1)
        one_hot = jax.nn.one_hot(tgt_indices, self.vocab_size)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.sum(one_hot * log_probs, axis=-1)  # (batch*seq_len, num_components)
        mean_loss = jnp.mean(loss)

        # Per-component accuracy
        preds = jnp.argmax(logits, axis=-1)  # (batch*seq_len, num_components)
        acc = jnp.mean((preds == tgt_indices).astype(jnp.float32), axis=0)

        return mean_loss, acc

    def predict(self, obs_seq, rng, num_samples=1):
        """
        Inference: sample hidden state predictions.

        Args:
            obs_seq: (batch, seq_len, obs_dim)
            rng: PRNGKey
            num_samples: number of samples to draw per timestep
        Returns:
            samples: (num_samples, batch, seq_len, num_components) — in original value space
        """
        batch, seq_len, _ = obs_seq.shape
        embedded = nn.relu(self.obs_embed(obs_seq.astype(jnp.float32)))
        contexts, _ = self.encoder(embedded)
        ctx_flat = contexts.reshape(batch * seq_len, -1)

        all_samples = []
        for k in range(num_samples):
            rng, sub_rng = jax.random.split(rng)
            _, samples_k = self.decoder(ctx_flat, targets=None, rng=sub_rng)
            # Convert from token indices back to values
            samples_k = samples_k - self.value_offset
            samples_k = samples_k.reshape(batch, seq_len, self.num_components)
            all_samples.append(samples_k)

        return jnp.stack(all_samples, axis=0)

    def init_carry(self, batch_size=1):
        """Return a zero-initialized LSTM carry for streaming inference."""
        c = jnp.zeros((batch_size, self.encoder_hidden))
        h = jnp.zeros((batch_size, self.encoder_hidden))
        return (c, h)

    def step_encode(self, obs, carry):
        """Process a single observation and update the encoder LSTM carry.

        Args:
            obs: (batch, obs_dim) — single-step observation
            carry: LSTM carry tuple (c, h)
        Returns:
            context: (batch, encoder_hidden) — context vector for this timestep
            new_carry: updated LSTM carry
        """
        embedded = nn.relu(self.obs_embed(obs.astype(jnp.float32)))
        # Feed single timestep through encoder: add and remove seq dim
        embedded = embedded[:, None, :]  # (batch, 1, hidden)
        contexts, new_carry = self.encoder(embedded, initial_state=carry)
        context = contexts[:, 0, :]  # (batch, hidden)
        return context, new_carry

    def decode_belief(self, context, rng):
        """Decode a belief sample from an encoder context vector.

        Args:
            context: (batch, encoder_hidden)
            rng: PRNGKey
        Returns:
            sample: (batch, num_components) — predicted hidden state values
        """
        _, samples = self.decoder(context, targets=None, rng=rng)
        return samples - self.value_offset
