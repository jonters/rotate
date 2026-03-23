# Design Doc: Auto-Regressive Belief Predictor for Hidden State Estimation

## 1. Overview

This document specifies an encoder-decoder architecture for predicting a hidden-state vector from a trajectory of observations. The method is a generalization of the auto-regressive belief model from *Other-Play for Zero-Shot Coordination* (Hu et al.), originally applied to Hanabi hand prediction. Here we abstract it to a general setting.

### Problem Statement

Given a trajectory τ = (o₁, o₂, ..., o_T) of observations at each timestep, we want to predict a hidden-state vector **v** = (v₁, v₂, ..., v_n) that is not directly observable. We assume:

- The agent receives partial observations at each timestep but cannot directly see **v**.
- **v** is a structured vector whose components are interdependent (i.e., knowing v₁ gives you information about v₂).
- **v** may change at each timestep, so the model must produce a prediction at any point in the trajectory.

### Why Auto-Regressive?

A naive approach would predict all components of **v** independently in one shot. This fails to capture correlations between components. By predicting v₁, then v₂ conditioned on v₁, etc., the model factorizes the joint distribution:

```
p(v | τ) = p(v₁ | τ) · p(v₂ | τ, v₁) · p(v₃ | τ, v₁, v₂) · ... · p(v_n | τ, v₁:n-1)
```

This captures all pairwise and higher-order dependencies between components of **v**.

---

## 2. Architecture

The model consists of two modules: an **Encoder** and an **Auto-Regressive Decoder**.

### 2.1 Encoder

**Purpose:** Compress the observation trajectory into a fixed-size context vector.

**Input:** A sequence of observation vectors (o₁, o₂, ..., o_T), where each o_t ∈ ℝ^d_obs.

**Architecture:** An LSTM (or GRU) that processes observations sequentially.

```
h_t^E = LSTM_encoder(o_t, h_{t-1}^E)
x_t = h_t^E   (context vector at time t)
```

**Output:** A context vector x_t ∈ ℝ^d_ctx at each timestep t, summarizing all observations up to and including time t.

**Implementation Notes:**

- Input observations should be embedded/preprocessed before feeding into the LSTM. If observations are discrete or multimodal (e.g., a mix of categorical and continuous features), apply appropriate embedding layers first. Define an `ObservationEncoder` module that maps raw observations to a fixed-size vector before the LSTM.
- The LSTM can be unidirectional (causal, for online/real-time prediction) or bidirectional (if the full trajectory is available at prediction time). Default to unidirectional for generality.
- Hidden size `d_ctx` is a hyperparameter. Start with 256 or 512.

### 2.2 Auto-Regressive Decoder

**Purpose:** Predict each component of the hidden vector **v** one at a time, conditioned on the context vector and all previously predicted components.

**Architecture:** A Decoder LSTM that runs for n steps (one per component of **v**).

```
For i = 1 to n:
    input_i = concat(x_t, embed(v_{i-1}))      # v_0 is a learned start token
    h_i^D = LSTM_decoder(input_i, h_{i-1}^D)
    logits_i = Linear(h_i^D)
    p(v_i | τ, v_{1:i-1}) = output_head(logits_i)
```

**Output at each step:** A probability distribution over possible values for component v_i.

**Implementation Notes:**

- `h_0^D` (initial decoder hidden state): Use a learned initial state, or project x_t through a linear layer to initialize it: `h_0^D = tanh(W_init · x_t + b_init)`. Experiment with both.
- `v_0` (input at the first decoding step): Use a learned start-of-sequence embedding vector.
- x_t is fed at **every** decoding step, not just the first. This is important — the context does not change during decoding.
- The `embed()` function maps a predicted component value to a dense vector. Its design depends on the type of v_i (see Section 3).

### 2.3 Output Head

The output head depends on the type of each component v_i:

| Component Type | Output Head | Loss |
|---|---|---|
| **Categorical** (finite set of values) | Linear → Softmax | Cross-entropy |
| **Continuous scalar** | Linear → μ, σ (Gaussian) | Negative log-likelihood of Gaussian |
| **Continuous vector** | Linear → μ, Σ (multivariate Gaussian) or use mixture density network | Negative log-likelihood |
| **Binary** | Linear → Sigmoid | Binary cross-entropy |

If all components share the same type (e.g., all categorical with the same vocabulary), the decoder can reuse a single output head. If components are heterogeneous, use per-component output heads selected by index i.

---

## 3. Component Embedding

The `embed()` function used to feed back predicted components varies by type:

| Component Type | Embedding Strategy |
|---|---|
| **Categorical** | Learned embedding table: `nn.Embedding(vocab_size, d_embed)` |
| **Continuous scalar** | Linear projection: `nn.Linear(1, d_embed)` |
| **Continuous vector** | Linear projection: `nn.Linear(d_component, d_embed)` |
| **Binary** | Learned embedding table with 2 entries |

`d_embed` should match or be close to `d_ctx` for balanced concatenation. A typical choice is `d_embed = d_ctx // 4` or `d_embed = 64`.

---

## 4. Training

### 4.1 Loss Function

Train end-to-end with maximum likelihood. The loss for a single sample is:

```
L(v | τ) = - Σ_{i=1}^{n} log p(v_i | τ, v_{1:i-1})
```

During training, use **teacher forcing**: feed the ground-truth v_{i-1} as input at step i, not the model's own prediction. This is standard for auto-regressive model training.

### 4.2 Training Loop: Collect-Train-Report-Repeat

Training follows a simple iterative loop controlled by two config values: `num_rollouts` (X) and `num_epochs` (Y).

```
for iteration = 1, 2, 3, ...:
    # Phase 1: Collect
    rollouts = collect_rollouts(policy, num_rollouts=X)
    # Each rollout is a trajectory τ paired with ground-truth hidden vectors v at each timestep.

    # Phase 2: Train
    for epoch = 1 to Y:
        for batch in shuffle_and_batch(rollouts):
            loss = belief_model.forward(batch.observations, batch.ground_truth_v)
            loss.backward()
            optimizer.step()

    # Phase 3: Report
    metrics = evaluate(belief_model, rollouts)  # or a held-out subset
    log(iteration, metrics)

    # Phase 4: Repeat
    # Loop continues. Each iteration collects fresh rollouts, so the model
    # sees new data every cycle. The policy remains fixed throughout.
```

**Config parameters:**

| Parameter | Description |
|---|---|
| `num_rollouts` (X) | Number of trajectories to collect per iteration |
| `num_epochs` (Y) | Number of training epochs over the collected rollouts per iteration |
| `num_iterations` | Total number of collect-train-report cycles (or run until convergence) |

**Notes:**

- The policy used to collect rollouts is **fixed** — it is not being updated. It only serves to generate realistic trajectories.
- Each iteration uses freshly collected data. This means the model trains on different rollouts each cycle, providing natural regularization across iterations.
- Within a single iteration, the model trains for Y epochs on the same X rollouts, so standard overfitting concerns apply. Keep Y moderate (e.g., 5–20) or monitor validation loss within each iteration.
- Optionally hold out a fraction of each iteration's rollouts (e.g., 10%) as a validation set for the metrics reported in Phase 3.

### 4.3 Training Hyperparameters (Starting Points)

| Parameter | Suggested Value |
|---|---|
| num_rollouts (X) | 1000-10000 per iteration |
| num_epochs (Y) | 5-20 per iteration |
| num_iterations | Until convergence (monitor NLL) |
| Optimizer | Adam |
| Learning rate | 1e-3 to 1e-4 |
| Batch size | 128-512 |
| Encoder LSTM layers | 1-2 |
| Decoder LSTM layers | 1 |
| d_ctx (encoder hidden size) | 256-512 |
| d_embed (component embedding) | 64-128 |
| Gradient clipping | 1.0 |
| Teacher forcing ratio | 1.0 during training |

---

## 5. Inference

At inference time, the model predicts **v** by **sampling** from the decoder, not argmaxing. This is critical because the belief model represents a **distribution** over possible hidden states, not a point estimate.

### 5.1 Single-Sample Inference

```
x_t = Encoder(o_1, ..., o_T)
v_0 = start_token
for i = 1 to n:
    p_i = Decoder_step(x_t, embed(v_{i-1}), h_{i-1}^D)
    v_i ~ p_i       # SAMPLE, don't argmax
return (v_1, ..., v_n)
```

### 5.2 Multi-Sample Inference (for downstream use)

To approximate the full belief distribution, draw K independent samples:

```
for k = 1 to K:
    v^(k) = sample_from_decoder(x_t)
return {v^(1), ..., v^(K)}
```

This set of samples can then be used by a downstream planner or policy to reason about the hidden state under uncertainty (e.g., by evaluating actions against each sampled hidden state and averaging).

---

## 6. Interface Specification

### 6.1 Module: `ObservationEncoder`

Preprocesses raw observations into a fixed-size vector suitable for the Encoder LSTM.

```
Input:  raw_observation (dict, tensor, or environment-specific format)
Output: o_t ∈ ℝ^d_obs
```

This module is **environment-specific** and must be implemented per use case.

### 6.2 Module: `BeliefEncoder`

```
Input:  sequence of o_t vectors, shape (batch, seq_len, d_obs)
Output: x_t at each timestep, shape (batch, seq_len, d_ctx)
```

### 6.3 Module: `BeliefDecoder`

```
Input:  x_t (batch, d_ctx), n (number of components to predict)
Output: list of n distributions (one per component), sampled values (batch, n)
```

### 6.4 Module: `BeliefModel` (Top-Level)

Wraps encoder + decoder. Exposes:

- `forward(observations, ground_truth_v)` → loss (for training with teacher forcing)
- `predict(observations, num_samples=1)` → sampled hidden vectors (for inference)

---

## 7. Ordering of Components

The decoder predicts components in a fixed order (v₁ first, v_n last). The choice of ordering can affect performance:

- **If there is a natural ordering** (e.g., temporal, spatial, oldest-to-newest): use it.
- **If there is no natural ordering:** experiment with different orderings. Heuristics: order by decreasing variance or information content so that the most informative components are predicted first and can condition later predictions.
- **Advanced:** learn the ordering, or use a permutation-invariant approach (e.g., predict in all orderings and average losses). This adds complexity and is not recommended as a first pass.

---

## 8. Evaluation Metrics

| Metric | What It Measures |
|---|---|
| **Per-component accuracy** (categorical) | Fraction of correctly predicted components when argmaxing |
| **Negative log-likelihood** | Direct measure of how well the model captures the true distribution |
| **Calibration** | Whether predicted probabilities match empirical frequencies |
| **Sample diversity** | Entropy or variance across K sampled hidden vectors (should match true uncertainty) |
| **Downstream task performance** | If the belief model feeds into a planner/policy, measure end-to-end task reward |

The most important metric is **negative log-likelihood** — it directly corresponds to the training objective and measures distributional accuracy, not just point-prediction quality.

---

## 9. File Structure

```
belief_model/
├── model/
│   ├── encoder.py          # BeliefEncoder (LSTM)
│   ├── decoder.py          # BeliefDecoder (auto-regressive LSTM)
│   ├── belief_model.py     # Top-level BeliefModel combining encoder + decoder
│   ├── observation_encoder.py  # Environment-specific observation preprocessing
│   └── output_heads.py     # Categorical, Gaussian, etc.
├── training/
│   ├── trainer.py          # Collect-train-report-repeat loop
│   ├── rollout_collector.py # Collects X rollouts from a fixed policy
│   └── metrics.py          # Evaluation and metric logging
├── inference/
│   └── sampler.py          # Single-sample and multi-sample inference
├── configs/
│   └── default.yaml        # Hyperparameters
├── tests/
│   ├── test_encoder.py
│   ├── test_decoder.py
│   └── test_end_to_end.py
└── README.md
```

---

## 10. Implementation Checklist

1. Implement `ObservationEncoder` for your specific environment/domain.
2. Implement `BeliefEncoder` (standard LSTM, straightforward).
3. Implement component embedding layer(s) based on your component types.
4. Implement `BeliefDecoder` with the auto-regressive loop, feeding x_t at every step.
5. Implement output head(s) matching your component types.
6. Wire into `BeliefModel` with teacher-forcing `forward()` and sampling `predict()`.
7. Implement rollout collector that runs the fixed policy for X episodes and packages (τ, v) pairs.
8. Implement the collect-train-report-repeat loop with configurable X, Y, and num_iterations.
9. Train, monitoring NLL on held-out data each iteration.
10. Evaluate calibration and downstream performance.

---

## 11. Key Design Decisions to Make Per Application

| Decision | Options |
|---|---|
| Component type (categorical, continuous, mixed) | Determines output heads and embedding |
| Number of components n | Fixed or variable per sample? If variable, need a stop token. |
| Component ordering | Natural order, by information content, or learned |
| Encoder type | LSTM (default), GRU, or Transformer for very long trajectories |
| Data source | Collect-train-report-repeat loop with configurable X rollouts and Y epochs |
| Decoder init | Learned parameter vs. projected encoder output |
| Inference mode | Single sample, multi-sample, or beam search |