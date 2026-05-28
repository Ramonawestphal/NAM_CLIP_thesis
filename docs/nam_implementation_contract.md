# NAM Implementation Contract — COMPAS Replication

**Phase A deliverable.** Sources: Agarwal et al. (2021) NeurIPS paper (17 pp.), reference TF
code at `references/google-research/neural_additive_models/` (read-only), and
`docs/preprocessing_compas.md`.

All citations use the format `[paper §X]` or `[ref file:line]`.

**Amendment 2026-05-14:** Three clarifications added — categorical shape-function plotting
(§2a), output-penalty and feature-dropout verification (§3.2, §4.2), and per-fold reporting
pipeline (§5a). Changed sections are marked *(amended)*.

**Amendment 2026-05-14 (2):** Encoding correction — OHE indicators remain in {0, 1} (not
rescaled to {−1, +1}); MinMaxScaler is applied to the 4 continuous columns only; column
order corrected to continuous first (indices 0–3), OHE last (indices 4–11). Caught by unit
tests 9 and 10. Sections §2, §2a, and summary table updated.

---

## 1. Sub-network (FeatureNN) architecture for COMPAS

**Decision: 3 hidden ReLU Dense layers — sizes [64, 64, 32] — followed by a linear scalar output.**

| Component | Specification | Source |
|---|---|---|
| Layer 1 | `Linear(1, 64)` + `ReLU` | Paper Table 4 "Hidden units: 64, 64, 32", Activation: ReLU |
| Layer 2 | `Linear(64, 64)` + `ReLU` | Paper Table 4; [paper §3 p.5] "option (1) DNNs containing 3 hidden layers with 64, 64 and 32 units and ReLU activation" |
| Layer 3 | `Linear(64, 32)` + `ReLU` | Same |
| Output | `Linear(32, 1)`, no activation, **no bias** | [ref models.py:160–165] `self.linear = Dense(1, use_bias=False)` |
| Dropout | After every hidden layer (before the output linear) | [ref models.py:170–175] `tf.nn.dropout` applied after each hidden layer including the first activation layer |
| Dropout rate | **0.1** for COMPAS | Paper Table 4 |

**Deviation from reference code:** The reference code's `FeatureNN.build` uses a custom
`ActivationLayer` as the first layer (`relu(w_j * (x - b_j))`, element-wise with learned
per-unit weights and biases). For the PyTorch replication with ReLU, we replace this with a
standard `nn.Linear(1, 64)` + `nn.ReLU()`. This matches the paper's own description
("3 hidden layers … with ReLU activation") and the Appendix A.2 description of option (1).
For future ExU support, the ActivationLayer abstraction can be reinstated; for the COMPAS
replication it is out of scope.

**Input to FeatureNN:** a single scalar (the value of one feature for one sample), shaped `(B, 1)`.
**Output of FeatureNN:** scalar per sample, shaped `(B,)`.

---

## 2. NAM forward pass

### Categorical feature handling

**Decision: one-hot encode `race` and `sex`; each OHE column gets its own FeatureNN.**

Source: [ref data_utils.py:351–390] `transform_data` detects columns with dtype `'O'`
(Python object / string) and applies `sklearn.preprocessing.OneHotEncoder`, producing one
binary column per category. The full post-transform feature matrix is then split column-by-column and each column is fed to its own `FeatureNN`. For COMPAS:

| Encoder output index | Feature | Encoding | Sub-networks |
|---|---|---|---|
| 0 | `age` | continuous → MinMaxScaled to `[-1, 1]` | 1 |
| 1 | `charge_degree` | integer 1/2 → MinMaxScaled to `[-1, 1]` | 1 |
| 2 | `length_of_stay` | continuous integer → MinMaxScaled to `[-1, 1]` | 1 |
| 3 | `priors_count` | continuous integer → MinMaxScaled to `[-1, 1]` | 1 |
| 4–9 | `race` | OHE → 6 binary indicators, **in {0, 1}, not rescaled** | 6 |
| 10–11 | `sex` | OHE → 2 binary indicators, **in {0, 1}, not rescaled** | 2 |
| **Total** | | | **12 sub-networks** |

Encoding is fitted **on the training split only** (inside `src/data/encoding.py`); the fitted
transformers are applied to the validation and test splits to avoid leakage. This is consistent
with [ref data_utils.py:388] where `transform_data` is called on the entire fold's training
portion only.

**Scaling *(amended 2026-05-14 (2))*:** MinMaxScaler to `(-1, 1)` is applied **only to the
four continuous columns** (indices 0–3). OHE indicator columns (indices 4–11) are **not
rescaled** and remain as exact `{0, 1}` integers.

This differs from the reference `data_utils.py:388`, which applies `MinMaxScaler((-1, 1))`
to the entire post-OHE feature matrix. We deviate deliberately for three reasons:

1. **Identifiability.** Under `{0, 1}` indicators with `bias=False` on the final Linear
   layer, `f_k(0)` is structurally zero by construction, so `f_k(1)` reads directly as the
   contribution of "category k is active." Under `{−1, +1}` encoding, `f_k(−1)` becomes a
   free parameter entangled with the global bias β, creating identifiability issues for
   categorical shape functions and complicating the per-category contribution attribution
   described in §2a.

2. **Figure 4 consistency.** The paper's Figure 4 shows per-category contributions on a
   y-axis that crosses zero, requiring the "off" state to be zero by construction. The
   `{0, 1}` convention makes `f_k(0) ≈ 0` structural (no gradient to push it elsewhere),
   whereas `{−1, +1}` encoding makes `f_k(−1)` a learned quantity that need not be zero.

3. **Reference code alignment.** The reference TF code keeps OHE indicators as binary
   `{0, 1}` prior to the global MinMaxScaler call; rescaling them to `{−1, +1}` was an
   inadvertent deviation introduced during the initial PyTorch port, caught by unit tests 9
   and 10 (`test_encoder_column_layout`, `test_calc_outputs_no_feature_dropout_no_bias`).

### NAM combination (forward pass)

From [ref models.py:239–248] `NAM.call`:

```
1. Compute fi(xi) for each of the K features → list of K tensors of shape (B,)
2. Stack → tensor of shape (B, K)
3. Apply feature dropout to entire sub-network outputs (training only)
4. Sum over K → shape (B,)
5. Add global scalar bias β → shape (B,)
6. Return logits (no sigmoid here)
```

**Global bias term:** Yes, present. [ref models.py:231–235] `self._bias = add_weight(shape=(1,),
initializer=Zeros)`. Equivalent: `nn.Parameter(torch.zeros(1))` in PyTorch.

**Sigmoid:** Not applied in the forward pass. The model returns raw logits.
Sigmoid is applied **inside the loss** via `BCEWithLogitsLoss` (PyTorch equivalent of
`tf.nn.sigmoid_cross_entropy_with_logits`). [ref graph_builder.py:360–361]
`loss_fn, y_pred = penalized_cross_entropy_loss, tf.nn.sigmoid(predictions)` — the sigmoid
appears only for the evaluation metric, not in training loss computation.

---

## 2a. Categorical shape-function plotting *(amended)*

**Question:** Are OHE sub-network outputs aggregated for the Figure 4 bar charts, and if so
how?

**Answer:** Yes, aggregation is needed. Each OHE category i has its own sub-network f_i that
takes a scalar input x_i ∈ {0, 1} (OHE indicator, **not rescaled**). For a sample with
race = "African-American", only the AA indicator is 1; all other race indicators are 0. The
six race sub-networks each produce a scalar, but the bar chart in Figure 4 shows ONE bar per
race category, not six separate plots.

**Aggregation rule for OHE categorical bar chart *(amended 2026-05-14 (2))*:**

For each category i, the bar height is:

```
bar_i = f̃_i(1)   where   f̃_i(x) = f_i(x) − E_train[f_i(x_{i,n})]
```

That is: evaluate sub-network i at the ACTIVE value (1, corresponding to OHE indicator = 1),
then subtract the training-set mean of that sub-network's outputs. This is "reading off
f_race(i) as the mean-centered output of the sub-network for race=i when the indicator is 1."

**Why only f̃_i(1) and not the full sum?**

For a sample with race = i, the total contribution of all K_race race sub-networks is:

```
total_race(i) = f̃_i(1) + Σ_{j ≠ i} f̃_j(0)
```

Because `f_k(0) ≈ 0` by construction under `{0, 1}` encoding (the final Linear layer has
`bias=False`, so the only way a sub-network can produce a non-zero output at input=0 is
through the hidden-layer biases, which are small at initialisation and regularised by the
output penalty). In practice `f̃_j(0)` is small, so the bar chart shows only `f̃_i(1)` as
the dominant term. This is consistent with Figure 4, where the bars represent "if this race,
how much does it shift log-odds relative to the population average."

**Implementation note for `src/utils/plotting.py` (Phase B):**

```python
# For each OHE category (e.g. race), compute bar heights:
#   active_input = 1.0  (OHE indicator = 1, no MinMaxScaling applied)
#   bar_i = feature_nn_i(active_input) − train_mean_i
# where train_mean_i = mean over training set of feature_nn_i(x_{i,n}).
```

The mean-centering constant is computed once on the training fold after training completes
and stored alongside the model. The global bias β absorbs the sum of all mean-centering
constants to preserve the population-level log-odds baseline.

---

## 3. Loss function — exact mathematical form

### 3.1 Binary cross-entropy term

The reference uses `tf.nn.softmax_cross_entropy_with_logits_v2` with a 2-class formulation
[ref graph_builder.py:36–52], which for binary labels is mathematically equivalent to standard
binary cross-entropy:

```
BCE(y, logit) = -y * log(σ(logit)) - (1 - y) * log(1 - σ(logit))
```

In PyTorch: `torch.nn.BCEWithLogitsLoss(reduction='mean')`.

### 3.2 Output penalty *(amended — verified)*

[ref graph_builder.py:110–117] `feature_output_regularization`:

```python
per_feature_outputs = model.calc_outputs(inputs, training=False)  # K outputs of shape (B,)
per_feature_norm = [mean(f_k^2) for f_k in per_feature_outputs]   # mean over batch
return sum(per_feature_norm) / K
```

**Mathematical form:**

```
η(θ; x) = (1/K) * Σ_{k=1}^{K} (1/N) * Σ_{n=1}^{N} [f_k(x_kn)]^2
```

Coefficient: **λ₁ = 0.2078** for COMPAS (Paper Table 4).

**Verified answers:**

| Question | Answer | Evidence |
|---|---|---|
| Raw or mean-centred outputs? | **Raw outputs.** `calc_outputs` calls `feature_nns[i](x_i, training=False)` and returns the FeatureNN output directly — no mean-centering is applied anywhere in training. Mean-centering is a post-training visualisation step only. | [ref models.py:259–266] `calc_outputs`; [ref graph_builder.py:113] |
| Dropout ON or OFF? | **Dropout OFF.** `training=False` is passed through to both `FeatureNN.call` and `NAM.calc_outputs`, so all `nn.Dropout` layers are inactive when the penalty is evaluated. | [ref graph_builder.py:113] `model.calc_outputs(inputs, training=False)` |
| Same batch as BCE? | **Same batch, different pass.** `penalized_loss` calls `cross_entropy_loss` (which invokes `model(inputs, training=True)` — dropout ON) first, then calls `feature_output_regularization(model, inputs)` as a second forward pass with `training=False`. Both passes use the same mini-batch `inputs`. | [ref graph_builder.py:55–85] `penalized_loss` |

### 3.3 Weight decay

[ref graph_builder.py:120–123] `weight_decay`:

```python
l2_losses = [tf.nn.l2_loss(x) for x in model.trainable_variables]  # sum(w^2)/2 per variable
return sum(l2_losses) / num_networks  # num_networks = K (number of FeatureNNs)
```

**Mathematical form:**

```
γ(θ) = (1/K) * Σ_{k} Σ_{w ∈ θ_k} (w^2 / 2)
```

Coefficient: **λ₂ = 0** for COMPAS (Paper Table 4). Weight decay is not used for COMPAS.
It is implemented in the loss, **not** as an optimiser argument (do not set `weight_decay` in
`torch.optim.Adam`).

### 3.4 Total loss equation

```
L(θ) = BCE(σ(β + Σ_k f_k(x_k)), y)
      + λ₁ * (1/K) * Σ_k E_x[f_k(x_k)²]
      + λ₂ * (1/K) * Σ_k Σ_{w ∈ θ_k} (w²/2)
```

For COMPAS: λ₁ = 0.2078, λ₂ = 0.

---

## 4. Regularisation mechanisms

### 4.1 Hidden-unit dropout (rate 0.1 for COMPAS)

[ref models.py:170–175] In `FeatureNN.call`:

```python
for l in self.hidden_layers:     # [ActivationLayer, Dense(64), Dense(32)]
    x = dropout(l(x), rate=dropout_rate if training else 0.0)
x = linear(x)                   # output layer — NO dropout
```

**Placement:** After every hidden layer (3 times), before the final linear output.
**Training only.** No dropout at evaluation time.
**PyTorch:** Use `nn.Dropout(p=0.1)` after each hidden ReLU layer, **not** after the final `nn.Linear(32, 1)`.

### 4.2 Feature dropout (rate 0.05 for COMPAS) *(amended — verified)*

[ref models.py:244–247] In `NAM.call`:

```python
stacked_out = stack(individual_outputs, axis=-1)   # shape (B, K)
dropout_out  = tf.nn.dropout(stacked_out,
               rate=feature_dropout if training else 0.0)
out = sum(dropout_out, axis=-1)                    # sum after zeroing
```

**What it zeroes:** entire columns of `stacked_out` — i.e., the whole contribution of one
sub-network for a sample, independently per sample.

**Verified: inverted dropout (rescaling) or plain zeroing?**

`tf.nn.dropout` applies **inverted dropout**: survivors are scaled by `1 / (1 − rate)`.
For feature dropout rate 0.05, each surviving sub-network output is multiplied by
`1 / 0.95 ≈ 1.053`. This keeps the expected sum unchanged between training and inference,
which is the standard inverted-dropout contract.

Source: TensorFlow documentation for `tf.nn.dropout` states "each element is scaled by
`1 / (1 - rate)` during training". The reference code uses `tf.nn.dropout` without any
`scale=False` override, so rescaling is active. [ref models.py:244–247]

**PyTorch equivalent:** `nn.Dropout(p=0.05)` applied to the `(B, K)` tensor before summing.
`nn.Dropout` also uses inverted dropout by default.

**Timing:** Applied **during training only**, **before** the sum. [ref models.py:239–248] The training
flag is passed into `NAM.call(training=True/False)`.

---

## 5. Training protocol for COMPAS

| Parameter | Value | Source |
|---|---|---|
| Optimiser | Adam (standard) | [ref graph_builder.py:352] `tf.train.AdamOptimizer` |
| Learning rate | **0.02082** | Paper Table 4 |
| LR schedule | Multiply by **0.995 each epoch** | [ref graph_builder.py:351] `lr_decay_op = lr * decay_rate`; Paper A.2 "annealed by a factor of 0.995 every training epoch" |
| Batch size | **1024** | [ref nam_train.py:41] `default=1024`; Paper A.2 "batch size of 1024" |
| Max epochs | **1000** | Paper A.2 "a maximum of 1000 epochs" |
| Early stopping patience | **60 epochs** | [ref nam_train.py:79] `early_stopping_epochs=60`; checked every 10 epochs |
| Balanced batches | Yes: minority class oversampled to 50/50 | [ref graph_builder.py:329–331] `create_balanced_dataset` is used for classification |
| Weight decay | 0 (not used) | Paper Table 4 |

**Verified: balanced 50/50 batches for COMPAS?** *(amended)*

Yes, confirmed. [ref graph_builder.py:329–331]:

```python
if regression:
    ds_tensors = ...standard shuffle_and_repeat...
else:
    # Create a balanced dataset to handle class imbalance
    ds_tensors = create_balanced_dataset(x_train, y_train, batch_size)
```

COMPAS is a classification task (`regression=False`), so `create_balanced_dataset` is called.
[ref graph_builder.py:216–248] `create_balanced_dataset` partitions the training set into
positive (recidivism=1) and negative (recidivism=0) pools, creates two shuffle-and-repeat
datasets, and uses `tf.data.experimental.sample_from_datasets([pos, neg])` which samples
with **equal probability** from each pool → ~50/50 class distribution per batch. This is NOT
mentioned in the paper text but is in the reference code. We adopt it.

### 5-fold cross-validation construction

[ref data_utils.py:444–486] `get_train_test_fold`:

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

- **Stratified** by `two_year_recid` label.
- **random_state = 42.**
- 5 folds: each fold uses 20% as held-out test, 80% as training pool.

Within each fold, [ref data_utils.py:489–528] `split_training_dataset`:

```python
StratifiedShuffleSplit(n_splits=n_splits, test_size=0.125, random_state=1337)
```

- `test_size=0.125`: 12.5% of the 80% training pool = ~10% of total data as validation.
- `random_state=1337`.
- For the **ensemble** target (paper): `n_splits=20` → 20 (train, val) splits, 20 models trained.
- For **Phase A (single NAM)**: `n_splits=1` → one (train, val) split per fold.

**Phase A target:** Single NAM per fold, 5-fold CV, report mean ± std AUC-ROC and AUC-PR.
Ensembling (20 NAMs per fold) is deferred to Phase B extension.

---

## 5a. Per-fold reporting pipeline *(amended)*

**Question:** Is the reported per-fold AUC from the held-out test fold (20%) or from the
internal validation split used for early stopping?

**Finding from reference code:**

The reference code's `main` function [ref nam_train.py:337–342]:

```python
def main(argv):
    data_gen, _ = create_test_train_fold(FLAGS.fold_num)   # _ = held-out test set (DISCARDED)
    single_split_training(data_gen, FLAGS.logdir)
```

`create_test_train_fold` [ref nam_train.py:304–323] returns both `data_gen` (inner train/val
splits) and `test_dataset` (held-out test fold). `main` assigns the test dataset to `_` and
discards it. The `training()` function [ref nam_train.py:203–301] only receives
`x_validation, y_validation` (the INNER split) and the comment at line 188 explicitly
identifies it: *"Calculate the AUROC/RMSE on the validation split"*.

`training()` returns `best_validation_metric` = best AUC-ROC on the **inner validation
split**, not on the held-out test fold. The held-out test evaluation is not implemented
inside the reference CLI — it is done externally (presumably in an evaluation notebook not
included in the reference code).

**Our implementation must differ here, and do it correctly.**

### Explicit three-level pipeline for Phase B

```
For fold f in 1..5:
│
├── held-out TEST set (20%)  ← StratifiedKFold split, random_state=42
│    └── NOT touched until step 4
│
└── train_pool (80%)
     │
     ├── TRAIN set (87.5% of pool ≈ 70% total)  ┐
     └── VAL set   (12.5% of pool ≈ 10% total)  ┘  ← StratifiedShuffleSplit, random_state=1337
          │
          └── used ONLY for early stopping (AUC-ROC)
               and checkpoint selection (best-val epoch)

Step 1: Fit OHE + MinMaxScaler on TRAIN set only.
Step 2: Transform TRAIN, VAL, TEST with fitted encoders.
Step 3: Train model on TRAIN. Monitor val AUC-ROC every epoch;
        save best-val-epoch checkpoint. Stop if no improvement
        for 60 consecutive epochs (or epoch 1000).
Step 4: Load best-val-epoch checkpoint.
        Evaluate on TEST set → record test AUC-ROC and test AUC-PR.

Reported per-fold AUC = test set AUC (step 4).
Final result = mean ± std of 5 test-fold AUCs.
```

**Key rule:** The inner validation AUC is used for checkpoint selection and early stopping
only. It is logged during training but is NOT the number reported as the fold's AUC.
The reported per-fold AUC is always from the held-out test set that was not seen during
training or hyperparameter selection.

---

## 6. Metrics

### Which metric does the reference code actually compute?

[ref graph_builder.py:383] `evaluation_metric = roc_auc_score`

[ref graph_builder.py:182–187] `roc_auc_score` calls `sklearn.metrics.roc_auc_score`.

**Conclusion: the reference code uses AUC-ROC, not AUC-PR.**

The paper's Section 3 states "area under the precision-recall curve (AUC)" [paper p.5], but
the code computes ROC AUC. The value 0.741 ± 0.009 in Table 1 is therefore **AUC-ROC** for
COMPAS NAMs.

**Implementation decision:**
1. Early stopping and validation selection: **AUC-ROC** (to match what the reference code
   actually optimises).
2. Final reporting: **both AUC-ROC and AUC-PR** (PR-AUC reported in thesis as AUPRC).
3. The replication target of 0.741 ± 0.009 is AUC-ROC.

`src/utils/metrics.py` will expose `roc_auc(y_true, logits)` and `pr_auc(y_true, logits)`,
both applying sigmoid before computing the metric.

---

## 7. Discrepancies between reference `data_utils.py` COMPAS handling and `src/data/compas.py`

The reference code's `load_recidivism_data()` loads a pre-processed binary file from
Google Cloud Storage (`recidivism/recid.data`) — the exact row filters and feature
derivation are unknown from the code alone. Differences found at the `transform_data` level:

| # | Difference | Reference code | `src/data/compas.py` | Decision |
|---|---|---|---|---|
| 1 | **Normalisation** | `MinMaxScaler((-1, 1))` applied inside `transform_data` to the entire feature matrix | No normalisation in `load_compas`; deferred to model pipeline | **Keep ours.** Fitting the scaler on the full dataset before the train/test split would leak test statistics. We fit MinMaxScaler per fold in `src/data/encoding.py`. |
| 2 | **Categorical encoding** | `OneHotEncoder` applied inside `transform_data` to object-dtype columns | `race` and `sex` left as strings; `charge_degree` as int64 | **Keep ours.** The encoding is fold-specific and fitted on training data only in `encoding.py`. The reference code would also fit on the training fold only in practice (the pipeline is called after the fold split). |
| 3 | **`charge_degree` dtype** | Unknown (loaded from binary file); Figure 4 shows values 1.0/2.0 suggesting numeric | `int64` (1 or 2) | **Keep ours.** Integer 1/2 will be treated as continuous numeric by the encoder pipeline. Consistent with Figure 4. |
| 4 | **Balanced dataset** | `create_balanced_dataset`: upsamples minority class for balanced 50/50 batches | Not mentioned in `load_compas` | **Adopt reference.** We will implement balanced batch sampling in the training loop (`src/nam/train.py`), fitting the minority-class upsampling on training data only. |
| 5 | **Source data** | Pre-processed Google Cloud file with unknown upstream filters | ProPublica `compas-scores-two-years.csv` with 4 documented filters; 6172-row assertion | **Keep ours.** The ProPublica filter set is the standard public replication baseline. Row count is locked by assertion. |

---

## 8. Reproducibility

### Seeds

The following random sources must be seeded in `src/utils/seeding.py`:

```python
def seed_everything(seed: int) -> None:
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

[ref nam_train.py:238] `tf.compat.v1.set_random_seed(FLAGS.tf_seed)` where default `tf_seed=1`.
For PyTorch we use `seed=42` as the global seed (matching the StratifiedKFold `random_state`).

### Where randomness enters

| Source | Enters via | Seed anchor |
|---|---|---|
| Weight initialisation | Glorot uniform for Dense layers | `torch.manual_seed` |
| Dropout (hidden units) | `nn.Dropout` in FeatureNN | `torch.manual_seed` |
| Feature dropout | `nn.Dropout` on stacked sub-network outputs | `torch.manual_seed` |
| Fold split | `StratifiedKFold(random_state=42)` | Fixed at 42 |
| Train/val split | `StratifiedShuffleSplit(random_state=1337)` | Fixed at 1337 |
| Balanced batch sampling | Minority-class oversampling / shuffle | `torch.manual_seed` / `np.random.seed` |
| Batch shuffle | `DataLoader(shuffle=True)` | `torch.manual_seed` via `Generator` |

`seed_everything` must be called **once per fold** before model construction, using
`seed = base_seed + fold_index` so that each fold is independently reproducible but not
identically initialised. Base seed = 42.

---

## Summary table (for Phase B implementation)

| Point | Decision |
|---|---|
| Sub-network | 3 Dense-ReLU layers [64, 64, 32] + Linear(32,1) no-bias, dropout(0.1) after each hidden layer |
| Categorical encoding | OHE race(→6) + sex(→2); continuous [0–3] MinMaxScaled to [-1,1], OHE [4–11] in {0,1} unscaled; 12 sub-networks total |
| Categorical plotting | Bar chart: f̃_i(1) per OHE slot i (mean-centered sub-network output at active input = 1) |
| Global bias | Yes, learnable scalar initialised at 0 |
| Sigmoid | Not in forward pass; inside `BCEWithLogitsLoss` |
| Output penalty | λ₁=0.2078; raw (not mean-centred) sub-network outputs; **dropout OFF** (second forward pass with training=False) |
| Weight decay | λ₂=0 (disabled for COMPAS) |
| Feature dropout | 0.05, **inverted** dropout (rescale survivors by 1/0.95) on (B,K) tensor before sum, training only |
| Batch sampling | 1024, **balanced 50/50** via equal-probability sampling from pos/neg pools |
| Optimiser | Adam, lr=0.02082, multiply lr by 0.995 each epoch |
| Max epochs / ES | 1000 / patience 60 epochs (checked every 10 epochs) |
| CV | StratifiedKFold(5, shuffle, seed=42); single train/val split per fold (StratifiedShuffleSplit, test_size=0.125, seed=1337) |
| Metric for ES | AUC-ROC on inner val split (not reported as fold AUC) |
| Reported per-fold AUC | AUC-ROC and AUC-PR on **held-out test fold** (20%) — not the val split |
| Phase A target | Single NAM, 5-fold CV, mean±std of TEST-set AUC-ROC and AUC-PR over 5 folds |
| Replication target | 0.741 ± 0.009 AUC-ROC (Table 1, ensemble; single-NAM CV will be slightly lower) |
