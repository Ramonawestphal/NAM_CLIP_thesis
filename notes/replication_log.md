# NAM COMPAS Replication — Bug and Decision Log

Entries are appended chronologically. Each entry records a discrepancy, bug,
or non-obvious decision that future-me (or a reviewer) would otherwise have to
reconstruct from the git history.

---

## 2026-05-14 — Encoding column order and OHE scaling

**Bug caught at unit-test stage (tests 9 and 10).**

### What was wrong

- **Column order:** The initial `CompasEncoder` placed OHE columns at indices 0–7 and
  continuous columns at indices 8–11.
- **OHE scaling:** The MinMaxScaler was applied to all 12 columns jointly (including the
  OHE indicators), rescaling them from `{0, 1}` to roughly `{−1, +1}`.

### Fix applied (`src/data/encoding.py`)

- Continuous columns (`age`, `charge_degree`, `length_of_stay`, `priors_count`) moved to
  indices **0–3**; MinMaxScaler to `(−1, 1)` applied to these four columns only.
- Race OHE (6 columns) at indices **4–9**; sex OHE (2 columns) at indices **10–11**.
  Both remain as exact `{0, 1}` binary indicators — no rescaling.

### Why it matters

Under `{−1, +1}`-encoded OHE, `f_k(−1)` becomes a free parameter entangled with the
global bias β. This hurts identifiability of categorical shape functions and breaks the
Figure 4 "read off `f_race(i)` at the active slot" plotting convention described in §2a of
the implementation contract. Under `{0, 1}` encoding, `f_k(0) ≈ 0` is structurally
enforced (the final FeatureNN layer has `bias=False`), so `f_k(1)` cleanly represents the
per-category contribution.

### Tests that caught this

- `tests/test_nam_unit.py::test_encoder_column_layout` (test 9)
- `tests/test_nam_unit.py::test_calc_outputs_no_feature_dropout_no_bias` (test 10)

### Contract updated

`docs/nam_implementation_contract.md` amended in §2 (feature table + scaling paragraph),
§2a (active input changed from +1 to 1; inactive-sum formula updated), and summary table
("all cols scaled to [−1,1]" corrected to "continuous [0–3] in [−1,1], OHE [4–11] in {0,1}").
