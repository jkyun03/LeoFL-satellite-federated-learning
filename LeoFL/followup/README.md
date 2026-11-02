# Follow-up study: capacity-aware local-step throttling as an offloading replacement

This package compares four ways of handling compute-speed heterogeneity ("stragglers") across
satellites in LH-FL — three that **never move data** (step-count throttling) and one that
reproduces the existing LH-FL's data-offloading mechanism **inside the same engine**, so the
"move data vs. throttle steps" comparison is controlled for everything except that one variable.
The actual existing LH-FL offloading scenario (`scenarios.OffloadScenario`, unmodified) is kept
as a fifth, supplementary reference — see "Why two different offloading comparisons" below.

This replaces an earlier version of this package that explored contact-timing-aware step
scheduling — that direction is retired; see "What changed and why" below.

**It does not modify any existing file.** New code lives entirely under `LeoFL/followup/`,
importing (never editing) shared assets from `LeoFL.Simulator`/`LeoFL.EuroSAT`. The existing
LH-FL run paths (`run_simulator.py`, `scenarios*.py`, `simulation_eurosat.py`) are untouched and
still runnable exactly as before.

## Research framing

The existing LH-FL mitigates stragglers by **moving data before training**:
`fl_core.offload_within_plane` redistributes a plane's samples proportional to capacity share,
`n_i' = (C_i / ΣC_j) × Σn_j`. That costs extra communication (moving samples between
satellites) and exposes a privacy risk (raw data leaving its original satellite).

This study asks: can the same straggler mitigation be achieved by **redistributing the local
*step budget* instead of the data**, keeping every sample on its original satellite?
`policies.capacity_proportional_steps` implements the direct analogue:
`step_i = round((C_i / ΣC_j) × S_total)`, with `S_total` held constant across methods so the
comparison is "same total compute, different distribution" vs. the baseline, not "more compute
wins." Unequal step counts are corrected at intra-orbit aggregation time with FedNova rather
than naive averaging.

## Why two different offloading comparisons

Running the *actual* `OffloadScenario` (`legacy_reference.py`) side by side with the new
step-throttling methods turned out to be confounded by more than just "data moved or not":

- `fl_core.train_one_epoch` always uses **Adam** (hardcoded) — the new methods use vanilla SGD
  (required for FedNova's normalization). No existing parameter anywhere in the call chain
  (`SimConfig`, `OffloadScenario`, `run_simulator.run_simulation`) can switch this without
  editing `fl_core.py`.
- `fl_core.fedavg_aggregate_fn` (legacy's intra-orbit aggregation) is a **plain, unweighted
  mean** across plane members — not sample-size weighted, unlike every method in this package
  (FedNova/weighted-avg, both weighted by `n_i`).
- The legacy tick-based time model (`sample_train_rounds_with_exp_jitter`, Laplace jitter noise,
  60s rounds) differs mechanically from this package's continuous-time model.

Fixing just the optimizer would still leave the other two confounds, so a literal
"legacy-but-SGD" patch would not actually produce a fair comparison. Instead, **Method
"offload"** reuses the existing, unmodified data-redistribution primitives
(`fl_core.offload_balance_datasets` / `offload_within_plane` — the *exact* math
`OffloadScenario` itself uses) but runs everything downstream of that (optimizer, time model,
aggregation weighting, barrier, mixing) through the *same* engine as "fixed"/"capacity"/
"capacity_minwork". This isolates "data moved once before training, vs. not" as the only
variable among these four methods. `legacy_reference.py` (Adam, unweighted mean, tick-based) is
kept as a supplementary reference showing what the actual, currently-used project code achieves
— useful context, but not mechanically comparable to the other four on a single-variable basis.

## What changed and why (vs. the earlier contact-timing version)

Calibration of the earlier (contact-timing-aware) version of this package found: (1) `lr=1e-3`
(inherited from the legacy Adam default) was far too small for the vanilla-SGD optimizer
FedNova requires here — an LR sweep (Method "fixed", toy constellation) found `lr=0.1` reached
67.9% vs. `lr=1e-3`'s 14.7% on the same short run; (2) at the real 220-satellite/5000 km
topology, most top-1 leader-neighbor edges turned out to be almost continuously in contact
range (leader selection inherently favors the most-frequently-met neighbor, which tends to be
a near-continuous one), so the contact-timing mechanism rarely had anything meaningful to
adapt to.

Given that, and the user's decision to sharpen the research question to "data movement vs.
step throttling," contact-window extraction, the discrete-event rendezvous engine, and the
contact-driven step policies were **removed** (not merely disabled) and replaced with a simple
**synchronous, whole-constellation-barrier** engine (`engine.py`) that reproduces
`OffloadScenario`'s actual cycle structure (`len(completed_this_cycle) == n_clients`) on a
continuous clock. `fednova.py`, `time_model.py`, `data_iterators.py`, `metrics.py`, and
`topology.py` are carried over unchanged — none of that logic was contact-timing-specific.

## Module map

| File | Responsibility |
|---|---|
| `time_model.py` | Step/epoch/batch definitions; per-satellite compute-speed draw (`epoch_minutes_i ~ Uniform{5..30}`, same recipe as legacy `per_sat_minutes`); `capacity_weights` (`C_i = 1/epoch_minutes_i`, same definition as `OffloadScenario`'s `capacity_i`) |
| `policies.py` | Step-count rules for all four methods; `AggregationPolicy` (FedNova / weighted-avg) dispatch |
| `fednova.py` | FedNova aggregation (vanilla-SGD/momentum=0 scope) + weighted-avg + inter-orbit convex mixing |
| `topology.py` | Constellation/leader/neighbor-graph construction (unchanged; reuses `fl_core.precompute_meetings_and_select_leaders` + `scenarios._build_topk_leader_plane_neighbors`) |
| `engine.py` | The synchronous whole-constellation-barrier cycle loop, incl. Method "offload"'s one-shot pre-training data redistribution (see below) |
| `data_iterators.py` | Per-satellite persistent shuffled iterator |
| `metrics.py` | Per-class recall/F1, time-to-target, accuracy-at-budget, slow-satellite subgroup metric |
| `logging_schema.py` | CSV schema + `RunLogs` (`followup_{method}_{seed}_{artifact}.csv`) |
| `config.py` | `ExperimentConfig` dataclass + JSON I/O |
| `run_experiment.py` | CLI entry point for the four engine-internal methods |
| `legacy_reference.py` | CLI entry point + `LegacyReferenceConfig` for the supplementary existing-offloading reference |
| `plotting.py` | Comparison plots (accuracy/time, step distribution, comm-volume across both CSV schemas, min-work fraction trade-off) |
| `configs/*.json` | One config per method + the legacy reference (section "Configuration") |
| `tests/*.py` | `unittest`-based validation tests |

## The four methods

All four share one engine; only the step-count policy (and, for "offload", a one-shot
pre-training step) differs. All four aggregate intra-orbit with FedNova — mathematically
identical to weighted-avg for "fixed"/"offload" since every satellite does the same number of
steps there.

| Method | Step count | Degenerates to / isolates |
|---|---|---|
| `fixed` | Same fixed value for every satellite, every cycle (baseline; no mitigation) | — |
| `capacity` | `round(S_total(plane) × C_i / ΣC_j)`, `S_total(plane) = fixed_steps × plane_size` | exactly `fixed` when every satellite in a plane has equal capacity (`tests/test_capacity_step_allocation.py`) |
| `capacity_minwork` | `capacity`'s allocation, raised to `minimum_steps_i = max(1, ceil(fraction × steps_per_epoch_i))` wherever it falls short | `capacity` when the floor never binds |
| `offload` | Same fixed value as `fixed`, **after** a one-shot `fl_core.offload_balance_datasets` call rebalances each plane's `N_i/C_i` | isolates "data moved once, same fixed step policy" vs. `fixed`'s "data untouched" — the fair "move data" comparison point |

`minimum_epoch_fraction` candidates to compare: 0.1, 0.25, 0.5, 1.0 (one config file each) — not
assumed to be good values, exactly as instructed; the floor's cost (more total compute when it
binds, not clawed back from faster satellites) is the straggler-mitigation-vs-data-quality
trade-off this study is designed to show (`plotting.plot_minwork_fraction_tradeoff`).

## Engine: one synchronous, whole-constellation-barrier cycle

Reproduces `OffloadScenario.run()`'s actual mechanics, on a continuous (float-seconds) clock
instead of a tick countdown:

0. **(Method "offload" only, once, before cycle 1):** compute `C_i = 1/epoch_minutes_i`
   (`time_model.draw_epoch_minutes`, independent of `n_i`, normalized to mean 1 — matching
   `OffloadScenario.__init__`'s own convention exactly), then call the existing, unmodified
   `fl_core.offload_balance_datasets(client_datasets, meeting_counts=topology.meeting_counts,
   plane_ids=topology.plane_ids, train_capacities=C)`, which mutates `client_datasets` in place.
   `steps_per_epoch_i`/`step_time_i` are then rebuilt from the new `n_i` (same seed, so
   `epoch_minutes_i` — a satellite's intrinsic speed — is unchanged; only how much data it now
   has to get through changes). The moved-sample byte cost
   (`offload_stats["offload_hops"] * calc_sample_bytes(...)`) is logged as a single
   `kind="data_offload"` row in `comm_log`, at `sim_time_s=0.0`.
1. Decide `step_i` for every satellite, per plane (the method's policy).
2. Train every satellite for `step_i` real vanilla-SGD steps (`torch.optim.SGD`, momentum=0 —
   required for FedNova's normalization; `fl_core.train_one_epoch`'s Adam is not reused).
3. `cycle_duration` = max over **every satellite in every plane** of (training time + intra-orbit
   relay to its leader), plus one shared intra-orbit aggregation cost, plus one shared
   inter-orbit transfer cost. This whole-constellation barrier (`len(completed_this_cycle) ==
   n_clients` in the legacy code) is intentionally unchanged from `OffloadScenario`.
4. Intra-orbit aggregate every plane (FedNova).
5. Inter-orbit convex-mix every plane with its static top-k neighbors
   (`fednova.convex_mix_states`), **unconditionally, every cycle** — same as `OffloadScenario`.
   No contact-timing gate (see "What changed and why").
6. Advance `sim_time`; checkpoint accuracy (throttled to `accuracy_checkpoint_interval_s`).

## Legacy reference (supplementary, not fairness-controlled)

`legacy_reference.py` builds the **same** client split and **same** `per_sat_minutes` draw as
`run_experiment.py`, then calls `run_simulator.run_simulation(mode="offload", ...)` — zero
changes to `scenarios.py`/`run_simulator.py`. `LegacyReferenceConfig.lr` (default `1e-3`, Adam)
is a *separate* field from `ExperimentConfig.lr` (`0.1`, SGD) — do not cross-apply them.
`ExperimentConfig.leader_select_max_dist_km` defaults to `2000.0` to match what
`run_simulator.run_simulation(mode="offload")` hardcodes for `OffloadScenario`'s leader
selection (not configurable through that function), so the engine-internal methods and this
reference at least pick the *same* leaders/neighbor graph. It writes the **existing** CSV schema
(`{prefix}_accuracy.csv`: `round, accuracy`; `{prefix}_comm.csv`: `time, model_bytes,
offload_bytes, model_hops, offload_hops, note`) under its own `log_dir` —
`plotting.plot_comm_volume_comparison` reads both schemas side by side.

## Configuration

`ExperimentConfig` (`config.py`): JSON files under `configs/` only need to override what differs
from the dataclass defaults. Key fields: `method` (`"fixed"|"capacity"|"capacity_minwork"|
"offload"`), `aggregation` (`"fednova"|"weighted_avg"`), `fixed_steps` (the step count for
"fixed"/"offload", and the per-plane budget basis for "capacity"/"capacity_minwork"),
`minimum_epoch_fraction` (Method "capacity_minwork"), `offload_ratio`/`offload_top_pairs_per_sat`
(Method "offload" — note: `offload_balance_datasets` only uses these in its "swap" mode; in the
"capacity" mode this engine calls, it always fully equalizes `N_i/C_i` within each plane
regardless of these two values, the same way `OffloadScenario` itself passes-but-doesn't-use
them in capacity mode — kept for call-signature parity, not because they currently do anything
here), `max_local_steps`, `leader_select_max_dist_km` (2000.0, matches the legacy reference),
`constellation.kind` (`"starlink_active10"` or `"toy_ring"` for smoke tests). `LegacyReferenceConfig`
(`legacy_reference.py`) is a separate, smaller dataclass with only the fields that scenario needs.

**Naming caution**: output files are named `followup_{method}_{seed}_{artifact}.csv`. Two runs
with the *same* method+seed but a *different* swept parameter (e.g. `minimum_epoch_fraction`)
will silently overwrite each other if pointed at the same `log_dir` — always use a distinct
`log_dir` per sweep point, exactly as the four `method_capacity_minwork_*.json` configs do.

## Smoke test (already run; toy constellation, 12 satellites / 3 planes)

```bash
# run from the repository root
python -m unittest discover -s LeoFL/followup/tests -v
python -m LeoFL.followup.run_experiment --config LeoFL/followup/configs/smoke.json --method fixed --seed 0
python -m LeoFL.followup.run_experiment --config LeoFL/followup/configs/smoke.json --method capacity --seed 0
python -m LeoFL.followup.run_experiment --config LeoFL/followup/configs/smoke.json --method capacity_minwork --seed 0
python -m LeoFL.followup.run_experiment --config LeoFL/followup/configs/smoke.json --method offload --seed 0
python -m LeoFL.followup.legacy_reference --config LeoFL/followup/configs/smoke_legacy.json
```

**Actually executed, results**: all 18 unit tests pass (`py_compile` clean on every file first,
including two new tests validating Method "offload" specifically). Smoke-scale results (toy
constellation, 3000s horizon, target 60%):

| Method | Time to 60% | Final accuracy |
|---|---|---|
| `fixed` (baseline) | 2971.6s | 76.8% |
| `capacity` | 1784.0s | 76.7% |
| `capacity_minwork` (fraction=0.25) | 1784.0s | **83.8%** |
| `offload` (fair, SGD) | **1578.6s** | 78.8% |
| legacy (Adam, unweighted, supplementary) | — (reached 85%+ by round 40) | 88.0% |

The accuracy-vs-time curve shows `fixed` plateauing clearly around 1500-2200s (the straggler
stall) while the other three avoid it — `offload` reaches the target fastest, with
`capacity`/`capacity_minwork` close behind despite moving zero data.

**Offload-balancing check** (the main thing this round needed to verify): comparing per-satellite
`N_i` against the same `epoch_minutes_i` used by `fixed`, offload reduced `N_i × epoch_minutes_i`
(proportional to `N_i/C_i`) from a roughly 3-4x within-plane spread down to **near-perfect
equalization** (e.g. one plane went from `{47007, 29463, 31194, 14652}` to
`{27270, 27300, 27270, 27276}`) — `offload_within_plane`'s actual intent, confirmed working
end-to-end through the new engine. The one-shot redistribution was logged as a single
`data_offload` comm row (26,728 bytes, 13 hops) at `sim_time_s=0`.

**Communication-volume nuance** (worth reporting honestly rather than forcing a clean story):
total bytes were `fixed`=7.49MB, `capacity`≈`capacity_minwork`=8.99MB, `offload`=9.01MB,
legacy=20.37MB. The *engine-internal* four methods use well under half the legacy reference's
total — but that gap is **not mainly about the one-shot offload cost** (26.7KB is tiny next to
the recurring per-cycle model-exchange traffic); it mostly reflects the legacy scenario's
different mixing/accounting pattern over its own 50 rounds. Within the four fairness-controlled
methods, `offload`'s one-shot data cost did not make it cheaper than `capacity`/`capacity_minwork`
in this short run — it actually ran slightly more cycles (converging faster) and so used
marginally *more* total bytes. The real headline here is "the whole engine-internal family uses
much less communication than the legacy implementation," not "offload costs more than
step-throttling because it moves data" — that distinction should hold up at full scale, but
needs re-checking there, not assumed from a 3000s smoke run.

Minor known cosmetic issue: in `plot_comm_volume_comparison`'s bar chart, the first (leftmost)
rotated x-tick label can slightly overlap the y-axis "0" tick at small figure sizes — functional,
not a correctness issue, not fixed in this pass.

## Main experiment (executed: 10 seeds, 0–9)

```bash
python -m LeoFL.followup.run_experiment --config LeoFL/followup/configs/method_fixed.json
python -m LeoFL.followup.run_experiment --config LeoFL/followup/configs/method_capacity.json
python -m LeoFL.followup.run_experiment --config LeoFL/followup/configs/method_capacity_minwork_0.1.json
python -m LeoFL.followup.run_experiment --config LeoFL/followup/configs/method_capacity_minwork_0.25.json
python -m LeoFL.followup.run_experiment --config LeoFL/followup/configs/method_capacity_minwork_0.5.json
python -m LeoFL.followup.run_experiment --config LeoFL/followup/configs/method_capacity_minwork_1.0.json
python -m LeoFL.followup.run_experiment --config LeoFL/followup/configs/method_offload.json
python -m LeoFL.followup.legacy_reference --config LeoFL/followup/configs/legacy_offload_reference.json
```
For multiple seeds, override `--seed`/`--log-dir` per run. The 10-seed logs (seeds 0–9) are in
`logs_followup/method_*/`; `python -m LeoFL.followup.make_report` turns them into the figures and
tables in [`report/`](report/), summarized in [`RESEARCH_SUMMARY.md`](RESEARCH_SUMMARY.md).

## Assumptions and limitations

1. **Capacity definition.** `C_i = 1/epoch_minutes_i`, exactly `OffloadScenario`'s own
   `capacity_i` — not separately disentangled from local data volume (same simplification the
   legacy code already makes).
2. **Step-time model is a new assumption**, not measured: `step_time_i =
   epoch_minutes_i*60/steps_per_epoch_i`.
3. **Communication timing is a simplified point model**: dedicated bandwidth per ISL link, no
   cross-link contention, one shared (not per-edge) transfer-time addend per cycle.
4. **Whole-constellation barrier is deliberate** — it is what makes "fixed" reproduce the exact
   problem offloading was built to solve.
5. **`legacy_reference.py` is supplementary, not fairness-controlled** against the other four
   (different optimizer, different aggregation weighting, different time model — see "Why two
   different offloading comparisons"). Method "offload" is the fair "moves data" comparison.
6. **`offload_ratio`/`offload_top_pairs_per_sat` are currently inert** in the capacity-mode
   offload path this engine uses (see "Configuration") — kept for signature parity with
   `OffloadScenario`'s own call, not because they do anything here.
7. **`fixed_steps`, `minimum_epoch_fraction` defaults, and the communication-timing constants
   are starting points, not tuned/validated values.**
8. **`lr=0.1`** is empirically the best of five values tested (0.001/0.01/0.03/0.1/0.3) on a
   toy-scale, Method-"fixed"-only sweep — not re-validated at full 220-satellite scale.

## What has and has not been executed

**Executed and passing**: `py_compile` on every file; all 18 unit tests
(`python -m unittest discover -s LeoFL/followup/tests -v`), including
`test_offload_balances_load.py` (verifies the within-plane `N_i/C_i` equalization and the
one-shot comm log entry) and `test_offload_...fixed_step_policy` (verifies offload trains with
the same fixed-step policy as Method "fixed"); a smoke-scale run of all four engine-internal
methods plus the legacy reference (toy constellation); comparison plots generated and visually
inspected across all five run types.

**Also executed (later)**: the full 220-satellite main experiment for all methods over 10 seeds
(0–9); logs in `logs_followup/method_*/`, report in [`report/`](report/).
