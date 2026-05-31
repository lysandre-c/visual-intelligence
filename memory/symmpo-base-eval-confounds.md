---
name: symmpo-base-eval-confounds
description: Cached llava_1.5 base eval results have a leaner old schema + two confounds that any before/after vs symDPO must correct
metadata:
  type: project
---

The cached base `results/full/llava_1.5_*_results.json` were written by an OLDER
code path than current symDPO runs. Verified 2026-05-31. Any before/after
comparison must account for this:

1. **`raw == null`** in base result JSONs — NO control responses, NO raw text.
   So control-fail has no "before" (symDPO only), and base cannot be
   control-gated. `audit/llava_1.5/*` is also from the old path (empty
   `raw_responses`, no control). Headline "control fails before vs after" is
   only computable for the after side.
2. **HEAS gating degenerates for base:** `compute_heas` argmax mode (threshold
   `-1` in `configs/experiments.yaml`) falls back to `correct >= illusory` when
   control is missing, which excludes every illusory prediction → base
   `p_illusory→0`, `HEAS→1−human`. Meaningless vs symDPO's real gate. The only
   fair before/after alignment metric is an UNGATED illusory rate / HEAS applied
   identically to both.
3. **Müller-Lyer illusory/other swap was fixed AFTER the base run** (user
   confirmed). So base `muller_lyer` records have illusory/other swapped vs
   symDPO. Must swap base muller illusory↔other (correct unchanged) before
   comparing.
4. **Soft vs hard estimator:** `raw==null` implies base predates the
   single-trial refactor (commit `c7602fb "Optimize VLM evaluation speed"`,
   which added "Randomly sample 1 ordering"). Base is likely MULTI-trial (soft
   fractional correct/illusory/other averaged over orderings×framings) while
   symDPO is single-trial hard {0,1}. `mean_prob_illusory` is then not directly
   comparable — trust `predicted_label`-based metrics.

**THE FIX (cheap): re-EVAL base, don't accept the broken cache.** At the
measured 0.77 s/it the full 4440-stimulus base scan is ~1h on 1 GPU
(`sbatch/rerun_base_eval.sbatch`, backs up old base then re-runs full_eval
--models llava_1.5). A fresh base gains control logs + single-trial + corrected
muller, fixing all 4 confounds and enabling a proper control-GATED before/after
plus control-fail before-vs-after.

`experiments/compare_sym_mpo.py` is ADAPTIVE: restricts to shared `stimulus_id`s;
if base has control logs it GATES both models identically (valid before/after),
else falls back to ungated + muller swap (`_normalize_base`, skipped for fresh
base) + soft/hard warning (`_is_soft`). Ungated HEAS is a TRAP (rewards
indiscriminate illusory). See [[symmpo-eval-no-load-bug]].

Run config: 7B fp16 → use `gpu:1` (single-stream faster than 2-GPU split).
`vlm.py` `_query` uses `max_new_tokens=8` (only first char parsed; greedy →
invariant vs base@150). `full_eval.py --max-per-type N` caps stimuli/type for
time-boxed runs; `experiments/bench_symdpo.py` measures s/it and writes a
recommended cap. `sbatch/run_eval_sym_mpo.sbatch` is self-tuning + self-protecting
(benchmark → cap → time-boxed eval → always-runs compare → expendable GradCAM).
