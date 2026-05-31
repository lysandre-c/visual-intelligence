---
name: symmpo-eval-no-load-bug
description: SymMPO LLaVA RL — there is NO eval/load/merge bug; real issues are control-gate exclusion + two training bugs
metadata:
  type: project
---

Investigated (2026-05-31) why the SymMPO-RL'd LLaVA-1.5 ("llava_symDPO", adapter at
`results/rl_alignment/final`) appeared to give identical eval outputs to base.

**Conclusion: there is NO loading/merge/eval bug.** Ruled out across clean tests:
- adapter trained fine (lora_B max norm 2.83, all 224 nonzero; projector present)
- target_modules regex matches at load (224 LoRA modules injected)
- transformers 5.9.0 / peft 0.19.1; loads fp16 even with no torch_dtype
- `merge_and_unload()` works even sharded across 2 GPUs (weight delta 0.075, logits shift, outputs differ)
- end-to-end `VLMProtocol`→`compute_heas` shows symDPO ≠ base at label and HEAS level

Could NOT reproduce the originally-reported "identical HEAS"; `heas_comparison.csv`
no longer exists. Most likely a stale/old results file — unconfirmed, do not assert.

**Real findings:**
1. `configs/experiments.yaml` `control_ceiling_threshold = -1` (argmax mode). symDPO answers
   the CONTROL images non-veridically, so `compute_heas` gate excludes nearly all stimuli
   (muller n=0→NaN; simultaneous_contrast n=1→0.92). symDPO HEAS is statistically
   uninterpretable. The NaN is the honest result — do NOT weaken the gate (p-hacking).
2. Train/eval surface mismatch: collator trains with symbol options ♣♠♦ (`SYMBOL_MAP`,
   `symbol_demo=True`) but eval prompts use A/B/C.
3. CONFIRMED bug in `src/rl/collator.py`: `__call__` calls `_prepare_text` separately per
   prefix, and `_prepare_text` consumes the shared stateful `self._rng` (symbol shuffle +
   option shuffle). So `orig_chosen` and `orig_rejected` get DIFFERENT prompts though they
   share `original_prompt`. Violates DPO's identical-prompt requirement (`loss.py` docstring
   assumes constant conditioning) → corrupted preference gradient.

Diagnostic scripts left in repo: `experiments/diagnose_adapter.py`,
`diagnose_eval_merge.py`, `diagnose_eval_real.py`, `diagnose_heas_chain.py`;
`sbatch/run_diagnose_merge.sbatch`. User fixed a Müller-Lyer illusory/other label swap
(re-run muller before quoting its split).
