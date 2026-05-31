---
name: cluster-run-workflow
description: How to run code for this project — never run python/GPU locally; hand the user an sbatch
metadata:
  type: feedback
---

The user runs everything on a SLURM cluster; the local Windows machine has no
project venv/GPU and cannot execute the code.

**Why:** Local `python ...` calls fail (no env) and waste a turn; the real
transformers/peft/torch stack only exists on the cluster.

**How to apply:** Do NOT run `python`, import the project, or check package
versions locally. When something must be executed (especially GPU work), write a
`.sbatch` file under `sbatch/` and ask the user to submit it and paste the
output. Reason about code statically (Read/Grep) instead of running it. See
[[symmpo-base-eval-confounds]] for the run config (7B fp16 → gpu:1).
