"""Quick test: does last_epoch correctly fast-forward the cosine LR schedule?"""
import torch
from transformers import get_cosine_schedule_with_warmup

max_steps = 1000
warmup = int(max_steps * 0.05)  # 50
base_lr = 5e-5
resume_step = 800

# ── Original run: step through 0→1000 ──────────────────────────────────
dummy = torch.nn.Linear(1, 1)
opt_orig = torch.optim.AdamW(dummy.parameters(), lr=base_lr)
sched_orig = get_cosine_schedule_with_warmup(
    opt_orig, num_warmup_steps=warmup, num_training_steps=max_steps
)

original_lrs = {}
for step in range(1, max_steps + 1):
    opt_orig.step()
    sched_orig.step()
    original_lrs[step] = sched_orig.get_last_lr()[0]

# ── Resumed run: create scheduler with last_epoch=resume_step-1 ────────
dummy2 = torch.nn.Linear(1, 1)
opt_resume = torch.optim.AdamW(dummy2.parameters(), lr=base_lr)

# PyTorch requires initial_lr when last_epoch != -1
for group in opt_resume.param_groups:
    group['initial_lr'] = group['lr']

sched_resume = get_cosine_schedule_with_warmup(
    opt_resume, num_warmup_steps=warmup, num_training_steps=max_steps,
    last_epoch=resume_step - 1,
)

resumed_lrs = {}
for step in range(resume_step + 1, max_steps + 1):
    opt_resume.step()
    sched_resume.step()
    resumed_lrs[step] = sched_resume.get_last_lr()[0]

# ── Compare ────────────────────────────────────────────────────────────
print(f"{'Step':>6}  {'Original LR':>14}  {'Resumed LR':>14}  {'Match':>6}")
print("-" * 50)
all_match = True
for step in range(resume_step + 1, max_steps + 1):
    orig = original_lrs[step]
    resu = resumed_lrs[step]
    match = abs(orig - resu) < 1e-12
    all_match = all_match and match
    if step % 10 == 0 or not match:  # print every 10th step + any mismatches
        print(f"{step:>6}  {orig:>14.8e}  {resu:>14.8e}  {'✓' if match else '✗':>6}")

print()
if all_match:
    print("✅ ALL learning rates match exactly for steps 801→1000.")
else:
    print("❌ MISMATCH detected — the approach is WRONG.")

# Also print a few key steps from original for reference
print(f"\nOriginal LR at step 800: {original_lrs[800]:.6e}")
print(f"Original LR at step 860: {original_lrs[860]:.6e}")
print(f"Original LR at step 1000: {original_lrs[1000]:.6e}")
