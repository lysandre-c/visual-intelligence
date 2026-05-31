#!/usr/bin/env python3
"""Rich before/after comparison: base llava_1.5 vs llava_symDPO.

Goes beyond HEAS to characterise *how* the model changed — including cases
where HEAS is similar but behaviour differs. Reuses cached base results
(llava_1.5_*) and freshly-run symDPO results (llava_symDPO_*) in results/full.

Metrics (per category + overall):
  - control success / FAIL rate              (does the model still do the task?)
  - HEAS + n_included / n_excluded           (how much data the gate keeps)
  - illusion-image label distribution        (correct / illusory / other)
  - mean soft probabilities
Paired metrics (matched by stimulus_id):
  - label flip rate                          (how often the answer changed)
  - base->symDPO label confusion matrix      (which way it moved)
  - control pass -> fail transitions         (regressions on the easy control)
  - chosen-letter distribution               (detects positional answer collapse)

Usage:
    python experiments/compare_sym_mpo.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.diagnostics import output_diagnostics
from src.metrics.heas import compute_heas
from src.analysis.plots import save_figure

BASE = "llava_1.5"
DPO = "llava_symDPO"
LABELS = ("correct", "illusory", "other")


def _load(results_dir: Path, model: str) -> list[dict]:
    out = []
    for p in sorted(results_dir.glob(f"{model}_*_results.json")):
        out.extend(json.loads(p.read_text()))
    return out


def _chosen_letter(r: dict) -> str:
    raw = r.get("raw") or {}
    resps = raw.get("raw_responses") or []
    if not resps:
        return "?"
    return (resps[0].strip().upper()[:1] or "?")


def _ctrl_correct(r: dict) -> bool | None:
    raw = r.get("raw") or {}
    if "ctrl_argmax_correct" in raw:
        return bool(raw["ctrl_argmax_correct"])
    cp = raw.get("probs_control")
    if cp is None:
        return None
    return int(cp.index(max(cp))) == 0


def _slice_metrics(results: list[dict], human_rate: float, ctrl_thr: float, category: str) -> dict:
    diag = output_diagnostics(results)
    heas = compute_heas(results, human_rate, ctrl_thr, category=category)
    ctrl_vals = [c for c in (_ctrl_correct(r) for r in results) if c is not None]
    ctrl_succ = (sum(ctrl_vals) / len(ctrl_vals)) if ctrl_vals else float("nan")
    return {
        "n": diag["n"],
        "control_success": round(ctrl_succ, 4) if ctrl_succ == ctrl_succ else ctrl_succ,
        "control_FAIL": round(1 - ctrl_succ, 4) if ctrl_succ == ctrl_succ else ctrl_succ,
        "heas": heas["heas"],
        "heas_n_incl": heas["n_stimuli"],
        "heas_n_excl": heas["n_excluded"],
        "p_correct": round(diag["p_pred_correct"], 4),
        "p_illusory": round(diag["p_pred_illusory"], 4),
        "p_other": round(diag["p_pred_other"], 4),
        "mean_prob_illusory": round(diag["mean_prob_illusory"], 4),
    }


def _plot_comparison(side: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar charts: control-FAIL rate and illusory rate, base vs symDPO."""
    import matplotlib.pyplot as plt

    cats = side["category"].tolist()
    x = np.arange(len(cats))
    w = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, 1.5 * len(cats)), 4.5))

    # Control-FAIL
    ax1.bar(x - w / 2, side["base_control_FAIL"], w, label="base", color="#4C72B0")
    ax1.bar(x + w / 2, side["dpo_control_FAIL"], w, label="symDPO", color="#C44E52")
    ax1.set_ylim(0, 1)
    ax1.set_title("Control FAIL rate\n(lower = still does the task)")
    ax1.set_ylabel("fraction")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cats, rotation=45, ha="right")
    ax1.legend()

    # Illusory-response rate
    ax2.bar(x - w / 2, side["base_p_illusory"], w, label="base", color="#4C72B0")
    ax2.bar(x + w / 2, side["dpo_p_illusory"], w, label="symDPO", color="#C44E52")
    ax2.set_ylim(0, 1)
    ax2.set_title("Illusory-response rate\n(p_pred == 'illusory')")
    ax2.set_ylabel("fraction")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cats, rotation=45, ha="right")
    ax2.legend()

    fig.suptitle("LLaVA-1.5: base vs SymMPO", fontsize=13)
    fig.tight_layout()
    save_figure(fig, out_path)


def main() -> None:
    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "experiments.yaml").read_text())
    human = cfg["human_baselines"]
    ctrl_thr = cfg["control_ceiling_threshold"]
    results_dir = PROJECT_ROOT / cfg["full_eval"]["output_dir"]
    out_dir = results_dir / "compare_sym_mpo"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_all = _load(results_dir, BASE)
    dpo_all = _load(results_dir, DPO)
    print(f"Loaded base={len(base_all)} symDPO={len(dpo_all)} results "
          f"(control_ceiling_threshold={ctrl_thr})")
    if not base_all or not dpo_all:
        print("ERROR: missing cached results for one of the models.")
        sys.exit(1)

    base_df = pd.DataFrame(base_all)
    dpo_df = pd.DataFrame(dpo_all)
    categories = sorted(set(base_df["category"]) & set(dpo_df["category"]))

    # ── 1. Per-category side-by-side metrics ────────────────────────────
    rows = []
    for cat in categories + ["ALL"]:
        if cat == "ALL":
            b, d = base_all, dpo_all
            hr = float(pd.Series([human.get(c, 0.5) for c in base_df["category"]]).mean())
        else:
            b = [r for r in base_all if r["category"] == cat]
            d = [r for r in dpo_all if r["category"] == cat]
            hr = human.get(cat, 0.5)
        mb = _slice_metrics(b, hr, ctrl_thr, cat)
        md = _slice_metrics(d, hr, ctrl_thr, cat)
        row = {"category": cat}
        for k in mb:
            row[f"base_{k}"] = mb[k]
            row[f"dpo_{k}"] = md[k]
        rows.append(row)
    side = pd.DataFrame(rows)
    side.to_csv(out_dir / "side_by_side_metrics.csv", index=False)
    _plot_comparison(side, out_dir / "figures" / "base_vs_symMPO.png")

    # ── 2. Paired analysis (matched stimulus_id) ────────────────────────
    base_by_id = {r["stimulus_id"]: r for r in base_all}
    dpo_by_id = {r["stimulus_id"]: r for r in dpo_all}
    shared = [sid for sid in base_by_id if sid in dpo_by_id]

    confusion = defaultdict(int)         # (base_label, dpo_label) -> count
    ctrl_transition = defaultdict(int)   # (base_pass, dpo_pass) -> count
    n_flip = 0
    for sid in shared:
        bl = base_by_id[sid]["predicted_label"]
        dl = dpo_by_id[sid]["predicted_label"]
        confusion[(bl, dl)] += 1
        n_flip += (bl != dl)
        cb, cd = _ctrl_correct(base_by_id[sid]), _ctrl_correct(dpo_by_id[sid])
        if cb is not None and cd is not None:
            ctrl_transition[(cb, cd)] += 1

    flip_rate = n_flip / len(shared) if shared else float("nan")

    conf_df = pd.DataFrame(
        [[confusion.get((b, d), 0) for d in LABELS] for b in LABELS],
        index=[f"base_{l}" for l in LABELS],
        columns=[f"dpo_{l}" for l in LABELS],
    )
    conf_df.to_csv(out_dir / "label_confusion.csv")

    # ── 3. Chosen-letter distribution (collapse detector) ───────────────
    base_letters = Counter(_chosen_letter(r) for r in base_all)
    dpo_letters = Counter(_chosen_letter(r) for r in dpo_all)

    # ── REPORT ──────────────────────────────────────────────────────────
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print("\n" + "=" * 72)
    print("SIDE-BY-SIDE METRICS (base vs symDPO)")
    print("=" * 72)
    print(side.to_string(index=False))

    print("\n" + "=" * 72)
    print(f"PAIRED ANALYSIS over {len(shared)} matched stimuli")
    print("=" * 72)
    print(f"label flip rate (answer changed): {flip_rate:.3f}  ({n_flip}/{len(shared)})")
    print("\nLabel confusion  (rows=base, cols=symDPO):")
    print(conf_df.to_string())
    print("\nControl pass->fail transitions  (True=answered control correctly):")
    for (cb, cd), n in sorted(ctrl_transition.items()):
        tag = "  <-- REGRESSION" if (cb and not cd) else ("  <-- recovery" if (not cb and cd) else "")
        print(f"  base={cb!s:5} -> symDPO={cd!s:5}: {n}{tag}")

    print("\nChosen-letter distribution (detects positional collapse):")
    print(f"  base  : {dict(base_letters.most_common())}")
    print(f"  symDPO: {dict(dpo_letters.most_common())}")

    print("\n" + "=" * 72)
    print(f"CSVs written to {out_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
