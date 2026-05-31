#!/usr/bin/env python3
"""Rich before/after comparison: base llava_1.5 vs llava_symDPO.

Goes beyond HEAS to characterise *how* the model changed. Reuses cached base
results (llava_1.5_*) and freshly-run symDPO results (llava_symDPO_*) in
results/full.

IMPORTANT — base schema reality (verified May 2026):
  The cached llava_1.5 result JSONs were written by an older code path and have
  ``raw == null``: no control responses, no raw text. Consequences, handled
  honestly here rather than papered over:
    * Control-fail rate has NO "before" — it exists only for symDPO. Base shows
      NaN for control columns, and we say so.
    * HEAS gating needs control; with base ungatable, the only apples-to-apples
      alignment metric is an UNGATED illusory rate / HEAS computed the same way
      for both models. symDPO's properly control-gated HEAS is reported
      separately as an after-only diagnostic.
    * Base has no raw reply text, so the chosen-letter collapse view is
      symDPO-only.

To keep every comparison fair the script first restricts BOTH models to their
shared set of stimulus_ids. This also makes the comparison valid when symDPO was
only partially evaluated (e.g. a time-limited run): we compare on exactly the
stimuli both models actually saw.

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


def _normalize_base(r: dict) -> dict:
    """Align a cached base record to the CURRENT label convention.

    The Müller-Lyer illusory/other tagging was fixed AFTER the base run, so base
    ``muller_lyer`` records have ``illusory`` and ``other`` swapped relative to
    freshly-run symDPO. Swap them back (correct is unchanged) so the before/after
    compares like-for-like. Only muller_lyer is affected.
    """
    if r.get("illusion_type") != "muller_lyer":
        return r
    r = dict(r)
    r["illusory"], r["other"] = float(r.get("other", 0.0)), float(r.get("illusory", 0.0))
    pl = r.get("predicted_label")
    if pl == "illusory":
        r["predicted_label"] = "other"
    elif pl == "other":
        r["predicted_label"] = "illusory"
    return r


def _is_soft(results: list[dict]) -> bool:
    """True if any score is strictly fractional — a fingerprint of the older
    multi-trial probe_pair (averaged over orderings × framings). The current
    single-trial code yields hard {0,1} scores."""
    for r in results:
        for k in LABELS:
            v = r.get(k)
            if v is not None and 0.0 < float(v) < 1.0:
                return True
    return False


def _chosen_letter(r: dict) -> str:
    raw = r.get("raw") or {}
    resps = raw.get("raw_responses") or []
    if not resps:
        return "?"
    return (resps[0].strip().upper()[:1] or "?")


def _ctrl_correct(r: dict) -> bool | None:
    """True/False if the model answered the CONTROL image correctly, else None
    when no control was logged (the case for all cached base results)."""
    raw = r.get("raw") or {}
    if "ctrl_argmax_correct" in raw:
        return bool(raw["ctrl_argmax_correct"])
    cp = raw.get("probs_control")
    if cp is None:
        return None
    return int(cp.index(max(cp))) == 0


def _slice_metrics(results: list[dict], human_rate: float) -> dict:
    """Ungated metrics that are comparable across base and symDPO.

    HEAS here is UNGATED (1 - |p_illusory - human|) so base and symDPO are
    treated identically; base cannot be control-gated. control_FAIL is NaN
    whenever control was not logged (always, for base)."""
    diag = output_diagnostics(results)
    ctrl_vals = [c for c in (_ctrl_correct(r) for r in results) if c is not None]
    ctrl_succ = (sum(ctrl_vals) / len(ctrl_vals)) if ctrl_vals else float("nan")
    p_ill = diag["p_pred_illusory"]
    heas_ungated = (1.0 - abs(p_ill - human_rate)) if diag["n"] else float("nan")
    return {
        "n": diag["n"],
        "n_ctrl_logged": len(ctrl_vals),
        "control_FAIL": round(1 - ctrl_succ, 4) if ctrl_succ == ctrl_succ else float("nan"),
        "heas_ungated": round(heas_ungated, 4) if heas_ungated == heas_ungated else float("nan"),
        "p_correct": round(diag["p_pred_correct"], 4),
        "p_illusory": round(p_ill, 4),
        "p_other": round(diag["p_pred_other"], 4),
        "mean_prob_illusory": round(diag["mean_prob_illusory"], 4),
    }


def _plot_comparison(side: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar charts: illusory rate (both) and symDPO control-FAIL."""
    import matplotlib.pyplot as plt

    cats = side["category"].tolist()
    x = np.arange(len(cats))
    w = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, 1.5 * len(cats)), 4.5))

    # Illusory-response rate — comparable before/after.
    ax1.bar(x - w / 2, side["base_p_illusory"], w, label="base", color="#4C72B0")
    ax1.bar(x + w / 2, side["dpo_p_illusory"], w, label="symDPO", color="#C44E52")
    ax1.set_ylim(0, 1)
    ax1.set_title("Illusory-response rate (base vs symDPO)\np_pred == 'illusory'")
    ax1.set_ylabel("fraction")
    ax1.set_xticks(x); ax1.set_xticklabels(cats, rotation=45, ha="right")
    ax1.legend()

    # Control-FAIL — symDPO only (base never logged control).
    ax2.bar(x, side["dpo_control_FAIL"], w * 1.3, color="#C44E52", label="symDPO")
    ax2.set_ylim(0, 1)
    ax2.set_title("Control-FAIL rate (symDPO only)\nbase control not logged in cache")
    ax2.set_ylabel("fraction")
    ax2.set_xticks(x); ax2.set_xticklabels(cats, rotation=45, ha="right")
    ax2.legend()

    fig.suptitle("LLaVA-1.5: base vs SymMPO (matched stimuli)", fontsize=13)
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
    print(f"Loaded base={len(base_all)} symDPO={len(dpo_all)} results.")
    if not base_all or not dpo_all:
        print("ERROR: missing cached results for one of the models.")
        sys.exit(1)

    # Correct the post-base Müller-Lyer illusory/other label swap on base records.
    n_muller = sum(1 for r in base_all if r.get("illusion_type") == "muller_lyer")
    base_all = [_normalize_base(r) for r in base_all]
    if n_muller:
        print(f"Applied muller_lyer illusory<->other correction to {n_muller} base records.")

    # Detect the soft (multi-trial) vs hard (single-trial) estimator mismatch.
    base_soft, dpo_soft = _is_soft(base_all), _is_soft(dpo_all)
    if base_soft != dpo_soft:
        print(f"WARNING: base soft(multi-trial)={base_soft}, symDPO soft={dpo_soft}. "
              "mean_prob_illusory is NOT directly comparable across them; "
              "rely on predicted_label-based metrics (p_illusory, flips, confusion).")

    # ── Restrict BOTH models to their shared stimulus_ids ───────────────
    base_by_id = {r["stimulus_id"]: r for r in base_all}
    dpo_by_id = {r["stimulus_id"]: r for r in dpo_all}
    shared = sorted(set(base_by_id) & set(dpo_by_id))
    print(f"Shared stimulus_ids: {len(shared)}  "
          f"(base-only {len(base_by_id) - len(shared)}, "
          f"symDPO-only {len(dpo_by_id) - len(shared)})")
    if not shared:
        print("ERROR: base and symDPO share no stimulus_ids — nothing to compare.")
        sys.exit(1)
    base_all = [base_by_id[s] for s in shared]
    dpo_all = [dpo_by_id[s] for s in shared]

    base_df = pd.DataFrame(base_all)
    dpo_df = pd.DataFrame(dpo_all)
    categories = sorted(set(base_df["category"]) & set(dpo_df["category"]))

    # ── 1. Per-category side-by-side (ungated, comparable) ──────────────
    rows = []
    for cat in categories + ["ALL"]:
        if cat == "ALL":
            b, d = base_all, dpo_all
            hr = float(pd.Series([human.get(c, 0.5) for c in base_df["category"]]).mean())
        else:
            b = [r for r in base_all if r["category"] == cat]
            d = [r for r in dpo_all if r["category"] == cat]
            hr = human.get(cat, 0.5)
        mb = _slice_metrics(b, hr)
        md = _slice_metrics(d, hr)
        row = {"category": cat, "human_illusory": round(hr, 3)}
        for k in mb:
            row[f"base_{k}"] = mb[k]
            row[f"dpo_{k}"] = md[k]
        rows.append(row)
    side = pd.DataFrame(rows)
    side.to_csv(out_dir / "side_by_side_metrics.csv", index=False)
    _plot_comparison(side, out_dir / "figures" / "base_vs_symMPO.png")

    # ── 1b. symDPO-only properly control-gated HEAS (after diagnostic) ──
    gated_rows = []
    for cat in categories:
        d = [r for r in dpo_all if r["category"] == cat]
        res = compute_heas(d, human.get(cat, 0.5), ctrl_thr, category=cat)
        gated_rows.append({
            "category": cat,
            "symDPO_heas_gated": round(res["heas"], 4) if res["heas"] == res["heas"] else float("nan"),
            "n_included": res["n_stimuli"],
            "n_excluded_ctrl_fail": res["n_excluded"],
        })
    gated_df = pd.DataFrame(gated_rows)
    gated_df.to_csv(out_dir / "symDPO_gated_heas.csv", index=False)

    # ── 2. Paired analysis (matched stimulus_id) ────────────────────────
    confusion = defaultdict(int)
    n_flip = 0
    for s in shared:
        bl = base_by_id[s]["predicted_label"]
        dl = dpo_by_id[s]["predicted_label"]
        confusion[(bl, dl)] += 1
        n_flip += (bl != dl)
    flip_rate = n_flip / len(shared) if shared else float("nan")
    conf_df = pd.DataFrame(
        [[confusion.get((b, d), 0) for d in LABELS] for b in LABELS],
        index=[f"base_{l}" for l in LABELS],
        columns=[f"dpo_{l}" for l in LABELS],
    )
    conf_df.to_csv(out_dir / "label_confusion.csv")

    # ── 3. Chosen-letter distribution (symDPO only; base has no text) ───
    base_letters = Counter(_chosen_letter(r) for r in base_all)
    dpo_letters = Counter(_chosen_letter(r) for r in dpo_all)

    # ── REPORT ──────────────────────────────────────────────────────────
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print("\n" + "=" * 72)
    print("NOTE: base (before) has no control logs and no raw text in cache.")
    print("  - control_FAIL: symDPO only; base shows NaN (not recorded).")
    print("  - HEAS in the side-by-side is UNGATED for both (only fair option).")
    print("  - symDPO's control-gated HEAS is in symDPO_gated_heas.csv.")
    print("  - base muller_lyer illusory<->other corrected (swap fixed post-base).")
    print(f"  - base multi-trial(soft)={base_soft}, symDPO soft={dpo_soft} "
          "(if differing, trust predicted_label metrics over mean_prob_illusory).")
    print("=" * 72)
    print("\nSIDE-BY-SIDE METRICS (matched stimuli, ungated):")
    print(side.to_string(index=False))

    print("\nsymDPO control-gated HEAS (after-only diagnostic):")
    print(gated_df.to_string(index=False))

    print("\n" + "=" * 72)
    print(f"PAIRED ANALYSIS over {len(shared)} matched stimuli")
    print("=" * 72)
    print(f"label flip rate (answer changed): {flip_rate:.3f}  ({n_flip}/{len(shared)})")
    print("\nLabel confusion  (rows=base, cols=symDPO):")
    print(conf_df.to_string())

    print("\nChosen-letter distribution (symDPO only; base text not logged):")
    print(f"  base  : {dict(base_letters.most_common())}")
    print(f"  symDPO: {dict(dpo_letters.most_common())}")

    print("\n" + "=" * 72)
    print(f"CSVs + figure written to {out_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
