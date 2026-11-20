#!/usr/bin/env python3
"""
T4 — reproduction of the paper's surface-target filter cascade.

WHAT THE PAPER DID
------------------
The reference paper does not train a model.  Its method is a filter cascade:
a starting set of annotated cell-surface genes, narrowed by four successive
criteria, ending in a short list that includes MPZL1.  The reported stage
counts are 1,492 -> 90 -> 20 -> 19 -> 6.

That matters for this project twice over.  It is why the checkpoint's question
about whether the paper uses a trained predictor has a clear answer: it does
not.  And it is why the cascade is worth reproducing separately from the
model -- the two do different jobs, and comparing them is the point.

WHAT THIS SCRIPT DOES
---------------------
Runs the same cascade over the LUAD feature table and reports, at every
stage, how many genes survive against how many the paper reported, and
whether MPZL1 is still in.

Two modes, and the difference between them is itself a result:

  stated      thresholds set to the paper's criteria as published.  Where the
              paper names a criterion but not its cut point, the cut point is
              marked INFERRED below and is a reconstruction, not a quotation.
  calibrated  thresholds moved automatically until each stage matches the
              paper's reported count.  This answers a different question --
              not "do our numbers match" but "given counts that match by
              construction, are they the same genes?"

Running both is deliberate.  If stated mode misses the paper's counts and
calibrated mode recovers them with different cut points, the gap is a finding
about the paper's reporting, not a failure of this reproduction.

DEPENDENCY IS NOT A FILTER HERE
-------------------------------
The cascade never filters on how much the tumor needs the gene, and neither
does this script.  A CAR-T cell kills whatever displays the antigen, so a
gene the tumor can live without is still a valid target -- MPZL1 is in the
paper's short list despite DepMap showing its knockout barely dents growth.
A guard below refuses to run if a dependency column is used as a criterion,
because quietly adding one would reproduce a different paper.

RUN IT
------
    python3 reproduce_cascade.py --features luad_genes.csv --outdir results
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# The paper's reported stage counts.  Everything is measured against these.
# ---------------------------------------------------------------------------
PAPER_COUNTS = {
    "surface_annotated": 1492,
    "tumor_enriched": 90,
    "amplified": 20,
    "consistently_expressed": 19,
    "abundantly_expressed": 6,
}
PAPER_HIT = "MPZL1"

# Thresholds for stated mode.  PUBLISHED means the paper gives this number.
# INFERRED means the paper names the criterion but not the cut point, and the
# value here is a reconstruction chosen to sit near the paper's stage count.
# Any INFERRED line is a caveat that belongs in the write-up, not a fact.
STATED = {
    "fc_min":      (1.0,  "INFERRED",  "log2 tumor/normal fold change"),
    "detect_min":  (0.50, "INFERRED",  "fraction of tumors with detectable expression"),
    "amp_min":     (0.10, "INFERRED",  "fraction of tumors with GISTIC amplification"),
    "sd_max":      (1.50, "INFERRED",  "expression SD across tumors (consistency)"),
    "expr_min":    (5.00, "INFERRED",  "mean expression in tumors (abundance)"),
}

# Column a stage needs -> the stage that needs it.  Checked up front so the
# script fails with a readable message instead of a KeyError three stages in.
REQUIRED = {
    "is_surface":       "surface_annotated",
    "expr_log2fc_tn":   "tumor_enriched",
    "expr_detect_frac": "tumor_enriched",
    "gistic_amp_freq":  "amplified",
    "expr_sd":          "consistently_expressed",
    "expr_mean_tumor":  "abundantly_expressed",
}

# No stage may key off dependency.  See the docstring.
FORBIDDEN_SUBSTRINGS = ("chronos", "effect", "depmap", "essential", "dependency", "label")


def guard_criteria(stage_columns) -> None:
    bad = [c for c in stage_columns
           if any(s in c.lower() for s in FORBIDDEN_SUBSTRINGS)]
    if bad:
        sys.exit("CASCADE GUARD FAILED — a dependency column is being used as a\n"
                 "filter criterion: " + ", ".join(bad) + "\n"
                 "The paper's cascade does not filter on essentiality, and a\n"
                 "nonessential antigen is still a valid CAR-T target. Remove it.")


# ---------------------------------------------------------------------------
def apply_cascade(df, t):
    """Run the four filters in order over the surface-annotated starting set.

    Returns the surviving frame at each stage, in order, so the caller can
    count them and trace individual genes through.
    """
    stages = {}
    s = df[df["is_surface"] == 1]
    stages["surface_annotated"] = s

    s = s[(s["expr_log2fc_tn"] >= t["fc_min"]) &
          (s["expr_detect_frac"] >= t["detect_min"])]
    stages["tumor_enriched"] = s

    s = s[s["gistic_amp_freq"] >= t["amp_min"]]
    stages["amplified"] = s

    s = s[s["expr_sd"] <= t["sd_max"]]
    stages["consistently_expressed"] = s

    s = s[s["expr_mean_tumor"] >= t["expr_min"]]
    stages["abundantly_expressed"] = s
    return stages


def calibrate(df, hit=PAPER_HIT):
    """Pick each threshold so its stage lands on the paper's reported count.

    Each filter is monotone in one direction, so the cut point that yields N
    survivors is just the Nth value in the sorted column.  Stages are
    calibrated in order, because each one sees only what the previous let
    through.  Where a stage cannot reach its target -- too few genes survive
    upstream -- the threshold is left wide open and the shortfall is reported
    rather than papered over.
    """
    t, notes = {}, {}

    def cut(series, n, high_is_better):
        """Threshold admitting exactly n genes, or the loosest possible."""
        v = series.dropna().to_numpy(float)
        if len(v) == 0:
            return (-np.inf if high_is_better else np.inf), "no data"
        if n >= len(v):
            return ((-np.inf if high_is_better else np.inf),
                    f"TARGET MISSED: only {len(v)} reached this stage")
        v = np.sort(v)[::-1] if high_is_better else np.sort(v)
        return float(v[n - 1]), "matched by construction, not by agreement"

    s = df[df["is_surface"] == 1]
    notes["surface_annotated"] = "fixed by annotation; not calibrated"

    # tumor_enriched: calibrate fold change, hold detection at the stated bar
    t["detect_min"] = STATED["detect_min"][0]
    pool = s[s["expr_detect_frac"] >= t["detect_min"]]
    t["fc_min"], notes["tumor_enriched"] = cut(
        pool["expr_log2fc_tn"], PAPER_COUNTS["tumor_enriched"], True)
    s = pool[pool["expr_log2fc_tn"] >= t["fc_min"]]

    t["amp_min"], notes["amplified"] = cut(
        s["gistic_amp_freq"], PAPER_COUNTS["amplified"], True)
    s = s[s["gistic_amp_freq"] >= t["amp_min"]]

    t["sd_max"], notes["consistently_expressed"] = cut(
        s["expr_sd"], PAPER_COUNTS["consistently_expressed"], False)
    s = s[s["expr_sd"] <= t["sd_max"]]

    t["expr_min"], notes["abundantly_expressed"] = cut(
        s["expr_mean_tumor"], PAPER_COUNTS["abundantly_expressed"], True)
    return t, notes


# ---------------------------------------------------------------------------
# Which column each stage tests, against which threshold, in which direction.
# Used to explain WHY a traced gene passed or failed, instead of only where.
CRITERIA = {
    "tumor_enriched": [("expr_log2fc_tn", "fc_min", ">="),
                       ("expr_detect_frac", "detect_min", ">=")],
    "amplified": [("gistic_amp_freq", "amp_min", ">=")],
    "consistently_expressed": [("expr_sd", "sd_max", "<=")],
    "abundantly_expressed": [("expr_mean_tumor", "expr_min", ">=")],
}


def explain(df, gene, thresholds):
    """Score one gene against every criterion, pass or fail.

    The cascade stops at the first failure, so the stage where a gene drops
    out tells you nothing about the other criteria it would have cleared.
    For a gene the paper selected and this reproduction did not, that is the
    whole question: did it miss narrowly on one axis, or is it a poor
    candidate throughout?
    """
    row = df[df["gene"] == gene]
    if row.empty:
        print(f"\n    {gene} is not in the feature table at all.")
        return None
    row = row.iloc[0]

    print(f"\n    {gene} AGAINST EVERY CRITERION (not just up to its drop-out)")
    print(f"      surface annotated: "
          f"{'yes' if row.get('is_surface') == 1 else 'NO'}")
    print(f"      {'criterion':<20}{'value':>10}{'bar':>10}{'':>4}  margin")
    out = []
    for stage, tests in CRITERIA.items():
        for col, key, op in tests:
            if col not in row.index or key not in thresholds:
                continue
            v, bar = float(row[col]), float(thresholds[key])
            if not np.isfinite(bar):
                ok, margin = True, np.nan
            else:
                ok = v >= bar if op == ">=" else v <= bar
                margin = (v - bar) if op == ">=" else (bar - v)
            print(f"      {col:<20}{v:>10.3f}{bar:>10.3f}"
                  f"{'  ok' if ok else ' FAIL':>4}  {margin:+.3f}")
            out.append({"gene": gene, "stage": stage, "criterion": col,
                        "value": v, "threshold": bar, "passed": bool(ok),
                        "margin": float(margin)})

    failed = [r for r in out if not r["passed"]]
    if len(failed) == 0:
        print(f"      {gene} clears every criterion.")
    elif len(failed) == 1:
        f = failed[0]
        print(f"      {gene} clears every criterion but one: {f['criterion']}, "
              f"short by {abs(f['margin']):.3f}.")
        print(f"      A single-criterion miss points at how that one quantity")
        print(f"      is computed, not at the gene's merit as a candidate.")
    else:
        print(f"      {gene} fails {len(failed)} criteria: "
              f"{', '.join(f['criterion'] for f in failed)}.")
    return out


def trace(stages, gene):
    """Where in the cascade a named gene drops out, if it does."""
    last = None
    for name, frame in stages.items():
        if gene in set(frame["gene"]):
            last = name
        else:
            return {"gene": gene, "survived": False,
                    "last_stage": last or "(not in the starting set)",
                    "dropped_at": name}
    return {"gene": gene, "survived": True, "last_stage": last, "dropped_at": None}


def report(label, stages, thresholds, notes, hit):
    print(f"\n  {label.upper()} THRESHOLDS")
    for k, v in thresholds.items():
        tag = STATED.get(k, (None, "", ""))[1] if label == "stated" else "CALIBRATED"
        desc = STATED.get(k, (None, "", "?"))[2]
        shown = "none" if not np.isfinite(v) else f"{v:.4g}"
        print(f"    {k:<12} {shown:>10}   [{tag}]  {desc}")

    print(f"\n  {label.upper()} CASCADE")
    print(f"    {'stage':<26}{'ours':>8}{'paper':>8}{'diff':>8}   note")
    rows = []
    for name, frame in stages.items():
        ours, paper = len(frame), PAPER_COUNTS[name]
        print(f"    {name:<26}{ours:>8}{paper:>8}{ours - paper:>+8}   "
              f"{notes.get(name, '')}")
        rows.append({"mode": label, "stage": name, "ours": ours,
                     "paper": paper, "difference": ours - paper,
                     "note": notes.get(name, "")})

    if label == "calibrated":
        print("\n    A zero difference above is not evidence of agreement: the")
        print("    thresholds were chosen to produce these counts. What the")
        print("    calibrated run tests is which GENES fill the quota, and at")
        print("    what cut points -- compare these thresholds to the stated ones.")

    tr = trace(stages, hit)
    if tr["survived"]:
        print(f"\n    {hit} survives the full cascade.")
    else:
        print(f"\n    {hit} drops out at '{tr['dropped_at']}' "
              f"(last seen in '{tr['last_stage']}').")
    return rows, tr


def main():
    p = argparse.ArgumentParser(
        description="Reproduce the paper's surface-target filter cascade.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--features", type=Path, required=True,
                   help="per-gene feature CSV (same one the model uses)")
    p.add_argument("--hit", default=PAPER_HIT,
                   help="gene to trace through every stage")
    p.add_argument("--outdir", type=Path, default=Path("results"))
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit("ERROR: the feature table is missing columns the cascade needs:\n"
                 + "".join(f"    {c}  (needed by stage '{REQUIRED[c]}')\n"
                           for c in missing))
    guard_criteria(list(REQUIRED))

    print(f"  feature table      {args.features}")
    print(f"  genes              {len(df):,}")
    print(f"  surface-annotated  {int((df['is_surface'] == 1).sum()):,} "
          f"(paper starts from {PAPER_COUNTS['surface_annotated']:,})")
    print(f"  tracing            {args.hit}")

    stated_t = {k: v[0] for k, v in STATED.items()}
    stated_notes = {"surface_annotated": "fixed by annotation"}
    stated_stages = apply_cascade(df, stated_t)
    rows_a, trace_a = report("stated", stated_stages, stated_t, stated_notes, args.hit)

    cal_t, cal_notes = calibrate(df)
    cal_stages = apply_cascade(df, cal_t)
    rows_b, trace_b = report("calibrated", cal_stages, cal_t, cal_notes, args.hit)

    detail = explain(df, args.hit, cal_t)
    if detail:
        pd.DataFrame(detail).to_csv(args.outdir / "traced_gene_criteria.csv",
                                    index=False)

    counts = pd.DataFrame(rows_a + rows_b)
    counts.to_csv(args.outdir / "cascade_counts.csv", index=False)

    final = cal_stages["abundantly_expressed"].copy()
    keep = [c for c in ["gene", "expr_mean_tumor", "expr_log2fc_tn", "expr_sd",
                        "expr_detect_frac", "gistic_amp_freq", "label"]
            if c in final.columns]
    final = final[keep].sort_values("expr_log2fc_tn", ascending=False)
    final.to_csv(args.outdir / "cascade_shortlist.csv", index=False)

    print(f"\n  FINAL SHORT LIST (calibrated mode, {len(final)} genes)")
    if len(final):
        with pd.option_context("display.width", 160):
            print(final.round(3).to_string(index=False))
        if "label" in final.columns and final["label"].notna().any():
            print("\n    The 'label' column is dependency, shown for context only.")
            print("    It was not a filter. Low values are not disqualifying --")
            print("    a gene the tumor does not need can still be a good antigen.")
    else:
        print("    empty — see the stage table above for where it ran dry")

    (args.outdir / "cascade_metadata.json").write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(),
        "features_file": str(args.features),
        "paper_counts": PAPER_COUNTS,
        "stated_thresholds": stated_t,
        "stated_threshold_provenance": {k: v[1] for k, v in STATED.items()},
        "calibrated_thresholds": {k: (None if not np.isfinite(v) else v)
                                  for k, v in cal_t.items()},
        "traced_gene": args.hit,
        "trace_stated": trace_a,
        "trace_calibrated": trace_b,
        "dependency_used_as_filter": False,
        "versions": {"python": sys.version.split()[0],
                     "numpy": np.__version__, "pandas": pd.__version__},
    }, indent=2, default=str))

    print(f"\n  wrote: cascade_counts.csv, cascade_shortlist.csv, "
          f"cascade_metadata.json")
    print("\n  NOTE: thresholds marked INFERRED are reconstructions, not values")
    print("  quoted from the paper. Any stage where stated and calibrated mode")
    print("  disagree is a reporting gap in the paper worth saying out loud.")


if __name__ == "__main__":
    main()
