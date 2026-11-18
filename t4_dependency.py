#!/usr/bin/env python3
"""
T4 — LUAD gene-dependency predictor.  Checkpoint 2 implementation.

WHAT THIS DOES, IN ORDER
------------------------
The checkpoint requires that preprocessing and tuning happen on training and
validation genes, that model choices are frozen before any test run, and that
the primary model is named rather than merely compared.  This script enforces
that order:

  PHASE 1  tune    fit on train, score on validation, pick one configuration
                   per model family.  Equal budget: 4 candidate configs each.
  PHASE 2  freeze  name the primary model and write the success target to
                   frozen_choices.json, computed only from validation numbers.
  PHASE 3  test    refit the frozen configurations and score the held-out test
                   genes exactly once, across several seeds for variability.
  PHASE 4  rank    rank the held-out surface candidates and save checkpoints.

Nothing in phases 1 and 2 reads the test split.  Phase 3 refuses to start
without frozen_choices.json on disk, so the order cannot be skipped.  Run it
in one go, or in two steps if you want the freeze on the record separately:

    python3 t4_dependency.py --features luad_genes.csv --phase tune
    python3 t4_dependency.py --features luad_genes.csv --phase test

THE LABEL IS NOT TARGET VALIDITY
--------------------------------
The label is dependency: how much a LUAD cell needs the gene.  A gene that the
tumor does not need is often the better CAR-T target, because a CAR-T cell
kills whatever displays the antigen.  The candidate ranking in phase 4 is
therefore ordered by surface eligibility, tumor enrichment and amplification,
with predicted dependency carried as an annotation column.  Nonessential
surface antigens stay eligible.

RUN IT
------
    python3 t4_dependency.py --features luad_genes.csv --outdir results
    python3 t4_dependency.py --demo                       # synthetic, no data
"""


from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

try:
    import joblib
except ImportError:
    joblib = None


# ---------------------------------------------------------------------------
# Leakage control.  Section 4 of the checkpoint excludes every effect-derived
# predictor.  Any feature computed from DepMap effect scores carries the label,
# so a model using one scores well and has learned nothing.  These patterns are
# refused by name; dropping them quietly would let a leak pass unnoticed.
# ---------------------------------------------------------------------------
LEAKY_PATTERNS = [
    r"chronos", r"ceres", r"gene_?effect", r"\beffect\b", r"depmap",
    r"essential", r"dependency", r"codependen", r"crispr", r"rnai",
    r"achilles", r"sanger_score", r"bagel", r"\blfc\b",
]
_LEAK_RE = re.compile("|".join(LEAKY_PATTERNS), re.IGNORECASE)
PROTECTED = {"label", "gene", "is_surface", "split"}


def assert_no_leakage(cols) -> None:
    bad = [c for c in cols if c not in PROTECTED and _LEAK_RE.search(c)]
    if bad:
        sys.exit("LEAKAGE CHECK FAILED — these columns look effect-derived:\n"
                 + "".join(f"    {c}\n" for c in bad)
                 + "Rename if the match is a false positive; drop if it is real.")


# ---------------------------------------------------------------------------
# Candidate configurations.  Four per tunable family keeps the tuning budget
# comparable across models, which the grading rubric's comparability standard
# asks for.  The constant model is a control and is not tuned.
# ---------------------------------------------------------------------------
GRIDS: dict[str, list[dict]] = {
    "ridge": [{"alpha": a} for a in (0.1, 1.0, 10.0, 100.0)],
    "random_forest": [{"min_samples_leaf": m, "n_estimators": 400}
                      for m in (1, 2, 5, 10)],
    "gradient_boosting": [
        {"learning_rate": 0.05, "n_estimators": 300, "max_depth": 3},
        {"learning_rate": 0.05, "n_estimators": 300, "max_depth": 2},
        {"learning_rate": 0.10, "n_estimators": 200, "max_depth": 3},
        {"learning_rate": 0.02, "n_estimators": 600, "max_depth": 3},
    ],
}
TRAINED_MODELS = ["ridge", "random_forest", "gradient_boosting"]
CONTROL_MODEL = "constant"


def build(name: str, cfg: dict, seed: int):
    """One model as a pipeline.  Imputation and scaling are fitted inside the
    pipeline, so they only ever see training rows."""
    num = lambda est: Pipeline([("impute", SimpleImputer(strategy="median")),
                                ("scale", StandardScaler()), ("est", est)])
    tree = lambda est: Pipeline([("impute", SimpleImputer(strategy="median")),
                                 ("est", est)])
    if name == "constant":
        return num(DummyRegressor(strategy="mean"))
    if name == "ridge":
        return num(Ridge(random_state=seed, **cfg))
    if name == "random_forest":
        return tree(RandomForestRegressor(n_jobs=-1, random_state=seed, **cfg))
    if name == "gradient_boosting":
        return tree(GradientBoostingRegressor(random_state=seed, **cfg))
    raise ValueError(name)


# ---------------------------------------------------------------------------
def subsample_labeled(df, n_target, n_test_surface, seed):
    """Cut the labeled set to n_target genes, keeping the surface rate.

    DepMap screens ~18,000 genes across 53 LUAD lines, so nearly everything
    clears the three-line bar and the labeled set far exceeds the 2,000 the
    checkpoint specifies.  Dropped genes are set unlabeled rather than deleted:
    they keep their features and stay available as candidates, because a
    candidate does not need a label.
    """
    rng = np.random.default_rng(seed)
    lab = df.index[df["label"].notna()]
    if len(lab) <= n_target:
        return df
    surf = df.index[(df["label"].notna()) & (df["is_surface"] == 1)].to_numpy().copy()
    other = df.index[(df["label"].notna()) & (df["is_surface"] == 0)].to_numpy().copy()
    rng.shuffle(surf); rng.shuffle(other)
    rate = len(surf) / max(len(surf) + len(other), 1)
    n_surf = min(max(int(round(n_target * rate)), n_test_surface * 3), len(surf))
    n_other = min(n_target - n_surf, len(other))
    keep = set(surf[:n_surf]) | set(other[:n_other])
    out = df.copy()
    out.loc[[i for i in lab if i not in keep], "label"] = np.nan
    return out


def make_splits(df, seed, n_test_surface=20, fracs=(0.70, 0.15, 0.15)):
    """Surface-stratified 70/15/15 by gene.  No gene appears in two splits.

    Stratifying matters because surface genes are a small minority; an
    unstratified draw can leave too few in test to measure the metric the
    project is actually about.
    """
    rng = np.random.default_rng(seed)
    lab = df[df["label"].notna()]
    surf = lab.index[lab["is_surface"] == 1].to_numpy().copy()
    other = lab.index[lab["is_surface"] == 0].to_numpy().copy()
    rng.shuffle(surf); rng.shuffle(other)
    if len(surf) < n_test_surface:
        sys.exit(f"Only {len(surf)} labeled surface genes; need >= {n_test_surface}.")

    n = len(lab)
    n_val = int(round(fracs[1] * n))
    n_test_total = int(round(fracs[2] * n))
    test_s, rest_s = surf[:n_test_surface], surf[n_test_surface:]
    need = max(n_test_total - n_test_surface, 0)
    test_o, rest_o = other[:need], other[need:]
    test_idx = np.concatenate([test_s, test_o])

    s_rate = len(rest_s) / max(len(rest_s) + len(rest_o), 1)
    n_val_s = min(int(round(n_val * s_rate)), len(rest_s))
    val_idx = np.concatenate([rest_s[:n_val_s], rest_o[:max(n_val - n_val_s, 0)]])
    train_idx = np.concatenate([rest_s[n_val_s:], rest_o[max(n_val - n_val_s, 0):]])

    out = df.copy()
    out["split"] = pd.NA
    out.loc[train_idx, "split"] = "train"
    out.loc[val_idx, "split"] = "val"
    out.loc[test_idx, "split"] = "test"
    assert not (set(train_idx) & set(val_idx)), "train/val overlap"
    assert not (set(test_idx) & set(val_idx)), "val/test overlap"
    assert not (set(train_idx) & set(test_idx)), "train/test overlap"
    return out


def score(y_true, y_pred, surface_mask) -> dict:
    def pair(yt, yp):
        if len(yt) < 3:
            return np.nan, np.nan
        rho = np.nan if np.ptp(yp) == 0 else stats.spearmanr(yt, yp).statistic
        return mean_absolute_error(yt, yp), rho
    mae_a, rho_a = pair(y_true, y_pred)
    mae_s, rho_s = pair(y_true[surface_mask], y_pred[surface_mask])
    return {"mae_overall": mae_a, "spearman_overall": rho_a,
            "mae_surface": mae_s, "spearman_surface": rho_s,
            "n_eval": len(y_true), "n_surface": int(surface_mask.sum())}


def fit_and_score(df, feat_cols, name, cfg, seed, eval_split):
    tr = df[df["split"] == "train"]
    ev = df[df["split"] == eval_split]
    model = build(name, cfg, seed)
    model.fit(tr[feat_cols].to_numpy(float), tr["label"].to_numpy(float))
    pred = model.predict(ev[feat_cols].to_numpy(float))
    m = score(ev["label"].to_numpy(float), pred, ev["is_surface"].to_numpy() == 1)
    return model, pred, m


# ---------------------------------------------------------------------------
# PHASE 1 — tune on validation
# ---------------------------------------------------------------------------
def phase_tune(df_raw, feat_cols, seeds, n_labeled, n_test_surface, outdir):
    print("\nPHASE 1 — tuning on validation genes (the test split is not read)")
    rows = []
    for seed in seeds:
        d = subsample_labeled(df_raw, n_labeled, n_test_surface, seed)
        d = make_splits(d, seed, n_test_surface)
        for name in TRAINED_MODELS:
            for i, cfg in enumerate(GRIDS[name]):
                _, _, m = fit_and_score(d, feat_cols, name, cfg, seed, "val")
                rows.append({"seed": seed, "model": name, "config_id": i,
                             "config": json.dumps(cfg), **m})
        _, _, m = fit_and_score(d, feat_cols, CONTROL_MODEL, {}, seed, "val")
        rows.append({"seed": seed, "model": CONTROL_MODEL, "config_id": 0,
                     "config": "{}", **m})

    val = pd.DataFrame(rows)
    val.to_csv(outdir / "validation_scores.csv", index=False)

    agg = (val.groupby(["model", "config_id", "config"], as_index=False)
              .agg(mae_mean=("mae_overall", "mean"),
                   mae_sd=("mae_overall", "std"),
                   rho_mean=("spearman_overall", "mean")))
    chosen = {}
    for name in TRAINED_MODELS:
        best = agg[agg["model"] == name].nsmallest(1, "mae_mean").iloc[0]
        chosen[name] = json.loads(best["config"])
        print(f"    {name:<18} best val MAE {best['mae_mean']:.4f} "
              f"(sd {best['mae_sd']:.4f})  config {best['config']}")
    const_mae = float(agg[agg["model"] == CONTROL_MODEL]["mae_mean"].iloc[0])
    print(f"    {CONTROL_MODEL:<18} val MAE {const_mae:.4f}   (control, not tuned)")
    return val, agg, chosen, const_mae


# ---------------------------------------------------------------------------
# PHASE 2 — freeze
# ---------------------------------------------------------------------------
def phase_freeze(agg, chosen, const_mae, outdir, margin):
    print("\nPHASE 2 — freezing the primary model and the success target")
    best = (agg[agg["model"].isin(TRAINED_MODELS)]
            .nsmallest(1, "mae_mean").iloc[0])
    primary = best["model"]
    primary_val = float(best["mae_mean"])

    rival = (agg[(agg["model"].isin(TRAINED_MODELS)) & (agg["model"] != primary)]
             .nsmallest(1, "mae_mean").iloc[0])

    target = round(const_mae * (1 - margin), 4)

    decl = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "primary_model": primary,
        "primary_config": chosen[primary],
        "selection_basis": "lowest mean validation MAE across seeds; the test "
                           "split was not read during selection",
        "primary_metric": "overall MAE on held-out test genes",
        "secondary_metrics": ["surface-subset MAE", "overall Spearman",
                              "surface-subset Spearman"],
        "success_target_mae": target,
        "success_target_rule": f"test MAE at or below {1 - margin:.0%} of the "
                               f"constant control's validation MAE "
                               f"({const_mae:.4f})",
        "comparability_requirement": "primary must match or beat the strongest "
                                     "trained comparator on the primary metric",
        "strongest_comparator_on_val": {"model": rival["model"],
                                        "val_mae": round(float(rival["mae_mean"]), 4)},
        "validation_mae_of_primary": round(primary_val, 4),
        "all_frozen_configs": chosen,
        "tuning_budget_per_family": {k: len(v) for k, v in GRIDS.items()},
    }
    (outdir / "frozen_choices.json").write_text(json.dumps(decl, indent=2))

    print(f"    primary model      {primary}  {json.dumps(chosen[primary])}")
    print(f"    primary metric     overall MAE on held-out test genes")
    print(f"    success target     MAE <= {target}   "
          f"({margin:.0%} below the constant control's val MAE {const_mae:.4f})")
    print(f"    strongest rival    {rival['model']} at val MAE "
          f"{float(rival['mae_mean']):.4f}")
    print(f"    wrote {outdir/'frozen_choices.json'}")
    return decl


# ---------------------------------------------------------------------------
# PHASE 3 — test, once
# ---------------------------------------------------------------------------
def phase_test(df_raw, feat_cols, seeds, n_labeled, n_test_surface,
               decl, outdir, ablation):
    print("\nPHASE 3 — scoring held-out test genes with the frozen choices")
    chosen = decl["all_frozen_configs"]
    rows, keep = [], {}

    amp_cols = [c for c in feat_cols if "gistic" in c.lower() or "amp" in c.lower()]
    feature_sets = {"full": feat_cols}
    if ablation:
        feature_sets["no_amplification"] = [c for c in feat_cols if c not in amp_cols]

    for tag, cols in feature_sets.items():
        for seed in seeds:
            d = subsample_labeled(df_raw, n_labeled, n_test_surface, seed)
            d = make_splits(d, seed, n_test_surface)
            for name in TRAINED_MODELS + [CONTROL_MODEL]:
                cfg = chosen.get(name, {}) if name != CONTROL_MODEL else {}
                model, pred, m = fit_and_score(d, cols, name, cfg, seed, "test")
                rows.append({"feature_set": tag, "seed": seed, "model": name,
                             "n_features": len(cols), **m})
                if tag == "full" and seed == seeds[0]:
                    keep[name] = (model, d, pred)

    test = pd.DataFrame(rows)
    test.to_csv(outdir / "test_scores_by_seed.csv", index=False)

    summ = (test.groupby(["feature_set", "model"], as_index=False)
                .agg(mae_mean=("mae_overall", "mean"), mae_sd=("mae_overall", "std"),
                     rho_mean=("spearman_overall", "mean"),
                     rho_sd=("spearman_overall", "std"),
                     mae_surf_mean=("mae_surface", "mean"),
                     mae_surf_sd=("mae_surface", "std"),
                     rho_surf_mean=("spearman_surface", "mean"),
                     rho_surf_sd=("spearman_surface", "std"),
                     n_seeds=("seed", "nunique")))
    summ.to_csv(outdir / "test_summary.csv", index=False)
    return test, summ, keep


def verdict(summ, decl, outdir):
    """Did the frozen primary model meet its declared target, and did it match
    or beat the strongest trained comparator?  Both are reported either way."""
    full = summ[summ["feature_set"] == "full"]
    primary = decl["primary_model"]
    p = full[full["model"] == primary].iloc[0]
    rivals = full[(full["model"].isin(TRAINED_MODELS)) & (full["model"] != primary)]
    best_rival = rivals.nsmallest(1, "mae_mean").iloc[0]
    const = full[full["model"] == CONTROL_MODEL].iloc[0]

    met = bool(p["mae_mean"] <= decl["success_target_mae"])
    beats = bool(p["mae_mean"] <= best_rival["mae_mean"])

    out = {
        "primary_model": primary,
        "declared_target_mae": decl["success_target_mae"],
        "achieved_test_mae": round(float(p["mae_mean"]), 4),
        "achieved_test_mae_sd": round(float(p["mae_sd"]), 4),
        "target_met": met,
        "strongest_trained_comparator": best_rival["model"],
        "comparator_test_mae": round(float(best_rival["mae_mean"]), 4),
        "primary_matches_or_beats_comparator": beats,
        "metric_gap_vs_comparator": round(
            float(p["mae_mean"] - best_rival["mae_mean"]), 4),
        "constant_control_test_mae": round(float(const["mae_mean"]), 4),
        "improvement_over_control_pct": round(
            100 * (1 - p["mae_mean"] / const["mae_mean"]), 1),
    }
    (outdir / "verdict.json").write_text(json.dumps(out, indent=2))

    print("\n  DECLARED TARGET CHECK")
    print(f"    primary                {primary}")
    print(f"    target MAE             {out['declared_target_mae']}")
    print(f"    achieved MAE           {out['achieved_test_mae']} "
          f"(sd {out['achieved_test_mae_sd']} across {int(p['n_seeds'])} seeds)")
    print(f"    target met             {'YES' if met else 'NO'}")
    print(f"    strongest comparator   {best_rival['model']} at "
          f"{out['comparator_test_mae']}")
    print(f"    primary >= comparator  {'YES' if beats else 'NO'} "
          f"(gap {out['metric_gap_vs_comparator']:+.4f})")
    print(f"    vs constant control    {out['improvement_over_control_pct']}% better")
    return out


# ---------------------------------------------------------------------------
# PHASE 4 — rank the held-out surface candidates
# ---------------------------------------------------------------------------
def phase_rank(keep, decl, outdir):
    """Rank the held-out surface genes as CANDIDATES, not by dependency.

    Candidacy follows the paper's logic: the gene must sit on the surface, run
    higher in tumor than normal, and sit in an amplified region.  Predicted
    dependency is carried as an annotation, because a gene the tumor does not
    need is often the better antigen.  Sorting by dependency would invert the
    paper's own reasoning.
    """
    primary = decl["primary_model"]
    if primary not in keep:
        return None
    model, d, pred = keep[primary]
    te = d[d["split"] == "test"].copy()
    te["pred_dependency"] = pred
    cand = te[te["is_surface"] == 1].copy()
    if cand.empty:
        return None

    def z(s):
        sd = s.std(ddof=0)
        return (s - s.mean()) / sd if sd and np.isfinite(sd) else s * 0.0

    enrich = z(cand["expr_log2fc_tn"]) if "expr_log2fc_tn" in cand else 0.0
    amp = z(cand["gistic_amp_freq"]) if "gistic_amp_freq" in cand else 0.0
    cand["candidacy_score"] = enrich + amp
    cand = cand.sort_values("candidacy_score", ascending=False).reset_index(drop=True)
    cand.insert(0, "rank", np.arange(1, len(cand) + 1))

    cols = ["rank", "gene", "candidacy_score", "expr_log2fc_tn",
            "gistic_amp_freq", "expr_mean_tumor", "pred_dependency", "label"]
    cols = [c for c in cols if c in cand.columns]
    cand[cols].to_csv(outdir / "surface_candidate_ranking.csv", index=False)

    print(f"\nPHASE 4 — ranked {len(cand)} held-out surface candidates")
    print("    ordered by tumor enrichment + amplification; predicted")
    print("    dependency is an annotation, not the sort key")
    with pd.option_context("display.width", 150):
        print(cand[cols].head(10).round(3).to_string(index=False))
    return cand


# ---------------------------------------------------------------------------
def make_demo(n_genes, n_labeled, seed):
    rng = np.random.default_rng(seed)
    genes = [f"GENE{i:05d}" for i in range(n_genes)]
    em = rng.gamma(3.0, 1.4, n_genes).clip(0, 14)
    sd = (0.9 + 0.25 * rng.gamma(2.0, 1.0, n_genes) - 0.03 * em).clip(0.05, None)
    fc = rng.normal(0.1, 1.15, n_genes)
    af = rng.beta(1.25, 7.0, n_genes)
    surf = (rng.random(n_genes) < (0.10 + 0.55 * af).clip(0.02, 0.9)).astype(int)
    pan = (em > 9.0) & (sd < 1.15) & (surf == 0)
    lab = (0.055 * em - 0.30 * sd + 0.10 * fc + 0.22 * af + 0.85 * pan
           + rng.normal(0, 0.26, n_genes))
    df = pd.DataFrame({"gene": genes, "expr_mean_tumor": em.round(4),
                       "expr_log2fc_tn": fc.round(4), "expr_sd": sd.round(4),
                       "expr_iqr": (sd * 1.33).round(4),
                       "expr_detect_frac": (0.15 + 0.062 * em).clip(0, 1).round(4),
                       "gistic_amp_freq": af.round(4),
                       "gistic_mean_score": (af * 2.6).clip(-2, 2).round(4),
                       "is_surface": surf, "label": lab.round(4)})
    unl = rng.choice(n_genes, size=n_genes - n_labeled, replace=False)
    df.loc[unl, "label"] = np.nan
    return df


def main():
    p = argparse.ArgumentParser(
        description="T4 LUAD dependency predictor — Checkpoint 2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--features", type=Path, help="per-gene feature CSV")
    p.add_argument("--demo", action="store_true", help="synthetic data")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                   help="repeated runs, for variability evidence")
    p.add_argument("--n-labeled", type=int, default=2000)
    p.add_argument("--n-test-surface", type=int, default=20)
    p.add_argument("--margin", type=float, default=0.10,
                   help="success target: beat the control's val MAE by this much")
    p.add_argument("--no-ablation", action="store_true")
    p.add_argument("--phase", choices=["all", "tune", "test"], default="all",
                   help="'tune' stops after freezing; 'test' requires an "
                        "existing frozen_choices.json")
    p.add_argument("--outdir", type=Path, default=Path("results"))
    args = p.parse_args()

    if not args.demo and not args.features:
        p.error("give either --demo or --features")
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        print("DEMO mode — synthetic data. Numbers are not biology.")
        df = make_demo(18000, args.n_labeled, args.seeds[0])
    else:
        df = pd.read_csv(args.features)
        for need in ("gene", "label", "is_surface"):
            if need not in df.columns:
                sys.exit(f"ERROR: --features needs a '{need}' column.")

    feat_cols = [c for c in df.columns
                 if c not in PROTECTED and pd.api.types.is_numeric_dtype(df[c])]
    assert_no_leakage(df.columns)

    print(f"  genes {len(df):,} | labeled {int(df['label'].notna().sum()):,} "
          f"| surface {int((df['is_surface'] == 1).sum()):,}")
    print(f"  features ({len(feat_cols)}): {', '.join(feat_cols)}")
    print(f"  seeds: {args.seeds}")

    frozen_path = args.outdir / "frozen_choices.json"

    if args.phase in ("all", "tune"):
        val, agg, chosen, const_mae = phase_tune(
            df, feat_cols, args.seeds, args.n_labeled, args.n_test_surface,
            args.outdir)
        decl = phase_freeze(agg, chosen, const_mae, args.outdir, args.margin)
        if args.phase == "tune":
            print("\n  Stopped after the freeze. The test split has not been "
                  "read.\n  Run again with --phase test to score it.")
            return
    else:
        if not frozen_path.exists():
            sys.exit(f"--phase test needs {frozen_path}, which does not exist.\n"
                     "Run --phase tune first; choices must be frozen before "
                     "the test split is read.")
        decl = json.loads(frozen_path.read_text())
        print(f"\n  Loaded frozen choices from {frozen_path}")
        print(f"    frozen at      {decl['frozen_at']}")
        print(f"    primary model  {decl['primary_model']} "
              f"{json.dumps(decl['primary_config'])}")
        print(f"    target MAE     {decl['success_target_mae']}")

    test, summ, keep = phase_test(
        df, feat_cols, args.seeds, args.n_labeled, args.n_test_surface,
        decl, args.outdir, not args.no_ablation)

    print("\n  TEST RESULTS (mean +/- sd across seeds)")
    show = summ.copy()
    for c in show.columns:
        if show[c].dtype.kind == "f":
            show[c] = show[c].round(4)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(show.to_string(index=False))

    v = verdict(summ, decl, args.outdir)
    cand = phase_rank(keep, decl, args.outdir)

    if joblib is not None:
        ck = args.outdir / "checkpoints"
        ck.mkdir(exist_ok=True)
        for name, (model, _, _) in keep.items():
            joblib.dump(model, ck / f"{name}_seed{args.seeds[0]}.joblib")
        print(f"\n  saved {len(keep)} trained checkpoints to {ck}")
    else:
        print("\n  NOTE: joblib not installed, checkpoints not saved "
              "(pip install joblib)")

    (args.outdir / "run_metadata.json").write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(),
        "mode": "demo" if args.demo else "real",
        "features_file": str(args.features) if args.features else None,
        "seeds": args.seeds, "n_labeled": args.n_labeled,
        "n_test_surface": args.n_test_surface,
        "features_used": feat_cols,
        "leakage_patterns_enforced": LEAKY_PATTERNS,
        "tuning_budget_per_family": {k: len(v) for k, v in GRIDS.items()},
        "verdict": v,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": __import__("sklearn").__version__},
    }, indent=2))

    print(f"\n  wrote: validation_scores.csv, frozen_choices.json,")
    print(f"         test_scores_by_seed.csv, test_summary.csv, verdict.json,")
    print(f"         surface_candidate_ranking.csv, run_metadata.json")
    print("\n  REMINDER: the label is dependency, not target validity.")
    print("  The candidate ranking is ordered by surface eligibility, tumor")
    print("  enrichment and amplification. Nonessential antigens stay eligible.")


if __name__ == "__main__":
    main()
