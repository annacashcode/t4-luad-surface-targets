#!/usr/bin/env python3
"""
T4 — build the per-gene feature table from the four public downloads.

This runs on YOUR laptop, where the downloaded files are. It turns:

    DepMap  CRISPRGeneEffect.csv + Model.csv   ->  label
    Xena    LUAD expression matrix             ->  expression + variability features
    cBioPortal  data_cna.txt (GISTIC)          ->  amplification features
    HPA     subcellular_location.tsv           ->  is_surface flag

into one CSV with one row per gene, which `t4_dependency.py --features` consumes.

TWO MODES
---------
Inspect one file as soon as it finishes downloading, before you have the others:

    python3 build_feature_table.py --inspect --depmap-effect ~/Downloads/CRISPRGeneEffect.csv \
                                             --depmap-model  ~/Downloads/Model.csv

Inspect accepts any subset. It reads only the first rows, prints what it found,
and tells you whether the file is the one you wanted. It writes nothing.

Build the table once all four are down:

    python3 build_feature_table.py \
        --depmap-effect CRISPRGeneEffect.csv --depmap-model Model.csv \
        --xena TCGA-LUAD.star_tpm.tsv.gz \
        --cna data_cna.txt \
        --hpa subcellular_location.tsv \
        --out luad_genes.csv

REQUIREMENTS:  python3 -m pip install pandas numpy

ON GENE IDENTIFIERS
-------------------
These four sources do not share one identifier. DepMap column headers are
"SYMBOL (ENTREZ)", cBioPortal carries Hugo_Symbol and Entrez_Gene_Id, HPA carries
Ensembl and a gene name, and Xena depends on which hub you downloaded from. The
join key here is the uppercased gene symbol, because it is the only field all
four reliably carry. Every merge reports how many rows it lost. Treat a loss
over ~10% as a problem to investigate, not a rounding error -- it usually means
one file uses Ensembl IDs and needs mapping first.
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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def say(msg: str = "") -> None:
    print(msg, flush=True)


def human(path: Path) -> str:
    n = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def sep_for(path: Path) -> str:
    """Pick the delimiter from the extension, seeing through .gz/.zip."""
    name = path.name.lower()
    for z in (".gz", ".zip", ".bz2", ".xz"):
        if name.endswith(z):
            name = name[: -len(z)]
    return "," if name.endswith(".csv") else "\t"


def pick_column(df: pd.DataFrame, preferences: list[str]) -> str | None:
    """Find a column by PREFERENCE order, not by the order columns happen to sit in.

    This matters: HPA's subcellular_location.tsv has both "Gene" (an Ensembl ID)
    and "Gene name" (the symbol), with "Gene" first. Scanning columns in file
    order and taking the first loose match picks the Ensembl column, and the
    join against gene symbols then silently matches nothing.
    """
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for want in preferences:
        if want in lowered:
            return lowered[want]
    return None


def norm_symbol(s) -> str:
    """DepMap headers look like 'A1BG (1)'. Strip the Entrez id, uppercase."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s*\(\d+\)\s*$", "", s).strip().upper()


def tcga_sample_type(barcode: str) -> str | None:
    """TCGA barcodes encode tumor/normal in the 4th field.

    TCGA-05-4384-01A-01R-1206-07
                    ^^ 01-09 tumor, 10-19 normal, 20-29 control.
    This is why no separate phenotype file is needed for the tumor/normal split.
    """
    parts = str(barcode).split("-")
    if len(parts) < 4 or len(parts[3]) < 2 or not parts[3][:2].isdigit():
        return None
    code = int(parts[3][:2])
    if 1 <= code <= 9:
        return "tumor"
    if 10 <= code <= 19:
        return "normal"
    return None


def merge_report(left: pd.DataFrame, right: pd.DataFrame, on: str,
                 label: str) -> pd.DataFrame:
    before = len(left)
    out = left.merge(right, on=on, how="left")
    matched = int(out[right.columns.difference([on])[0]].notna().sum())
    pct = 100 * matched / max(before, 1)
    flag = "" if pct >= 90 else "   <-- CHECK THIS"
    say(f"    merged {label:<22} {matched:>7,} / {before:,} genes matched "
        f"({pct:.1f}%){flag}")
    return out


# ---------------------------------------------------------------------------
# inspectors — read a few rows, report, write nothing
# ---------------------------------------------------------------------------
def inspect_depmap_effect(path: Path) -> None:
    say(f"\n[DepMap gene effect]  {path.name}   {human(path)}")
    head = pd.read_csv(path, nrows=3)
    first, cols = head.columns[0], list(head.columns[1:4])
    say(f"    rows are cell lines, columns are genes")
    say(f"    index column : {first}")
    say(f"    first genes  : {', '.join(cols)}")
    if not any(re.search(r"\(\d+\)$", str(c)) for c in cols):
        say("    NOTE: expected headers like 'A1BG (1)'. If these are plain")
        say("          symbols that is fine; if they are Ensembl IDs, you have")
        say("          the wrong file.")
    n_genes = len(head.columns) - 1
    say(f"    genes        : {n_genes:,}")
    if n_genes < 5000:
        say("    WARNING: far fewer genes than a genome-wide screen. Check you")
        say("             downloaded CRISPRGeneEffect.csv and not a subset.")


def inspect_depmap_model(path: Path) -> None:
    say(f"\n[DepMap model metadata]  {path.name}   {human(path)}")
    df = pd.read_csv(path, low_memory=False)
    say(f"    {len(df):,} cell lines, {len(df.columns)} columns")

    code_col = next((c for c in ("OncotreeCode", "oncotree_code")
                     if c in df.columns), None)
    if code_col is None:
        say("    ERROR: no OncotreeCode column. Columns present:")
        say(f"           {', '.join(list(df.columns)[:12])} ...")
        return

    luad = df[df[code_col].astype(str).str.upper() == "LUAD"]
    say(f"    LUAD lines   : {len(luad)}")
    if len(luad) >= 3:
        names = luad.get("StrippedCellLineName", luad.get("CellLineName"))
        if names is not None:
            say(f"    examples     : {', '.join(map(str, names.head(6)))}")
        say("    OK - the label needs at least 3 LUAD lines and you have enough.")
    else:
        say("    PROBLEM: fewer than 3 LUAD lines. The label as defined in the")
        say("             checkpoint cannot be built. Check the release, or ask")
        say("             whether a broader lung grouping is acceptable.")


def inspect_xena(path: Path) -> None:
    say(f"\n[Xena expression]  {path.name}   {human(path)}")
    head = pd.read_csv(path, sep=sep_for(path), nrows=4)
    say(f"    index column : {head.columns[0]}")
    say(f"    samples      : {len(head.columns) - 1:,}")

    ids = [str(c) for c in head.columns[1:]]
    types = [tcga_sample_type(c) for c in ids]
    n_t = sum(t == "tumor" for t in types)
    n_n = sum(t == "normal" for t in types)
    say(f"    first sample : {ids[0] if ids else '(none)'}")
    say(f"    parsed as    : {n_t} tumor, {n_n} normal, "
        f"{sum(t is None for t in types)} unparsed")

    if n_t == 0:
        say("    WARNING: no TCGA barcodes recognised in the column headers.")
        say("             Either this is not a TCGA matrix, or the columns are")
        say("             genes rather than samples (matrix is transposed).")
    elif n_n == 0:
        say("    NOTE: no normal samples. expr_log2fc_tn cannot be computed from")
        say("          this file alone; you would need GTEx as the comparator.")

    first_row = pd.read_csv(path, sep=sep_for(path), nrows=1)
    gene0 = str(first_row.iloc[0, 0])
    say(f"    first gene id: {gene0}")
    if gene0.upper().startswith("ENSG"):
        say("    NOTE: Ensembl IDs. They will need mapping to symbols before")
        say("          joining. Re-download the 'gene symbol' version if offered.")


def inspect_cna(path: Path) -> None:
    say(f"\n[cBioPortal GISTIC copy number]  {path.name}   {human(path)}")
    head = pd.read_csv(path, sep=sep_for(path), nrows=5, low_memory=False)
    say(f"    columns      : {', '.join(map(str, head.columns[:3]))} ...")
    say(f"    samples      : {max(len(head.columns) - 2, 0):,}")

    sample_cols = [c for c in head.columns
                   if str(c).upper().startswith("TCGA")]
    if not sample_cols:
        say("    WARNING: no TCGA sample columns found. Check this is data_cna.txt")
        say("             from the LUAD study and not a different file.")
        return

    vals = pd.to_numeric(
        head[sample_cols].to_numpy().ravel(), errors="coerce")
    vals = vals[~np.isnan(vals)]
    if len(vals):
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        say(f"    value range  : {lo:g} to {hi:g}")
        if lo < -2.5 or hi > 2.5:
            say("    NOTE: values outside -2..2. This looks like continuous log")
            say("          ratios, not GISTIC thresholded calls. You want the")
            say("          thresholded file (data_cna.txt), not data_log2_cna.txt.")
        else:
            say("    OK - looks like GISTIC thresholded calls (-2..2).")


def inspect_hpa(path: Path) -> None:
    say(f"\n[HPA subcellular location]  {path.name}   {human(path)}")
    df = pd.read_csv(path, sep=sep_for(path), nrows=200, low_memory=False)
    say(f"    columns      : {', '.join(map(str, df.columns[:6]))}")
    loc_cols = [c for c in df.columns if "location" in str(c).lower()]
    if not loc_cols:
        say("    WARNING: no location column. Check this is subcellular_location.tsv")
        return
    say(f"    location cols: {', '.join(map(str, loc_cols))}")
    blob = df[loc_cols].astype(str).apply(lambda s: s.str.lower())
    hits = blob.apply(lambda s: s.str.contains("plasma membrane", na=False)).any(axis=1)
    say(f"    plasma-membrane hits in first {len(df)} rows: {int(hits.sum())}")
    if hits.sum() == 0:
        say("    NOTE: none in this sample. Not necessarily wrong - the file is")
        say("          alphabetical and membrane proteins may appear later.")


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def build_label(effect: Path, model: Path, min_lines: int = 3) -> pd.DataFrame:
    say("\nBuilding label from DepMap ...")
    md = pd.read_csv(model, low_memory=False)
    code_col = next((c for c in ("OncotreeCode", "oncotree_code")
                     if c in md.columns), None)
    if code_col is None:
        sys.exit("ERROR: Model.csv has no OncotreeCode column.")

    id_col = "ModelID" if "ModelID" in md.columns else md.columns[0]
    luad_ids = set(md.loc[md[code_col].astype(str).str.upper() == "LUAD",
                          id_col].astype(str))
    say(f"    LUAD cell lines in metadata: {len(luad_ids)}")
    if len(luad_ids) < min_lines:
        sys.exit(f"ERROR: need at least {min_lines} LUAD lines, found {len(luad_ids)}.")

    eff = pd.read_csv(effect, low_memory=False)
    eff = eff.rename(columns={eff.columns[0]: "ModelID"})
    eff["ModelID"] = eff["ModelID"].astype(str)

    sub = eff[eff["ModelID"].isin(luad_ids)]
    say(f"    LUAD lines present in the effect matrix: {len(sub)}")
    if len(sub) < min_lines:
        sys.exit(f"ERROR: only {len(sub)} LUAD lines appear in CRISPRGeneEffect.csv.")

    gene_cols = [c for c in sub.columns if c != "ModelID"]
    med = sub[gene_cols].median(axis=0, skipna=True)

    # Require the gene to be measured in at least min_lines of those lines;
    # a median over one or two lines is not what the checkpoint asks for.
    counts = sub[gene_cols].notna().sum(axis=0)
    med = med[counts >= min_lines]
    say(f"    genes measured in >= {min_lines} LUAD lines: {len(med):,}")

    out = pd.DataFrame({
        "gene": [norm_symbol(g) for g in med.index],
        "label": -med.to_numpy(),           # negate: higher = more essential
    })
    return out[out["gene"] != ""].drop_duplicates("gene").reset_index(drop=True)


def build_expression(xena: Path) -> pd.DataFrame:
    say("\nBuilding expression features from Xena ...")
    df = pd.read_csv(xena, sep=sep_for(xena), low_memory=False)
    gene_col = df.columns[0]
    df[gene_col] = df[gene_col].astype(str).str.upper().str.strip()

    types = {c: tcga_sample_type(c) for c in df.columns[1:]}
    tumor = [c for c, t in types.items() if t == "tumor"]
    normal = [c for c, t in types.items() if t == "normal"]
    say(f"    {len(tumor)} tumor, {len(normal)} normal samples")
    if not tumor:
        sys.exit("ERROR: no tumor samples recognised. Run --inspect on this file.")

    T = df[tumor].apply(pd.to_numeric, errors="coerce")
    out = pd.DataFrame({
        "gene": df[gene_col],
        "expr_mean_tumor": T.mean(axis=1).to_numpy(),
        "expr_sd": T.std(axis=1).to_numpy(),
        "expr_iqr": (T.quantile(0.75, axis=1) - T.quantile(0.25, axis=1)).to_numpy(),
        "expr_detect_frac": (T >= 1).mean(axis=1).to_numpy(),
    })

    if normal:
        N = df[normal].apply(pd.to_numeric, errors="coerce")
        out["expr_log2fc_tn"] = T.mean(axis=1).to_numpy() - N.mean(axis=1).to_numpy()
    else:
        out["expr_log2fc_tn"] = np.nan
        say("    NOTE: no normal samples; expr_log2fc_tn left empty.")

    return out.drop_duplicates("gene").reset_index(drop=True)


def build_amplification(cna: Path) -> pd.DataFrame:
    say("\nBuilding amplification features from GISTIC calls ...")
    df = pd.read_csv(cna, sep=sep_for(cna), low_memory=False)
    sym_col = pick_column(
        df, ["hugo_symbol", "gene_symbol", "gene name", "gene"]) or df.columns[0]
    say(f"    joining on column '{sym_col}'")
    sample_cols = [c for c in df.columns if str(c).upper().startswith("TCGA")]
    if not sample_cols:
        sys.exit("ERROR: no TCGA sample columns in the CNA file.")
    say(f"    {len(sample_cols)} samples")

    V = df[sample_cols].apply(pd.to_numeric, errors="coerce")
    amplified = V.isin([1, 2])
    measured = V.notna().sum(axis=1).replace(0, np.nan)

    out = pd.DataFrame({
        "gene": df[sym_col].astype(str).str.upper().str.strip(),
        "gistic_amp_freq": (amplified.sum(axis=1) / measured).to_numpy(),
        "gistic_mean_score": V.mean(axis=1).to_numpy(),
    })
    return out.drop_duplicates("gene").reset_index(drop=True)


def build_surface(hpa: Path, strict: bool = True) -> pd.DataFrame:
    """Flag genes annotated to the plasma membrane.

    HPA's subcellular_location.tsv carries three location columns: "Main
    location", "Additional location" and "Extracellular location".  Scanning
    all three and accepting any hit is far too permissive -- a nuclear protein
    with a secondary plasma-membrane annotation gets flagged as a surface
    gene.  That is how UBE2C (a ubiquitin-conjugating enzyme) and ASPM (a
    spindle protein) end up in a cell-surface antigen short list.

    strict=True uses the main location only, which is what a CAR-T target
    requires: the antigen has to be on the surface predominantly, not
    incidentally.  Both counts are printed either way so the gap is visible.
    """
    say("\nBuilding surface flag from HPA ...")
    df = pd.read_csv(hpa, sep=sep_for(hpa), low_memory=False)
    # "gene name" first: HPA's "Gene" column holds Ensembl IDs, not symbols.
    name_col = pick_column(df, ["gene name", "gene_name", "gene symbol", "gene"])
    if name_col is None:
        sys.exit(f"ERROR: no gene-name column in HPA file. Columns: "
                 f"{list(df.columns)[:8]}")
    say(f"    joining on column '{name_col}'")
    if df[name_col].astype(str).str.upper().str.startswith("ENSG").mean() > 0.5:
        sys.exit(f"ERROR: column '{name_col}' holds Ensembl IDs, not symbols. "
                 f"The join would match nothing. Columns available: "
                 f"{list(df.columns)[:8]}")

    def pm_hits(cols):
        if not cols:
            return pd.Series(False, index=df.index)
        blob = df[cols].astype(str).apply(lambda s: s.str.lower())
        return blob.apply(
            lambda s: s.str.contains("plasma membrane", na=False)).any(axis=1)

    loc_cols = [c for c in df.columns if "location" in str(c).lower()]
    main_col = pick_column(df, ["main location"])
    any_pm = pm_hits(loc_cols)
    main_pm = pm_hits([main_col]) if main_col else None

    if main_pm is None:
        say("    WARNING: no 'Main location' column found; falling back to any "
            "location column. The surface set will be too permissive.")
        is_pm, mode = any_pm, "any location (fallback)"
    else:
        is_pm, mode = (main_pm, "main location") if strict else (any_pm, "any location")

    def count(mask):
        return int(pd.DataFrame({
            "gene": df[name_col].astype(str).str.upper().str.strip(),
            "f": mask.astype(int).to_numpy(),
        }).groupby("gene")["f"].max().sum())

    say(f"    plasma membrane as MAIN location:  "
        f"{count(main_pm) if main_pm is not None else 'n/a':>6}")
    say(f"    plasma membrane in ANY location:   {count(any_pm):>6}")
    say(f"    using: {mode}")

    out = pd.DataFrame({
        "gene": df[name_col].astype(str).str.upper().str.strip(),
        "is_surface": is_pm.astype(int).to_numpy(),
    })
    out = out.groupby("gene", as_index=False)["is_surface"].max()
    say(f"    {int(out['is_surface'].sum()):,} genes flagged as surface")
    return out


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Build the T4 per-gene feature table from the four downloads.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inspect", action="store_true",
                   help="report on whichever files are given; write nothing")
    p.add_argument("--depmap-effect", type=Path, help="CRISPRGeneEffect.csv")
    p.add_argument("--depmap-model", type=Path, help="Model.csv")
    p.add_argument("--xena", type=Path, help="LUAD expression matrix")
    p.add_argument("--cna", type=Path, help="data_cna.txt (GISTIC thresholded)")
    p.add_argument("--hpa", type=Path, help="subcellular_location.tsv")
    p.add_argument("--min-lines", type=int, default=3,
                   help="LUAD lines required per gene for the label")
    p.add_argument("--surface-any", action="store_true",
                   help="flag a gene as surface if plasma membrane appears in "
                        "ANY HPA location column, not just the main one. This "
                        "is the old, over-permissive behaviour; kept so the "
                        "two definitions can be compared.")
    p.add_argument("--out", type=Path, default=Path("luad_genes.csv"))
    args = p.parse_args()

    given = {k: v for k, v in vars(args).items()
             if isinstance(v, Path) and k != "out"}
    for name, path in given.items():
        if path and not path.exists():
            sys.exit(f"ERROR: {path} does not exist.")

    if args.inspect:
        if not any(given.values()):
            sys.exit("Give --inspect at least one file to look at.")
        say("=" * 68)
        say("INSPECT - reading a few rows of each file. Nothing is written.")
        say("=" * 68)
        if args.depmap_effect: inspect_depmap_effect(args.depmap_effect)
        if args.depmap_model:  inspect_depmap_model(args.depmap_model)
        if args.xena:          inspect_xena(args.xena)
        if args.cna:           inspect_cna(args.cna)
        if args.hpa:           inspect_hpa(args.hpa)
        say("\nDone. Re-run without --inspect once all four are downloaded.")
        return

    missing = [f"--{k.replace('_', '-')}" for k, v in given.items() if v is None]
    if missing:
        sys.exit("ERROR: building the table needs all of: "
                 + ", ".join(missing) + "\nUse --inspect to check files one at a time.")

    label = build_label(args.depmap_effect, args.depmap_model, args.min_lines)
    expr = build_expression(args.xena)
    amp = build_amplification(args.cna)
    surf = build_surface(args.hpa, strict=not args.surface_any)

    say("\nJoining on gene symbol ...")
    say(f"    starting from {len(expr):,} genes in the expression matrix")
    tbl = expr
    tbl = merge_report(tbl, amp, "gene", "amplification")
    tbl = merge_report(tbl, surf, "gene", "surface flag")
    tbl = merge_report(tbl, label, "gene", "label (DepMap)")

    tbl["is_surface"] = tbl["is_surface"].fillna(0).astype(int)

    n_lab = int(tbl["label"].notna().sum())
    n_surf_lab = int(((tbl["label"].notna()) & (tbl["is_surface"] == 1)).sum())

    cols = ["gene", "expr_mean_tumor", "expr_log2fc_tn", "expr_sd", "expr_iqr",
            "expr_detect_frac", "gistic_amp_freq", "gistic_mean_score",
            "is_surface", "label"]
    tbl[cols].to_csv(args.out, index=False)

    say("\n" + "=" * 68)
    say(f"  genes in table         {len(tbl):,}")
    say(f"  labeled                {n_lab:,}")
    say(f"  surface-eligible       {int((tbl['is_surface'] == 1).sum()):,}")
    say(f"  labeled AND surface    {n_surf_lab:,}")
    say("=" * 68)

    if n_surf_lab < 20:
        say("\n  WARNING: fewer than 20 labeled surface genes. The checkpoint")
        say("           asks for 20 surface candidates in the test split, so")
        say("           the split builder will refuse. Check the HPA join first.")
    if n_lab < 2000:
        say(f"\n  NOTE: {n_lab:,} labeled genes, fewer than the 2,000 the")
        say("        checkpoint specifies. Use what you have and say so, or")
        say("        loosen --min-lines after checking that is acceptable.")

    prov = args.out.with_name(args.out.stem + "_provenance.json")
    prov.write_text(json.dumps({
        "built_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {k: str(v) for k, v in given.items() if v},
        "min_lines_per_gene": args.min_lines,
        "join_key": "uppercased gene symbol",
        "n_genes": int(len(tbl)), "n_labeled": n_lab,
        "n_surface": int((tbl["is_surface"] == 1).sum()),
        "n_labeled_surface": n_surf_lab,
        "versions": {"python": sys.version.split()[0],
                     "pandas": pd.__version__, "numpy": np.__version__},
    }, indent=2))

    say(f"\n  wrote {args.out}")
    say(f"  wrote {prov}   <- keep this, it is your data card's evidence")
    say(f"\n  next:  python3 t4_dependency.py --features {args.out} --ablation")


if __name__ == "__main__":
    main()
