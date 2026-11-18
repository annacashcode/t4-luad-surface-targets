# T4 — Predicting gene dependency and prioritizing cell-surface targets in LUAD

Anna Cash · CPSC 6430, Fall 2026 · Checkpoint 2

This project does two separate things and keeps them separate on purpose.

1. **A trained predictor.** Given a gene's expression and copy-number features
   in lung adenocarcinoma, predict how much LUAD cells depend on it.
2. **A reproduction of the reference paper.** The paper selects CAR-T surface
   targets with a filter cascade and no model at all. This repository
   reproduces that cascade and reports where it agrees and where it does not.

The two are compared in the report, not merged. They answer different
questions, and conflating them is the error the whole project is built to
avoid.

---

## The one idea that explains the design

**Dependency is not target validity.**

A CAR-T cell kills whatever displays the antigen on the cell surface. It does
not care whether the tumor needs that protein. So a gene whose knockout barely
dents tumor growth can still be an excellent target — and MPZL1, the paper's
headline hit, is exactly that.

This has consequences throughout the code:

- The model predicts dependency, which is a useful descriptor of a gene, not a
  score of how good a target it is.
- The candidate ranking (`surface_candidate_ranking.csv`) is ordered by surface
  eligibility, tumor enrichment and amplification. Predicted dependency rides
  along as an annotation column. Sorting by it would invert the paper's logic.
- `reproduce_cascade.py` has a guard that aborts if a dependency column is ever
  used as a filter criterion.

---

## Requirements

Python 3.11 or newer. Developed and reported on Python 3.14.6 (Anaconda),
macOS.

```bash
python3 -m pip install -r requirements.txt
```

| package | version used |
|---|---|
| pandas | 3.0.5 |
| numpy | 2.4.6 |
| scikit-learn | 1.9.0 |
| scipy | 1.18.0 |
| joblib | 1.5.3 |

---

## Getting the data

The four source files total about 590 MB and are **not** in this repository.
All four are public and free; none require an account except cBioPortal, which
needs none for this dataset. `DOWNLOAD_WALKTHROUGH.md` has click-by-click
instructions.

| file | source | size | what it provides |
|---|---|---|---|
| `CRISPRGeneEffect.csv` | DepMap 24Q2, Chronos gene effect | 474 MB | the label |
| `Model.csv` | DepMap, cell-line metadata | 4 MB | which lines are LUAD |
| `HiSeqV2` | UCSC Xena, TCGA LUAD RNA-seq | 85 MB | expression features |
| `data_cna.txt` | cBioPortal, GISTIC 2.0 thresholded CNA | 29 MB | amplification features |
| `subcellular_location.tsv` | Human Protein Atlas | 2 MB | the surface flag |

Put all five in `data/`. `checksums.txt` records a SHA-256 for each file as
downloaded; verify with `shasum -c data/checksums.txt`.

---

## Layout

```
.
├── README.md
├── requirements.txt
├── DOWNLOAD_WALKTHROUGH.md      how to obtain the five source files
├── build_feature_table.py       five raw files  ->  luad_genes.csv
├── t4_dependency.py             the trained predictor
├── reproduce_cascade.py         the paper reproduction
├── luad_genes.csv               derived feature table, one row per gene
├── luad_genes_provenance.json   what went into it, and how much matched
├── data/                        source files (not committed, see above)
└── results/                     everything below is generated
    ├── frozen_choices.json           model + target, timestamped before testing
    ├── validation_scores.csv         every config on every seed
    ├── test_scores_by_seed.csv       per-seed test scores
    ├── test_summary.csv              mean and sd across seeds
    ├── verdict.json                  did the primary meet its declared target
    ├── surface_candidate_ranking.csv the 20 held-out surface candidates
    ├── run_metadata.json             versions, seeds, features, leak patterns
    ├── cascade_counts.csv            stage counts, ours vs the paper
    ├── cascade_shortlist.csv         the final six genes
    ├── traced_gene_criteria.csv      MPZL1 against every criterion
    ├── cascade_metadata.json         thresholds and their provenance
    └── checkpoints/                  trained models (.joblib)
```

---

## Running it

Three commands, in this order. Total runtime about 8 minutes on an M-series
MacBook; the first command is most of it, reading the 474 MB DepMap matrix.

**1 — Build the feature table** (~4 min)

```bash
python3 build_feature_table.py \
  --depmap-effect data/CRISPRGeneEffect.csv \
  --depmap-model  data/Model.csv \
  --xena          data/HiSeqV2 \
  --cna           data/data_cna.txt \
  --hpa           data/subcellular_location.tsv \
  --out           luad_genes.csv
```

Produces 20,530 genes, 15,707 labeled, 1,478 surface-eligible.

`--inspect` with any subset of those files reads a few rows and reports what it
found, without writing anything. Useful mid-download.

**2 — Tune, freeze, then test** (~3 min)

```bash
python3 t4_dependency.py --features luad_genes.csv --phase tune --outdir results
python3 t4_dependency.py --features luad_genes.csv --phase test --outdir results
```

Deliberately two commands. `--phase tune` fits on training genes, scores on
validation genes, picks one configuration per model family, names a primary
model, writes a success target, and stops. It never reads the test split.
`--phase test` refuses to start unless `frozen_choices.json` already exists, so
the order cannot be skipped. The timestamp in that file predates the test run.

`--phase all` does both in one go if you prefer.

**3 — Reproduce the paper's cascade** (~5 sec)

```bash
python3 reproduce_cascade.py --features luad_genes.csv --outdir results
```

---

## How the model is set up

**One row per gene.** 20,530 genes; 15,707 carry a label. The labeled set is
subsampled to 2,000 with the surface rate preserved. Genes dropped by the
subsample keep their features and become candidates — a candidate does not need
a label.

**Label.** Negated median Chronos gene effect across LUAD cell lines, requiring
at least 3 lines per gene. Negated so that larger means more depended-upon.

**Features (7).** Mean tumor expression, log2 tumor/normal fold change,
expression SD, expression IQR, detection fraction, GISTIC amplification
frequency, GISTIC mean score.

**Leakage control.** Thirteen regular expressions (`chronos`, `gene_effect`,
`depmap`, `essential`, `crispr`, …) are matched against every column name. A
hit aborts the run with an explanation rather than dropping the column quietly,
because a silent drop is a leak you never find out about.

**Splits.** 70/15/15 by gene, stratified on the surface flag, with exactly 20
surface candidates held out in test. Stratification matters because surface
genes are a minority and an unstratified draw can leave too few in test to
measure the thing the project is about. Three assertions check that no gene
appears in two splits.

**Models.** A constant-mean control, ridge, random forest, gradient boosting.
Each tunable family gets four candidate configurations, so the tuning budget is
equal across families. Imputation and scaling live inside the pipeline, so they
only ever see training rows.

**Seeds.** Five (42–46). Each seed redraws the split, so the reported standard
deviations cover split variability, not just model variance.

---

## Results

Primary metric: overall MAE on held-out test genes. Primary model and success
target were frozen from validation before the test split was read.

| | value |
|---|---|
| Primary model | random forest (`min_samples_leaf=10`, `n_estimators=400`) |
| Declared target | MAE ≤ 0.2054 |
| Achieved | **0.2020** (sd 0.0163 across 5 seeds) |
| Target met | yes |
| Strongest comparator | gradient boosting, 0.2077 |
| Improvement over constant control | 12.3% |

Test performance, mean ± sd over 5 seeds:

| model | MAE | Spearman | surface MAE | surface Spearman |
|---|---|---|---|---|
| constant control | 0.2304 ± 0.0230 | — | 0.1536 ± 0.0231 | — |
| ridge | 0.2234 ± 0.0219 | 0.390 ± 0.040 | 0.1476 ± 0.0258 | 0.207 ± 0.130 |
| gradient boosting | 0.2077 ± 0.0182 | 0.395 ± 0.034 | 0.1574 ± 0.0787 | 0.274 ± 0.249 |
| **random forest** | **0.2020 ± 0.0163** | **0.400 ± 0.047** | **0.1241 ± 0.0353** | **0.371 ± 0.252** |

**Amplification ablation.** Dropping the two GISTIC features moves MAE by
0.0006 for random forest — inside the seed-to-seed noise. Amplification
contributes nothing measurable to predicting dependency. That is consistent
with the paper rather than a strike against it: amplification is how you find
candidates, not how you predict whether a cell needs the gene.

### Paper reproduction

| stage | ours | paper | difference |
|---|---|---|---|
| surface annotated | 1,478 | 1,492 | −14 |
| tumor enriched | 131 | 90 | +41 |
| amplified | 116 | 20 | +96 |
| consistently expressed | 57 | 19 | +38 |
| abundantly expressed | 52 | 6 | +46 |

The script runs the cascade twice. **Stated mode** applies thresholds and
reports the gap. **Calibrated mode** moves each threshold until the stage count
matches the paper's exactly, then reports what cut point that required — a zero
difference there proves nothing, since the counts were forced, and the script
says so in its own output.

**The reproduction does not recover MPZL1.** It drops at the tumor-enrichment
stage. Against every criterion individually:

| criterion | MPZL1 | calibrated bar | |
|---|---|---|---|
| surface annotated | yes | — | pass |
| detection fraction | 1.000 | 0.500 | pass |
| amplification frequency | 0.703 | 0.314 | pass |
| expression SD | 0.463 | 2.465 | pass |
| mean tumor expression | 11.92 | 8.84 | pass |
| **log2 fold change** | **0.632** | **1.254** | **fail** |

Four criteria cleared comfortably, one missed. See the limitations below.

---

## Known limitations

**The normal reference compresses fold changes.** This project uses TCGA's 59
adjacent-normal LUAD samples as the baseline rather than GTEx. That avoids a
cross-study batch effect the paper is exposed to, but adjacent-normal tissue
from a cancer patient is not truly normal — it carries field effects from the
tumor beside it, which pulls every fold change toward zero. MPZL1 failing on
fold change alone, while clearing the other four criteria by wide margins, is
what that cost looks like. The trade was deliberate and is defensible in both
directions; it is not free, and this is the invoice.

**The amplification threshold is the largest unexplained gap.** Reproducing the
paper's count of 20 required an amplification frequency of 0.314 where 0.10 was
assumed — a wide distance that suggests a substantially stricter bar than
assumed here.

**Five of five thresholds are reconstructions.** Every cut point in stated mode
is marked `INFERRED` in the code and in `cascade_metadata.json`. None is quoted
from the paper. Where stated and calibrated mode disagree, the disagreement is
about assumptions, not findings.

**Surface-subset performance is weak and unstable.** Surface Spearman is
0.371 ± 0.252 — the spread is two-thirds of the value. With 20 surface genes in
test that is expected, and it is the honest version of the headline number: the
overall Spearman of 0.400 is flattered by housekeeping genes that are easy to
predict and irrelevant as targets.

**The two best models are close.** Random forest beat gradient boosting by
0.0021 on validation and 0.0057 on test, against a seed-to-seed spread of
roughly 0.017. The ordering held, but the margin is small relative to the
noise, and the result should be read as "these two perform comparably," not as
a decisive win.

**Dependency is a weak proxy for anything clinical.** DepMap measures what
immortalized cell lines in culture need. That is not what a tumor in a patient
needs, and this project does not claim otherwise.

---

## Reproducing the reported numbers exactly

Seeds 42–46 are the default. With the same five source files and the versions
in `requirements.txt`, the three commands above reproduce every number in this
README and in the report. `results/run_metadata.json` records the versions,
seeds, feature list and leakage patterns of the run that produced them.
