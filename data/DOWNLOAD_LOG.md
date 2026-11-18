# T4 — public data download and access log

Checkpoint 1 start-evidence package · CPSC 6430 · Anna Cash

All four sources are public. None required an application or a controlled-access
tier. Every file was verified with an automated inspection pass before being used
to build anything, and the verification output is reproduced below.

**How these files are identified.** Portal release strings are labels applied by
the publisher and are not carried inside the files themselves. Rather than record
a label I did not capture at download time, each file is identified here by two
properties that are intrinsic to the artifact: its SHA-256 checksum, and its
shape. A checksum identifies the exact bytes, so anyone repeating this work can
confirm they hold the same file, which a version string alone does not
guarantee. Row and column counts are release-specific and serve as a secondary
fingerprint.

---

## Summary

| # | Source | File | Identifying shape | Size | Downloaded | Supplies |
|---|---|---|---|---|---|---|
| 1 | DepMap | `CRISPRGeneEffect.csv` | 2,554 models × 18,434 genes | 474 MB | 2026-10-14 | the label |
| 2 | DepMap | `Model.csv` | 2,554 rows × 48 columns | 771 KB | 2026-10-14 | LUAD line identification |
| 3 | UCSC Xena | `HiSeqV2` | dataset version 2017-10-13; 576 samples | 81 MB | 2026-10-14 | expression features |
| 4 | cBioPortal | `data_cna.txt` | LUAD TCGA Firehose Legacy; 516 samples | 28 MB | 2026-10-14 | amplification features |
| 5 | Human Protein Atlas | `subcellular_location.tsv` | 13,603 genes covered | 2 MB | 2026-10-14 | surface eligibility |

## Checksums

Generated with `shasum -a 256` on 2026-10-23.

```
1e0c492944a7297c0eda179081a1e81a1cfe41a97628facd7eddbdda1a7504ee  CRISPRGeneEffect.csv
65eb1895800e71a963a3ab56e5d08c8d006cb0d9e7aa68f7e34a3810a2a3cbc7  Model.csv
64421a30147a689571f4cc537eb484b294e4325807e2b83031e7d918e07fe83c  HiSeqV2
7c2a3b0c9967d4e8ddafe6d18a50b764ce8e8d9ace343360a4c489ce7c675b2d  data_cna.txt
19ee031772375318e75556e43a5103cd8d919f7757094ef3fd134fba134fc095  subcellular_location.tsv
```

Anyone repeating this work can confirm they hold the identical files by saving
the block above as `checksums.txt` beside the data and running:

```
cd ~/t4/data && shasum -a 256 -c checksums.txt
```

Five lines reading `OK` means the files match byte for byte.

---

## 1–2 · DepMap — CRISPR gene effect and model metadata

**Access route:** depmap.org → Data → Downloads → current DepMap Public release.
Public download, email registration only.

**Release identification.** The portal release label was not captured at download
time. The file is identified instead by its checksum above and by its shape:
2,554 models and 18,434 genes, which is specific to one DepMap Public release.

**Verified on download:**

```
[DepMap gene effect]  CRISPRGeneEffect.csv   474 MB
    rows are cell lines, columns are genes
    index column : Unnamed: 0
    first genes  : A1BG (1), A1CF (29974), A2M (2)
    genes        : 18,434

[DepMap model metadata]  Model.csv   771 KB
    2,554 cell lines, 48 columns
    LUAD lines   : 98
    examples     : HCC827, NCIH1693, HCC827GR5, PC14, NCIH1819, RERFLCMS
    OK - the label needs at least 3 LUAD lines and you have enough.
```

**Note on the LUAD count.** 98 cell lines carry the Oncotree code LUAD in the
metadata, but only **53** of those appear in the CRISPR effect matrix. The label
is the median Chronos effect across those 53 screened lines. 17,930 genes are
measured in at least 3 of them and therefore carry a label.

---

## 3 · UCSC Xena — LUAD expression

**Access route:** xenabrowser.net/datapages → TCGA Hub → cohort "TCGA Lung
Adenocarcinoma (LUAD)" → gene expression RNAseq → IlluminaHiSeq.

**Dataset ID:** `TCGA.LUAD.sampleMap/HiSeqV2`
**Hub:** `https://tcga.xenahubs.net`
**Dataset version:** 2017-10-13
**Unit:** log2(norm_count + 1)
**Platform:** IlluminaHiSeq_RNASeqV2

**Why this dataset and not the GDC one.** Both hubs carry a LUAD cohort under
near-identical names. The TCGA hub dataset uses gene symbols; the GDC hub uses
Ensembl IDs. Every other source in this project joins on symbols, so the GDC
version would require a mapping table before anything could be merged. The
TCGA hub values are also already log2-transformed, matching what the features assume.

**Verified on download:**

```
[Xena expression]  HiSeqV2   81 MB
    index column : sample
    samples      : 576
    first sample : TCGA-69-7978-01
    parsed as    : 517 tumor, 59 normal, 0 unparsed
    first gene id: ARHGEF10L
```

Tumor and normal status is read from the TCGA barcode's fourth field (01–09 =
tumor, 10–19 = normal), so no separate phenotype file is needed. The 59
adjacent-normal samples are used as the comparator for tumor-versus-normal
enrichment, in place of GTEx.

---

## 4 · cBioPortal — GISTIC copy number

**Access route:** cbioportal.org/datasets → "Lung Adenocarcinoma (TCGA, Firehose
Legacy)" → study archive download → extract → `data_cna.txt`.

**Study:** Lung Adenocarcinoma (TCGA, Firehose Legacy)
**File taken:** `data_cna.txt` — GISTIC 2.0 thresholded calls
**File deliberately not taken:** `data_log2_cna.txt` — continuous log ratios,
which would not support a "fraction of samples amplified" feature.

**Verified on download:**

```
[cBioPortal GISTIC copy number]  data_cna.txt   28 MB
    columns      : Hugo_Symbol, Entrez_Gene_Id, TCGA-05-4244-01 ...
    samples      : 516
    value range  : -1 to 2
    OK - looks like GISTIC thresholded calls (-2..2).
```

516 samples matches the LUAD cohort size reported in the reference paper.

---

## 5 · Human Protein Atlas — subcellular location

**Access route:** proteinatlas.org/about/download → Subcellular section →
"Protein location data across 13603 genes" → `subcellular_location.tsv.zip`.

**Version identification.** The HPA version label was not captured at download
time. The file is identified by its checksum above and by its coverage of 13,603
genes, which is the figure the download page states for this release.

**Verified on download:**

```
[HPA subcellular location]  subcellular_location.tsv   2 MB
    columns      : Gene, Gene name, Reliability, Main location, Additional location, Extracellular location
    location cols: Main location, Additional location, Extracellular location
    plasma-membrane hits in first 200 rows: 38
```

**Join-column note.** This file carries both a `Gene` column of Ensembl IDs and a
`Gene name` column of symbols, with the Ensembl column first. Joining on the
wrong one matches nothing and produces a complete but empty surface flag. The
loader selects by preference order and refuses outright if the chosen column
turns out to hold Ensembl IDs. This was found by testing against mock files in
the real formats before any real data arrived.

---

## Resulting table

```
genes in table         20,530
labeled                15,707
surface-eligible        2,037
labeled AND surface     1,963
```

Merge rates against the 20,530-gene expression matrix: amplification 87.3%,
surface flag 56.8%, label 76.5%. These reflect genuine differences in gene
coverage between the four databases, not identifier mismatches — a wrong join
key produces a match rate near zero, not 57–87%.

Full provenance, including package versions and the exact input paths, is
recorded automatically in `luad_genes_provenance.json`.

---

## Reproducibility

Every run writes a metadata file capturing parameters, the random seed, and
package versions. Split membership is written to `split_dictionary.json`,
including the identity of the 20 held-out surface candidates. Nothing about the
splits depends on run order or on an unrecorded seed.
