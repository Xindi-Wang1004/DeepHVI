# Data

[简体中文](zh-CN/data.md) · [Documentation index](README.md)

## Sources and attribution

DeepHVI pairs human and viral proteins using interaction records from the [Human–Virus Interaction Database (HVIDB)](https://zzdlab.com/hvidb/) and protein sequences matched by identifier from [UniProt](https://www.uniprot.org/). The vector tables distributed here are derived from those matched and processed records.

Cite HVIDB as: Yang X, Lian X, Fu C, Wuchty S, Yang S, Zhang Z. *HVIDB: a comprehensive database for human-virus protein-protein interactions*. Briefings in Bioinformatics, 2021, 22:832–844. [doi:10.1093/bib/bbaa425](https://doi.org/10.1093/bib/bbaa425). Use UniProt's [official citation guidance](https://www.uniprot.org/help/publications) for its sequences.

UniProt's copyrightable database content is covered by [CC BY 4.0](https://www.uniprot.org/help/license). HVIDB compiles multiple public sources and does not state a unified dataset license in its reviewed pages or download table. Preserve source attribution and applicable upstream terms; the project's MIT license does not relicense these data. See [Licensing](licensing.md).

Original database versions, download dates, the complete ID mapping, and unmatched-record lists are unavailable. Checksums fix the distributed snapshot, but a fresh download from today's databases may not reproduce identical rows.

The [Baidu distribution copy](https://pan.baidu.com/s/1Ygk06WfLfOzHvYW7N4y6qQ), extraction code `g9cn`, is the authors' assembled data archive rather than an official HVIDB or UniProt endpoint. The same archive is distributed as `data/archives/data.7z` after resource extraction.

## Distributed tables

Run `python3 scripts/resource_packages.py extract` from the repository root to restore these files:

| Path | Rows | Purpose |
| --- | ---: | --- |
| `data/training/SH_SV_feature_with_vectors.csv` | 33,468 | Pretraining input and historical ESM cache generation |
| `data/training/SH_SV_feature_with_vectors_0.5.csv` | 36,465 | Classification: 24,312 positive and 12,153 negative rows |
| `data/auxiliary/SH_SV_feature_with_vectors_partial.csv` | 26,990 | Partial sequence-feature table |
| `data/auxiliary/human_virus_test.csv` | 7,623 | Labelled pairs with protein identifiers |
| `data/auxiliary/S.csv` | 90,057 | Candidate pairs |
| `data/auxiliary/inference_processed.csv` | 90,052 | Processed candidate pairs |

The two training files contain 28,931 and 34,195 unique sequence pairs, respectively. The second file's row counts describe the original table before deduplication. Detailed statistics are in `data/provenance/training_statistics.json`; content checksums are in `data/manifest.json`. Published paper totals and the available training tables use different counting scopes.

## CSV schema

The canonical columns are `pair_id,human_seq,viral_seq,label`. Sequence column aliases include `S_H,S_V` and `seq_a,seq_b`. `pair_id` is optional; otherwise a stable ID is generated from the sequences. Classification training requires 0/1 labels; prediction can omit labels. Override names with `--human-col`, `--viral-col`, `--label-col`, and `--pair-id-col`.

```csv
pair_id,human_seq,viral_seq,label
example_1,ACDEFGHIKLMNPQRSTVWY,MKTIIALSYIFCLVFAD,1
```

Whitespace and alignment gaps are removed, letters are uppercased, and a trailing stop marker is removed. Ambiguous residues receive zero AAindex descriptors. AAindex features use the first 100 residues and 12 properties per residue, producing 1,200 values with zero padding for short sequences. The packaged AAindex table is used to recompute descriptors rather than trusting embedded CSV vector strings.

```bash
uv run deephvi inspect-data --input data/training/SH_SV_feature_with_vectors_0.5.csv --count
uv run deephvi prepare-data --archive data/archives/data.7z --dest runs/extracted
```

The original archive has a top-level `data/` directory. Extraction records repairs of incomplete final CSV records.

## Deduplication and splitting

`prepare-features` stores sequence features, pair indices, labels, and IDs. `make-splits` stores exact row indices and fingerprints. Always use the same ordered input and configuration for both commands; saved sample splits in `data/splits/cpu_smoke/` include their ordered `pairs.csv`.

| Protocol | Meaning |
| --- | --- |
| `paper` | Held-out 20% test set; three validation folds within the training portion by default |
| `group` | Groups by connected proteins (`union_find`) or virus species (`virus_species`) |
| `server_historical` | Shuffled ten-fold server convention with an inner 90/10 split |
| `legacy` | Earlier unshuffled ten-fold convention; validation also serves as test |

Modern `paper` and `group` commands deduplicate exact full-sequence pairs, preserve the first row, and reject conflicting labels. The classification table becomes 34,195 pairs. Rebuild features and splits together when migrating from older artifacts. Pair-level splits can share individual proteins; connected-component grouping keeps each side's proteins together. Species grouping separates species groups and does not guarantee human-protein separation.

Historical protocols preserve original order and duplicates for comparison; they do not provide the same independent evaluation as modern deduplicated splits. Their staged commands must use the same historical configuration. Every split manifest records protein and pair overlap; `--strict-leakage` rejects cross-partition protein overlap. Prediction retains duplicate input rows and IDs.
