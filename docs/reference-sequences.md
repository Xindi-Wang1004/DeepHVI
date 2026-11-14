# Reference sequences

[简体中文](zh-CN/reference-sequences.md) · [Usage](usage.md)

Reference libraries provide natural protein candidates for `retrieve` and `generate`. Both commands rank candidates by the cosine similarity of their sequence embeddings.

| Library | Source |
| --- | --- |
| Human | Complete NCBI RefSeq `GCF_000001405.40` (GRCh38.p14) protein FASTA, including isoforms, patches and alternate loci |
| Viral | All UniProtKB entries matching `taxonomy_id:10239`, including reviewed/unreviewed entries and descendant taxa |

Each library merges identical complete sequences and retains every original accession. Download and preparation records include source versions, sequence counts and file checksums so the same library can be reused across runs.

Run the examples from the repository root after installing dependencies and [restoring the model resources](resources.md).

## Search a public library

Choose a library with `--reference-source`. The first run downloads the FASTA, prepares the catalog, computes embeddings and searches the resulting index:

```bash
uv run deephvi retrieve --resources-dir data \
  --pairs data/examples/inputs/new_pairs.csv \
  --reference-source viral --reference-dir data/reference \
  --top-k 5 --out runs/retrieval.json
```

Use `--reference-source human` for the human library. `--target` defaults to the selected library: `viral` reads queries from `viral_seq`, while `human` reads them from `human_seq`. `generate` accepts the same arguments and returns candidates from the selected library.

Later runs verify and reuse the saved library and index, then encode the queries. JSON results contain ranked hits, the index location and a `reference` object with source details and accession mapping paths. CSV output contains one row per hit; the CLI also prints the index location.

## Prepare libraries during training

Add the reference options to prepare both libraries alongside the full training workflow:

```bash
uv run deephvi reproduce --resources-dir data --config configs/default.yaml \
  --pairs data/training/SH_SV_feature_with_vectors_0.5.csv \
  --reference-source both --reference-dir data/reference --out runs/experiment
```

The workflow builds both reference indices with the sequence encoder used for training features. The folds share these indices, and the run saves their versions, fingerprints and locations in `references.json` and each checkpoint's `reference_libraries` field. Model training uses the supplied pair table; reference libraries supply candidates for subsequent retrieval.

Staged `train-pretrain` and `train-classifier` runs support the same reference options. Classification training inherits reference records from `--pretrain-checkpoint` when no new sources are selected. See [Usage](usage.md#training) for the staged commands.

You can also save the reference settings in YAML:

```yaml
references:
  sources: [human, viral]
  dir: data/reference
  overwrite: false
```

CLI options take precedence over YAML. The default `sources: []` prepares only the pair features needed for training. Select `human`, `viral` or `both` on the command line to add the corresponding reference resources.

To prepare references alongside pair classification, add `--reference-source both --reference-dir data/reference` to `predict`. It writes the index records to `<prediction-file>.references.json`. Reference preparation runs when selected through CLI or YAML; ordinary classification reads its pairs and model resources directly.

## Reuse a training index

Pass a checkpoint from the training run and choose the side to search:

```bash
uv run deephvi generate --checkpoint runs/experiment/fold0/checkpoints/best.pt \
  --pairs data/examples/inputs/new_pairs.csv --target viral --out runs/candidates.json
```

The checkpoint supplies the sequence encoder settings and recorded index. `--target human` selects the human index. Retrieval compares the chosen side's query embeddings with the reference embeddings. The index includes the source and encoding fingerprints used to verify this match.

When moving an index, copy the whole directory and specify its new location with `--checkpoint ... --index NEW_DIR --target human|viral`. Keeping the index files and `pipeline.json` together preserves the training record. For a checkpoint without a recorded library, choose one with `--reference-source`, `--reference` or `--index`.

## Download and prepare separately

The standalone scripts let you prepare libraries before training or process existing FASTA files offline. They use the Python standard library.

Download both sources:

```bash
uv run --frozen python scripts/download_reference_sequences.py \
  --source both --output data/reference/raw
```

Use `--source human` or `--source viral` to download one side. Then prepare each library:

```bash
uv run --frozen python scripts/prepare_reference_sequences.py \
  --source human \
  --fasta data/reference/raw/human/GCF_000001405.40_GRCh38.p14_protein.faa.gz \
  --download-receipt data/reference/raw/human/download.json \
  --output data/reference/prepared/human

uv run --frozen python scripts/prepare_reference_sequences.py \
  --source viral \
  --fasta data/reference/raw/viral/uniprotkb_taxonomy_id_10239.fasta.gz \
  --download-receipt data/reference/raw/viral/download.json \
  --output data/reference/prepared/viral
```

Preparation reads FASTA records in a stream and uses SQLite to deduplicate full sequences. It uppercases residues, removes whitespace, alignment gaps and a terminal stop, and checks sequence characters and accession mappings. Long proteins retain their complete sequences in the exported FASTA.

For an existing `.faa`, `.fasta` or gzipped FASTA, supply its path through `--fasta`. `--download-receipt` associates it with this downloader's source record. Omit that option for other local inputs; their manifests identify the input by its SHA-256.

Use the prepared FASTA directly:

```bash
uv run deephvi retrieve --resources-dir data \
  --pairs data/examples/inputs/new_pairs.csv \
  --reference data/reference/prepared/viral/reference.fasta \
  --index-out runs/viral-reference-index \
  --target viral --top-k 5 --out runs/retrieval.json
```

Later, replace `--reference ... --index-out ...` with `--index runs/viral-reference-index` to reuse that index. Use separate index directories for the human and viral libraries.

## Files and storage

By default, reference data lives in the Git-ignored `data/reference/`. `--reference-dir` overrides YAML `references.dir`. Training and inference share the same cache structure:

| Path | Contents |
| --- | --- |
| `raw/<source>/` | Original compressed FASTA and `download.json`; human downloads also save NCBI's MD5 table |
| `raw/viral/pages/` | Compressed UniProt pages, with progress in `raw/viral/download-progress.json` |
| `prepared/<source>/reference.fasta` | One full sequence per stable `<source>_<sequence_sha256>` candidate ID |
| `prepared/<source>/accessions.tsv` | Candidate IDs, all original accessions and FASTA headers |
| `prepared/<source>/catalog.sqlite3` | Sequence, accession and source metadata tables |
| `prepared/<source>/manifest.json` | Processing policy, source record, sequence statistics and output checksums |
| `indices/<source>/<fingerprint>/` | Encoder-specific index, metadata and `pipeline.json` completion record |

Use a hit's candidate ID to look up all original accessions in `accessions.tsv` or `catalog.sqlite3`. The FASTA description also includes a representative accession.

Automatic index locations depend on the source, preparation policy and encoder settings, including model contents, pooling, truncation and any external projection. Compatible runs share an index; a different encoder gets a separate directory. `--index-out DIR` selects a specific index location.

SQLite keeps preparation memory bounded. Encoding and search load the reference sequences and vectors into memory, so plan disk, RAM and encoding time for the full library. The supplied `server_esm2` encoder uses a 1024-token cap including BOS/EOS, normally up to 1022 residues; the FASTA itself retains the complete sequences.

## Versions and reruns

Completed downloads stay on their saved version. Each rerun checks the FASTA and index fingerprints before reuse. To obtain a new source snapshot, choose a new reference directory.

| Situation | Action |
| --- | --- |
| Download interrupted | Repeat the command. UniProt resumes from saved pages; the human file downloads again if incomplete. |
| UniProt release changes during download | Start in a new directory to obtain a consistent release. |
| Preparation or index build interrupted | Repeat the command to rebuild the incomplete stage. Gallery embedding starts from the beginning. |
| Index needs rebuilding | Use `--overwrite` for `retrieve/generate`, or `--reference-overwrite` for a new training or prediction run. |
| Source files or completed outputs fail validation | Restore the saved version, or prepare a fresh copy in a new directory. |
| Training resumed with `--resume` | Keep its recorded reference libraries and index versions. A change of library belongs in a new training run. |

Keep the original FASTA, source records and indices together for reproducible runs. NCBI downloads use the official MD5; UniProt downloads verify the release and total record count across pages. SHA-256 values identify the saved files throughout preparation and indexing.

Source attribution and terms are listed in [Licensing](licensing.md), [NCBI policies](https://www.ncbi.nlm.nih.gov/home/about/policies/) and [UniProt terms](https://www.uniprot.org/help/license).
