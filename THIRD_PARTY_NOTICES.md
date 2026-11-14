# Third-party notices

[English](THIRD_PARTY_NOTICES.md) · [简体中文](THIRD_PARTY_NOTICES.zh-CN.md)

## TMO-Net compatibility

`src/deephvi/compat/tmonet.py` retains the minimal inference methods required to
convert trusted legacy DeepHVI checkpoints whose pickle class names refer to
`TMONet.model.TMO_Net_model`. It is adapted from
[FengAoWang/TMONet](https://github.com/FengAoWang/TMONet), copyright (c) 2024
FengAoWang, MIT licensed. The full notice is preserved alongside the code in
[src/deephvi/compat/LICENSE.TMONet](src/deephvi/compat/LICENSE.TMONet).
The compatibility classes support conversion, not the original training scripts.

## DeepHVI reference data

`tests/fixtures/aaindex_legacy_examples.txt` contains only the embedded sequence
and feature literals extracted from the original DeepHVI prediction example.
They provide an independent check of the recovered `aaindex12.csv` table;
no legacy executable code is needed by the tests. The DeepHVI authors' example
code is covered by the root MIT license; underlying external data retain their
applicable source terms.

## External data and models

The authors identify the DeepHVI trained checkpoints as their own work. These
weights are covered by the root MIT license to the extent of their rights;
this does not change any third-party base-model license.

ESM-2 weights and tokenizer files originate from
[`facebook/esm2_t6_8M_UR50D`](https://huggingface.co/facebook/esm2_t6_8M_UR50D).
The distributed model card specifies MIT. The upstream copyright and complete
license are preserved in [LICENSE.ESM2](docs/licenses/LICENSE.ESM2), copyright
Meta Platforms, Inc. and affiliates. Source:
https://github.com/facebookresearch/esm/blob/main/LICENSE (checked 2026-11-11).
The extracted `data/models/esm2/` preserves the model card, and the resource
inventory records content checksums.

LucaProt's retained original `LICENSE` specifies Apache-2.0, and its `NOTICE`
attributes copyright to Alibaba Group and lists additional third-party notices.
LucaProt code and weights are not included in the current public tree or resource
archives. If they are added later, retain their LICENSE, NOTICE, relevant component
notices and modification notices. The repository's MIT license does not replace them.

The dataset combines human–virus interaction records from
[HVIDB](https://zzdlab.com/hvidb/) and protein sequences from
[UniProt](https://www.uniprot.org/), matched using protein identifiers.
UniProt's copyrightable database content is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), per its
[official terms](https://www.uniprot.org/help/license).
HVIDB aggregates publicly available interaction records from multiple sources;
its downloadable table includes Source_Database and Pubmed_ID provenance fields.
No unified data license is stated on the reviewed HVIDB pages or in that table.
DeepHVI retains source attribution and citations without assigning a new MIT or
CC license to the HVIDB compilation or its upstream content.
See [data provenance](docs/data.md) and [license scope](docs/licensing.md).

The Baidu archive is a distribution copy of the assembled data, not the original
database source. Data and model archives are included in this repository;
they are not all covered by one blanket license. The paper PDF and local
historical development copies are excluded from the public tree.
