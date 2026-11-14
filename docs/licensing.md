# Licensing and attribution

[简体中文](zh-CN/licensing.md) · [Documentation index](README.md)

Original DeepHVI code and author-owned classifier, pretraining, and projection weights are covered by the root [MIT license](../LICENSE), only to the extent of the authors' rights. MIT permits use, modification, redistribution, and commercial use subject to retaining the copyright and license notice. Third-party materials keep their own terms.

| Material | Terms | Retained attribution |
| --- | --- | --- |
| Original DeepHVI code and weights | MIT | Root `LICENSE` |
| TMO-Net compatibility classes | MIT | `src/deephvi/compat/LICENSE.TMONet` |
| ESM-2 encoder and tokenizer | Upstream model card specifies MIT | Model card, source links, and [complete upstream license](licenses/LICENSE.ESM2) |
| UniProt copyrightable database content | CC BY 4.0 | UniProt name, source, license link, and description of processing |
| HVIDB interaction compilation | No unified data license stated in reviewed sources | HVIDB citation and original database/literature attribution |

The current public tree and resource archives do not include LucaProt code or weights. Its upstream code is Apache-2.0; any future inclusion must retain its LICENSE, NOTICE, applicable component notices, and modification notices. Fine-tuning or repackaging a third-party model does not remove the base model's obligations.

## Dataset provenance

The authors assembled interaction records from [HVIDB](https://zzdlab.com/hvidb/) and matched protein identifiers to [UniProt](https://www.uniprot.org/) sequences, then organized pairs and computed features. HVIDB's [download table](https://zzdlab.com/hvidb/download/HVIDB_PPIs.txt) includes `Source_Database` and `Pubmed_ID` fields for source tracing.

- HVIDB: retain the database name, source link, and [database paper](https://doi.org/10.1093/bib/bbaa425). This project does not assign MIT or a Creative Commons license to the HVIDB compilation or upstream contents.
- UniProt: copyrightable database content uses [CC BY 4.0](https://www.uniprot.org/help/license). Attribute UniProt and describe DeepHVI's identifier matching, pair processing, and feature derivation.
- Derived tables: the code/model MIT license does not replace external data terms. Reuse must consider applicable original sources and citations.

See [Data](data.md) for file roles and citations, and [third-party notices](../THIRD_PARTY_NOTICES.md) for component details. Missing original database versions and ID mappings limit provenance reconstruction; a checksum establishes file identity rather than redistribution permission.
