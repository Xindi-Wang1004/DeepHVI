# Licensing and attribution

[简体中文](zh-CN/licensing.md) · [Documentation index](README.md)

Original DeepHVI code and author-owned classifier, pretraining, and projection weights use the root [MIT license](../LICENSE). MIT permits use, modification, redistribution, and commercial use with the copyright and license notice retained. Third-party models and data follow the terms listed below.

| Material | Terms | Retained attribution |
| --- | --- | --- |
| Original DeepHVI code and weights | MIT | Root `LICENSE` |
| TMO-Net compatibility classes | MIT | `src/deephvi/compat/LICENSE.TMONet` |
| ESM-2 encoder and tokenizer | Upstream model card specifies MIT | Model card, source links, and [complete upstream license](licenses/LICENSE.ESM2) |
| UniProt copyrightable database content | CC BY 4.0 | UniProt name, source, license link, and description of processing |
| HVIDB interaction compilation | No unified data license stated in reviewed sources | HVIDB citation and original database/literature attribution |

## Dataset provenance

The authors assembled interaction records from [HVIDB](https://zzdlab.com/hvidb/) and matched protein identifiers to [UniProt](https://www.uniprot.org/) sequences, then organized pairs and computed features. HVIDB's [download table](https://zzdlab.com/hvidb/download/HVIDB_PPIs.txt) includes `Source_Database` and `Pubmed_ID` fields for source tracing.

- HVIDB: retain the database name, source link, and [database paper](https://doi.org/10.1093/bib/bbaa425), along with the attribution and terms of the original sources.
- UniProt: copyrightable database content uses [CC BY 4.0](https://www.uniprot.org/help/license). Attribute UniProt and describe DeepHVI's identifier matching, pair processing, and feature derivation.
- Derived tables: follow the applicable source data terms and retain the original citations.

See [Data](data.md) for file roles, available provenance and citations, and [third-party notices](../THIRD_PARTY_NOTICES.md) for component details.
