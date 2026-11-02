# Research code (as run)

This folder contains the scripts that produced the results of the revised paper, published as they were run. Folder paths mirror the original project layout so that internal paths remain consistent:

- `scripts/python/`: training scripts for the multi-scale model, its aggregation variants and the benchmark models.
- `revision_work/scripts/`: shared analysis code (input construction, inference, portfolios, shrinkage, interventions, ETF-holdings reconstruction).
- `revision_work/outputs/<step>/`: code specific to each revision analysis, with its pre-specified protocol (`PROTOCOL_*.md`) where one exists.
- `docs/`: protocols for the multi-seed inference and the router diagnostics.

Where several versions of a script exist (`name.py`, `name_v2.py`, ...), the highest version is the one used for the reported results. Superseded drafts were removed; a lower version remains only when a later script imports or calls it.

`revision/INDEX.md` lists the entry points for every analysis in the paper. Data, model checkpoints, predictions and bootstrap draws are not included.
