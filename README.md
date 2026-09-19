# Escape Funnels

**Escape Funnels** provides an **MCMC and mean-field framework** to study how viral proteins evolve under immune pressure, with a focus on identifying **evolutionary funnels that drive antibody escape**.

## Data

Unzip all folders within repository so that the repository has direct access to the extracted files: lattice/paths.zip, covid/paths.zip, covid/paths_nt.zip .

Download the dataset results_scripts from [Zenodo](https://doi.org/10.5281/zenodo.19154385) and place it in the `mean_field_theory/` directory.  

Clone PGM repository (https://github.com/jertubiana/PGM) inside main folder.


![Schematic Overview](schematic.png)

---

## Repository Layout

| Path                  | Purpose                                                                                                                                                                     |
|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **lattice/**          | Train a Restricted Boltzmann Machine (RBM) on synthetic lattice protein sequences. Demonstrates how evolution is constrained to narrow funnels in protein space.             |
| **covid/**            | Distill information from ESM-IF into an RBM trained on SARS-CoV-2 RBD sequences. Sample escape paths with MCMC, compare their statistics to observed pandemic mutations, and evaluate antibody resilience. |
| **mean_field_theory/**| Apply mean-field approximations to quantify escape funnels. Assess the impact of single antibodies and antibody cocktails on viral evolution.   |
| **path/**             | Utilities for implementing MCMC sampling of escape paths and related workflows.                                                                                           |

---

## Citation

If you use **Escape Funnels** in your research, please cite:

```bibtex
@article{Huot_funnels,
  author  = {Marian Huot and Dianzhuo Wang and Eugene Shakhnovich and Rémi Monasson and Simona Cocco},
  title   = {Constrained evolutionary funnels shape viral immune escape},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {123},
  number  = {16},
  pages   = {e2536956123},
  year    = {2026},
  doi     = {10.1073/pnas.2536956123},
  url     = {https://www.pnas.org/doi/abs/10.1073/pnas.2536956123}
}
