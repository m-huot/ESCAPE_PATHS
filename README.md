# Escape Funnels

**Escape Funnels** provides an **MCMC and mean-field framework** to study how viral proteins evolve under immune pressure, with a focus on identifying **evolutionary funnels that drive antibody escape**.

## Data

Unzip all folders within repository so that the repository has direct access to the extracted files: lattice/paths.zip, covid/ab_resilience/escape_vectors.zip, covid/paths.zip, covid/paths_nt.zip .

Download the dataset from [Zenodo](https://doi.org/10.5281/zenodo.17016973) and place it in the `mean_field_theory/` directory.  


![Schematic Overview](schematic.png)

---

## Repository Layout

| Path                  | Purpose                                                                                                                                                                     |
|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **lattice/**          | Train a Restricted Boltzmann Machine (RBM) on synthetic lattice protein sequences. Demonstrates how evolution is constrained to narrow funnels in protein space.             |
| **covid/**            | Distill information from ESM-IF into an RBM trained on SARS-CoV-2 RBD sequences. Sample escape paths with MCMC, compare their statistics to observed pandemic mutations, and evaluate antibody resilience. |
| **mean_field_theory/**| Apply mean-field approximations to quantify escape funnels in SARS-CoV-2 RBD sequences. Assess the impact of single antibodies and antibody cocktails on viral evolution.   |
| **path/**             | Utilities for implementing MCMC sampling of escape paths and related workflows.                                                                                           |

---

## Citation

If you use **Escape Funnels** in your research, please cite:

```bibtex

