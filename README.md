# Weak Ordinal Supervision for Causal Structure Learning from Clinical Narratives

[![Paper](https://img.shields.io/badge/CIKM-2026-blue)](https://doi.org/10.1145/3799682.3840941)
[![License: CC_BY_4.0](https://img.shields.io/badge/License-CC%20BY%204.0-green.svg)](LICENSE)

Official repository for the CIKM 2026 paper **"Weak Ordinal Supervision for Causal Structure Learning from Clinical Narratives."**

This repository contains the code used to study how large language models can provide **weak ordinal supervision** for causal structure learning from clinical narratives. Rather than treating LLM outputs as hard causal judgments, the proposed method injects ordinal plausibility signals into score-based structure learning through soft regularization.

## Overview

The repository is organized around two components:

- `llm-causal-discovery/`: LLM- and RAG-based processing for extracting candidate variables and causal signals from clinical case reports.
- `fusion-and-evaluation/`: causal discovery, fusion, and evaluation pipelines used in the experimental study.

At a high level, the workflow is:

1. Clinical case reports are processed to extract variables and candidate causal relations.
2. An LLM assigns weak ordinal preferences over candidate edges.
3. These preferences are incorporated into score-based causal structure learning.
4. The resulting graphs are evaluated against benchmark structures across multiple settings.

## Repository Layout

```text
llm-causal-discovery/
  config/                      # runtime and environment configuration
  llm/                         # model-facing pipeline components
  rag/                         # retrieval-augmented generation components
  services/                    # orchestration and formatting helpers
  main.py                      # entry point for case-report processing

fusion-and-evaluation/
  config/                      # evaluation paths and settings
  results/                     # generated evaluation artifacts
  src/                         # experiments, metrics, and statistics
  main.py                      # entry point for experimental evaluation
```

## Benchmark Networks

Ground-truth Bayesian network structures are drawn from the **bnlearn Bayesian Network Repository**, which provides standard benchmark networks and multiple export formats commonly used in structure learning research.

Repository: https://www.bnlearn.com/bnrepository/

## Installation

Install dependencies for each module separately.

```bash
cd llm-causal-discovery
pip install -r requirements.txt
```

```bash
cd fusion-and-evaluation
pip install -r requirements.txt
```

## Running the Pipelines

Run the clinical narrative processing pipeline:

```bash
cd llm-causal-discovery
python main.py --model_name <MODEL_NAME> --benchmark <BENCHMARK>
```

Run the evaluation pipeline:

```bash
cd fusion-and-evaluation
python main.py
```

The accepted model names and benchmark settings are defined in the codebase configuration.

## Clinical Case Reports

The study uses clinical case reports indexed through PubMed and PubMed Central. The repository does **not** redistribute these source articles in the public release; they are referenced by identifier.

| Benchmark | PMID records | PMCID records |
| --- | --- | --- |
| `ALARM` | [16704627](https://pubmed.ncbi.nlm.nih.gov/16704627/), [20552629](https://pubmed.ncbi.nlm.nih.gov/20552629/), [31102782](https://pubmed.ncbi.nlm.nih.gov/31102782/), [39140386](https://pubmed.ncbi.nlm.nih.gov/39140386/) | [PMC270717](https://pmc.ncbi.nlm.nih.gov/articles/PMC270717/), [PMC2882006](https://pmc.ncbi.nlm.nih.gov/articles/PMC2882006/), [PMC3622308](https://pmc.ncbi.nlm.nih.gov/articles/PMC3622308/), [PMC6126365](https://pmc.ncbi.nlm.nih.gov/articles/PMC6126365/), [PMC6379930](https://pmc.ncbi.nlm.nih.gov/articles/PMC6379930/), [PMC8070294](https://pmc.ncbi.nlm.nih.gov/articles/PMC8070294/), [PMC8164485](https://pmc.ncbi.nlm.nih.gov/articles/PMC8164485/), [PMC10270465](https://pmc.ncbi.nlm.nih.gov/articles/PMC10270465/), [PMC11080974](https://pmc.ncbi.nlm.nih.gov/articles/PMC11080974/), [PMC11088200](https://pmc.ncbi.nlm.nih.gov/articles/PMC11088200/), [PMC11846252](https://pmc.ncbi.nlm.nih.gov/articles/PMC11846252/), [PMC12958415](https://pmc.ncbi.nlm.nih.gov/articles/PMC12958415/) |
| `CHILD` | [5644153](https://pubmed.ncbi.nlm.nih.gov/5644153/), [21173945](https://pubmed.ncbi.nlm.nih.gov/21173945/), [26760414](https://pubmed.ncbi.nlm.nih.gov/26760414/), [31511259](https://pubmed.ncbi.nlm.nih.gov/31511259/), [32323659](https://pubmed.ncbi.nlm.nih.gov/32323659/), [33400859](https://pubmed.ncbi.nlm.nih.gov/33400859/), [34397695](https://pubmed.ncbi.nlm.nih.gov/34397695/), [35981751](https://pubmed.ncbi.nlm.nih.gov/35981751/), [36308451](https://pubmed.ncbi.nlm.nih.gov/36308451/), [37081543](https://pubmed.ncbi.nlm.nih.gov/37081543/), [37250675](https://pubmed.ncbi.nlm.nih.gov/37250675/), [37903312](https://pubmed.ncbi.nlm.nih.gov/37903312/), [38043022](https://pubmed.ncbi.nlm.nih.gov/38043022/), [38289811](https://pubmed.ncbi.nlm.nih.gov/38289811/), [38552044](https://pubmed.ncbi.nlm.nih.gov/38552044/) | [PMC10603519](https://pmc.ncbi.nlm.nih.gov/articles/PMC10603519/) |
| `HEPAR II` | [18528934](https://pubmed.ncbi.nlm.nih.gov/18528934/), [31623992](https://pubmed.ncbi.nlm.nih.gov/31623992/), [32299307](https://pubmed.ncbi.nlm.nih.gov/32299307/), [33731401](https://pubmed.ncbi.nlm.nih.gov/33731401/), [35395817](https://pubmed.ncbi.nlm.nih.gov/35395817/) | [PMC3371426](https://pmc.ncbi.nlm.nih.gov/articles/PMC3371426/), [PMC3680334](https://pmc.ncbi.nlm.nih.gov/articles/PMC3680334/), [PMC3937526](https://pmc.ncbi.nlm.nih.gov/articles/PMC3937526/), [PMC4839135](https://pmc.ncbi.nlm.nih.gov/articles/PMC4839135/), [PMC5557326](https://pmc.ncbi.nlm.nih.gov/articles/PMC5557326/), [PMC8106784](https://pmc.ncbi.nlm.nih.gov/articles/PMC8106784/), [PMC8273458](https://pmc.ncbi.nlm.nih.gov/articles/PMC8273458/), [PMC10068984](https://pmc.ncbi.nlm.nih.gov/articles/PMC10068984/), [PMC10394758](https://pmc.ncbi.nlm.nih.gov/articles/PMC10394758/), [PMC11181619](https://pmc.ncbi.nlm.nih.gov/articles/PMC11181619/), [PMC13088203](https://pmc.ncbi.nlm.nih.gov/articles/PMC13088203/) |
| `GIST` | [15645649](https://pubmed.ncbi.nlm.nih.gov/15645649/), [15750270](https://pubmed.ncbi.nlm.nih.gov/15750270/), [15815855](https://pubmed.ncbi.nlm.nih.gov/15815855/), [16086122](https://pubmed.ncbi.nlm.nih.gov/16086122/), [16126886](https://pubmed.ncbi.nlm.nih.gov/16126886/), [17873750](https://pubmed.ncbi.nlm.nih.gov/17873750/), [18066563](https://pubmed.ncbi.nlm.nih.gov/18066563/), [18239881](https://pubmed.ncbi.nlm.nih.gov/18239881/), [18442217](https://pubmed.ncbi.nlm.nih.gov/18442217/), [23601997](https://pubmed.ncbi.nlm.nih.gov/23601997/), [23645766](https://pubmed.ncbi.nlm.nih.gov/23645766/), [23681598](https://pubmed.ncbi.nlm.nih.gov/23681598/), [23749608](https://pubmed.ncbi.nlm.nih.gov/23749608/), [24052432](https://pubmed.ncbi.nlm.nih.gov/24052432/), [24118605](https://pubmed.ncbi.nlm.nih.gov/24118605/), [26023595](https://pubmed.ncbi.nlm.nih.gov/26023595/) | - |

## Reproducibility

This repository provides the implementation used for the experimental study. Reproducibility depends on:

- access to the required benchmark data, case-report and guidelines inputs ,
- valid model/API configuration for the LLM pipeline,
- the dependency sets declared in each module.

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{carvalho2026weak,
  author    = {Carvalho, Anne I. R. and Oliveira, Gabriel P. and Duarte, Luis A. and
               Birchal, Eduardo and Teixeira, Matheus C. and Jacob, Gustavo and
               Silva, Ana Paula Couto da and Pappa, Gisele L. and Lacerda, Anisio},
  title     = {Weak Ordinal Supervision for Causal Structure Learning from Clinical Narratives},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  series    = {CIKM '26},
  year      = {2026},
  publisher = {ACM},
  address   = {New York, NY, USA},
  pages     = {11}
}
```
