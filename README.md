# Weak Ordinal Supervision for Causal Structure Learning from Clinical Narratives

*Weak Ordinal Supervision for Causal Structure Learning from Clinical Narratives*

*Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM 2026)*

[Paper] · [DOI] · [ACM CIKM 2026]

## Overview
This paper presents a systematic evaluation of a different perspective on incorporating Large Language Models (LLMs) into causal discovery. 
Rather than treating LLM outputs as definitive causal judgments, we use them as weak ordinal supervision integrated into the Hill-Climbing score through a soft regularization mechanism. 
Our framework extracts candidate causal relationships from clinical narratives and assigns ordinal plausibility labels using a domain-specialized LLM. 
We evaluate the approach on three established Bayesian network benchmarks, comparing Hill Climbing with and without ordinal priors across varying sample sizes.

## Reproducibility
All experiments reported in the paper can be reproduced using the
scripts and configurations provided in this repository.

## Citation

If you use this work in your research, please cite:

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