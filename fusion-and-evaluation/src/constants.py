from typing import Dict

GRAPH_DEF = {'HEPAR': {'n_nodes': 70, 'n_edges': 123, 'max_indegree': 6, 'n_params' : 1453},
             'CHILD': {'n_nodes': 20, 'n_edges': 25, 'max_indegree': 2, 'n_params' : 230},
             'ALARM': {'n_nodes': 37, 'n_edges': 46, 'max_indegree': 4, 'n_params' : 509}}

BENCHMARKS = {
    "CHILD": ("data/graphs/child.bif.gz", "data/raw/child_verifications.json"),
    "ALARM": ("data/graphs/alarm.bif.gz", "data/raw/alarm_verifications.json"),
    "HEPAR": ("data/graphs/hepar.bif.gz", "data/raw/hepar_verifications.json"),
}

ORDINAL_MAPPINGS: Dict[str, Dict[str, float]] = {
    "tight": {"HIGH": 0.95, "MEDIUM": 0.75, "LOW": 0.55},
    "default": {"HIGH": 0.90, "MEDIUM": 0.60, "LOW": 0.30},
    "wide": {"HIGH": 0.90, "MEDIUM": 0.50, "LOW": 0.10},
    "binary_soft": {"HIGH": 0.90, "MEDIUM": 0.90, "LOW": 0.30},
    "uniform": {"HIGH": 0.90, "MEDIUM": 0.90, "LOW": 0.90},
}