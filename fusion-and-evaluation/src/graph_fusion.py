from typing import Dict, List, Optional, Tuple

import logging
import pandas as pd
import numpy as np

from collections import defaultdict
from scipy.stats import entropy as scipy_entropy
from src.constants import ORDINAL_MAPPINGS

logger = logging.getLogger(__name__)

ORDINAL_MAP: Dict[str, float] = ORDINAL_MAPPINGS["default"]

def _parse_graphs(
        list_of_dicts: List[Dict],
        ordinal_map: Dict[str, float],
) -> List[Dict[Tuple[str, str], str]]:
    parsed = []

    for doc_graph in list_of_dicts:
        doc_edges: Dict[Tuple[str, str], str] = {}

        for cause, effects in doc_graph.items():
            cause_norm = cause.strip()

            for item in effects:
                effect_norm = item["effect"].strip()
                label = item["plausibility"]

                if label not in ordinal_map:
                    continue

                edge = (cause_norm, effect_norm)

                if edge not in doc_edges or ordinal_map[label] > ordinal_map[doc_edges[edge]]:
                    doc_edges[edge] = label

        parsed.append(doc_edges)

    return parsed


def _label_entropy(labels: List[str], om: Dict[str, float]) -> float:
    counts = np.array([labels.count(k) for k in om], dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = np.clip(counts / total, 1e-9, 1.0)
    H = scipy_entropy(probs)
    H_max = np.log(len(om))
    return float(H / H_max) if H_max > 0 else 0.0


def fuse_graphs(
        list_of_dicts: List[Dict],
        theta: float = 0.0,
        alpha: float = 1.0,
        penalize_disagreement: bool = True,
        penalize_direction: bool = True,
        penalize_entropy: bool = True,
        smooth_exp_disagreement: float = 0.1,
        smooth_exp_entropy: float = 0.3,
        ordinal_map: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[Tuple[str, str], float], pd.DataFrame, Dict[Tuple[str, str], List[str]]]:

    om = ordinal_map or ORDINAL_MAP

    parsed = _parse_graphs(list_of_dicts, om)
    n = len(parsed)

    if n == 0:
        empty_df = pd.DataFrame(columns=[
            "cause", "effect", "frequency", "n_docs",
            "mean_label", "std_label", "entropy", "bwd_frequency",
            "p_HIGH", "p_MEDIUM", "p_LOW",
            "w_freq", "w_disagreement", "w_direction", "w_entropy",
            "p_ij",
        ])
        return {}, empty_df, {}

    edge_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    edge_labels: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for doc_edges in parsed:
        for edge, label in doc_edges.items():
            edge_counts[edge] += 1
            edge_labels[edge].append(label)

    reverse_freq: Dict[Tuple[str, str], float] = {
        edge: edge_counts.get((edge[1], edge[0]), 0) / n
        for edge in edge_counts
    }

    prior_weights: Dict[Tuple[str, str], float] = {}
    stats_rows = []

    for edge, count in edge_counts.items():
        freq = count / n

        if freq < theta:
            continue

        labels = edge_labels[edge]
        weights = [om[lbl] for lbl in labels]

        mean_weight = float(np.mean(weights))
        std_weight = float(np.std(weights))
        H_norm = _label_entropy(labels, om)
        bwd_freq = reverse_freq[edge]

        w_freq = float(freq ** alpha)
        w_disagreement = (1.0 - std_weight) ** smooth_exp_disagreement if penalize_disagreement else 1.0
        w_direction = (1.0 - bwd_freq) if penalize_direction else 1.0
        w_entropy  = (1.0 - H_norm) ** smooth_exp_entropy if penalize_entropy else 1.0

        p_ij = mean_weight * w_disagreement * w_direction * w_entropy
        p_ij = float(np.clip(p_ij, 0.05, 0.95))

        prior_weights[edge] = p_ij

        stats_rows.append({
            "cause": edge[0],
            "effect": edge[1],
            "frequency": round(freq, 3),
            "n_docs": count,
            "mean_weight": round(mean_weight, 3),
            "std_label": round(std_weight, 3),
            "entropy": round(H_norm, 3),
            "bwd_frequency": round(bwd_freq, 3),
            "p_HIGH": round(labels.count("HIGH") / len(labels), 3),
            "p_MEDIUM": round(labels.count("MEDIUM") / len(labels), 3),
            "p_LOW": round(labels.count("LOW") / len(labels), 3),
            "w_freq": round(w_freq, 3),
            "w_disagreement": round(w_disagreement, 3),
            "w_direction": round(w_direction, 3),
            "w_entropy": round(w_entropy, 3),
            "p_ij": round(p_ij, 3),
        })

    edge_stats = (
        pd.DataFrame(stats_rows)
        .sort_values("p_ij", ascending=False)
        .reset_index(drop=True)
        if stats_rows
        else pd.DataFrame(columns=[
            "cause", "effect", "frequency", "n_docs",
            "mean_label", "std_label", "entropy", "bwd_frequency",
            "p_HIGH", "p_MEDIUM", "p_LOW",
            "w_freq", "w_disagreement", "w_direction", "w_entropy",
            "p_ij",
        ])
    )

    return prior_weights, edge_stats, dict(edge_labels)