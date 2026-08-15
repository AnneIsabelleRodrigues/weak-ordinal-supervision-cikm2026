import os
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.graph_fusion import fuse_graphs
from src.constants import BENCHMARKS
from src.treat_graphs import (
    parse_causal_verifications,
    filter_and_map,
    read_groundtruth_graph,
    build_dag_true,
)

def _dominant_label(labels: List[str]) -> str:
    return max(set(labels), key=labels.count)


def _ordinal_ordering(
        prior_weights: Dict[Tuple[str, str], float],
        edge_labels: Dict[Tuple[str, str], List[str]],
        dag_true: nx.DiGraph,
        benchmark: str = "",
) -> pd.DataFrame:

    rows = []

    for label in ("HIGH", "MEDIUM", "LOW"):
        label_edges = [
            e for e, lbls in edge_labels.items()
            if label in lbls and e in prior_weights
        ]
        if not label_edges:
            continue

        n_direct = sum(1 for e in label_edges if dag_true.has_edge(*e))

        n_ancestral = sum(
            1 for e in label_edges
            if dag_true.has_node(e[0]) and dag_true.has_node(e[1])
            and nx.has_path(dag_true, e[0], e[1])
        )

        n_no_path = sum(
            1 for e in label_edges
            if not (dag_true.has_node(e[0]) and dag_true.has_node(e[1])
                    and nx.has_path(dag_true, e[0], e[1]))
        )
        n_indirect = n_ancestral - n_direct

        rows.append({
            "benchmark": benchmark,
            "label": label,
            "n_edges": len(label_edges),
            "n_direct": n_direct,
            "n_ancestral": n_ancestral,
            "n_indirect": n_indirect,
            "n_no_path": n_no_path,
            "precision_direct": round(n_direct / len(label_edges), 4),
            "precision_ancestral":round(n_ancestral / len(label_edges), 4),
            "error_rate_structural": round(n_no_path / len(label_edges), 4),
            "error_rate_distance": round(n_indirect / len(label_edges), 4),
        })

    return pd.DataFrame(rows)


def _llm_calibration(
        prior_weights: Dict[Tuple[str, str], float],
        edge_labels: Dict[Tuple[str, str], List[str]],
        dag_true: nx.DiGraph,
        benchmark: str = "",
        n_bins: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    edges = [e for e in prior_weights if e in edge_labels]
    if not edges:
        return pd.DataFrame(), pd.DataFrame()

    p_vals = np.array([prior_weights[e] for e in edges])
    correct = np.array([1 if dag_true.has_edge(*e) else 0 for e in edges])
    ancestral = np.array([
        1 if (dag_true.has_node(e[0]) and dag_true.has_node(e[1])
              and nx.has_path(dag_true, e[0], e[1])) else 0
        for e in edges
    ])

    rho_direct, pval_direct = spearmanr(p_vals, correct)
    rho_anc, pval_anc = spearmanr(p_vals, ancestral)

    df_corr = pd.DataFrame([{
        "benchmark": benchmark,
        "spearman_rho_direct": round(rho_direct, 4),
        "spearman_p_direct": round(pval_direct, 4),
        "spearman_rho_ancestral": round(rho_anc, 4),
        "spearman_p_ancestral": round(pval_anc, 4),
        "n_edges": len(edges),
        "base_rate_direct": round(correct.mean(), 4),
        "base_rate_ancestral": round(ancestral.mean(), 4),
    }])

    bins = np.linspace(0, 1, n_bins + 1)
    ece_rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p_vals >= lo) & (p_vals < hi if i < n_bins - 1 else p_vals <= hi)
        if mask.sum() == 0:
            continue
        mean_conf = float(p_vals[mask].mean())
        acc_direct = float(correct[mask].mean())
        acc_ancestral = float(ancestral[mask].mean())
        ece_rows.append({
            "benchmark": benchmark,
            "bin_low": round(lo, 2),
            "bin_high": round(hi, 2),
            "n_edges": int(mask.sum()),
            "mean_p_ij": round(mean_conf, 4),
            "accuracy_direct": round(acc_direct, 4),
            "accuracy_ancestral":round(acc_ancestral, 4),
            "gap_direct": round(mean_conf - acc_direct, 4),
            "gap_ancestral": round(mean_conf - acc_ancestral, 4),
        })

    df_ece = pd.DataFrame(ece_rows)
    return df_corr, df_ece


def run_e2(
        bench: str = "CHILD",
        theta: float = 0.0,
        output_csv_ordering: str = "results/experiments/e2_ordinal_ordering.csv",
        output_csv_calibration: str = "results/experiments/e2_calibration.csv",
        output_csv_ece: str = "results/experiments/e2_ece.csv",
        verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    os.makedirs(os.path.dirname(output_csv_ordering), exist_ok=True)

    all_ordering = []
    all_calibration = []
    all_ece = []

    if bench not in BENCHMARKS:
        raise ValueError(f"Benchmark '{bench}' desconhecido.")

    bif_path, llm_path = BENCHMARKS[bench]

    if verbose:
        print(f"\n[E2] {bench.upper()}")

    model = read_groundtruth_graph(bif_path)
    dag_true = build_dag_true(model)

    records = parse_causal_verifications(llm_path)
    llm_graphs = filter_and_map(records, list(model.nodes()))

    prior_weights, _, edge_labels = fuse_graphs(
        llm_graphs, theta=theta, penalize_disagreement=True
    )

    df_ord = _ordinal_ordering(prior_weights, edge_labels, dag_true, bench)
    all_ordering.append(df_ord)

    if verbose:
        print("  Ordering:")
        print(df_ord[["label", "n_edges", "precision_direct",
                       "precision_ancestral", "error_rate_structural"]].to_string(index=False))

    df_corr, df_ece = _llm_calibration(prior_weights, edge_labels, dag_true, bench)
    if not df_corr.empty:
        all_calibration.append(df_corr)
    if not df_ece.empty:
        all_ece.append(df_ece)

    if verbose and not df_corr.empty:
        row = df_corr.iloc[0]
        print(f"  Calibration: ρ_direct={row['spearman_rho_direct']:.3f} "
              f"(p={row['spearman_p_direct']:.3f})  "
              f"ρ_ancestral={row['spearman_rho_ancestral']:.3f} "
              f"(p={row['spearman_p_ancestral']:.3f})")

    df_ordering = pd.concat(all_ordering, ignore_index=True) if all_ordering else pd.DataFrame()
    df_calibration = pd.concat(all_calibration, ignore_index=True) if all_calibration else pd.DataFrame()
    df_ece = pd.concat(all_ece, ignore_index=True) if all_ece else pd.DataFrame()

    df_ordering.to_csv(output_csv_ordering, index=False)
    df_calibration.to_csv(output_csv_calibration, index=False)
    df_ece.to_csv(output_csv_ece, index=False)

    if verbose:
        print(f"\n[E2] Saved:")
        print(f" {output_csv_ordering}")
        print(f" {output_csv_calibration}")
        print(f" {output_csv_ece}")

    return df_ordering, df_calibration, df_ece