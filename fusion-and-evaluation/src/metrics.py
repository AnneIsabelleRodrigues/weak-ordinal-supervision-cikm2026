import math
from itertools import product
from typing import Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd

from src.evaluation import evaluate

def sid(dag_pred: nx.DiGraph, dag_true: nx.DiGraph) -> int:

    nodes = list(dag_true.nodes())

    anc_true = {v: nx.ancestors(dag_true, v) for v in nodes}
    pa_true = {v: set(dag_true.predecessors(v)) for v in nodes}
    pa_pred = {v: set(dag_pred.predecessors(v)) if dag_pred.has_node(v) else set()
               for v in nodes}

    count = 0
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            if i not in anc_true[j]:
                if dag_pred.has_node(i) and dag_pred.has_node(j):
                    if nx.has_path(dag_pred, i, j):
                        count += 1
            else:
                desc_i_true = nx.descendants(dag_true, i) | {i}
                pa_j_relevant_true = pa_true[j] - desc_i_true

                if dag_pred.has_node(j):
                    desc_i_pred = (nx.descendants(dag_pred, i) | {i}
                                   if dag_pred.has_node(i) else {i})
                    pa_j_relevant_pred = pa_pred[j] - desc_i_pred
                else:
                    pa_j_relevant_pred = set()

                if pa_j_relevant_true != pa_j_relevant_pred:
                    count += 1

    return count


def bic_score(
        dag: nx.DiGraph,
        data: pd.DataFrame,
        cardinalities: Optional[Dict[str, int]] = None,
) -> float:

    if cardinalities is None:
        cardinalities = {col: int(data[col].nunique()) for col in data.columns}

    n = len(data)
    total_ll = 0.0
    total_k = 0

    for node in dag.nodes():
        if node not in data.columns:
            continue
        parents = list(dag.predecessors(node))
        ri = cardinalities.get(node, int(data[node].nunique()))
        qi = int(np.prod([cardinalities.get(p, 2) for p in parents])) if parents else 1
        k = qi * (ri - 1)
        total_k += k

        if not parents:
            counts = data[node].value_counts().reindex(range(ri), fill_value=0).values
            n_total = counts.sum()
            for c in counts:
                if c > 0:
                    total_ll += c * math.log(c / n_total)
        else:
            for _, group in data.groupby(parents):
                n_q = len(group)
                if n_q == 0:
                    continue
                nc = group[node].value_counts().reindex(range(ri), fill_value=0).values
                for c in nc:
                    if c > 0:
                        total_ll += c * math.log(c / n_q)

    penalty = 0.5 * total_k * math.log(n) if n > 0 and total_k > 0 else 0.0
    return round(total_ll - penalty, 4)


def cpdag_from_dag(dag: nx.DiGraph) -> nx.Graph:

    undirected = nx.Graph(dag)
    cpdag = nx.DiGraph()
    cpdag.add_nodes_from(dag.nodes())

    compelled = set()

    for z in dag.nodes():
        parents_z = list(dag.predecessors(z))
        for x, y in product(parents_z, parents_z):
            if x >= y:
                continue
            if not undirected.has_edge(x, y):
                compelled.add((x, z))
                compelled.add((y, z))

    for u, v in dag.edges():
        cpdag.add_edge(u, v, oriented=(u, v) in compelled, direction=(u, v))

    for u, v in dag.edges():
        if (u, v) not in compelled and (v, u) not in compelled:
            cpdag[u][v]["oriented"] = False

    return cpdag


def orientation_gain(
        dag_pred: nx.DiGraph,
        dag_true: nx.DiGraph,
        dag_stat: nx.DiGraph,
) -> Dict:

    cpdag_stat = cpdag_from_dag(dag_stat)

    ambiguous = set()
    for u, v in cpdag_stat.edges():
        if not cpdag_stat[u][v].get("oriented", True):
            ambiguous.add(frozenset([u, v]))

    if not ambiguous:
        return {
            "n_ambiguous": 0,
            "n_correctly_oriented": 0,
            "n_incorrectly_oriented": 0,
            "n_still_ambiguous": 0,
            "orientation_precision": None,
            "orientation_recall": None,
        }

    n_correct = 0
    n_wrong = 0
    n_still_amb = 0

    for pair in ambiguous:
        pair_list = list(pair)
        u, v = pair_list[0], pair_list[1]
        true_fwd = dag_true.has_edge(u, v)
        true_bwd = dag_true.has_edge(v, u)
        pred_fwd = dag_pred.has_edge(u, v)
        pred_bwd = dag_pred.has_edge(v, u)

        if not pred_fwd and not pred_bwd:
            n_still_amb += 1
        elif (pred_fwd and true_fwd) or (pred_bwd and true_bwd):
            n_correct += 1
        else:
            n_wrong += 1

    denom_prec = n_correct + n_wrong
    denom_rec = len(ambiguous)

    return {
        "n_ambiguous": len(ambiguous),
        "n_correctly_oriented": n_correct,
        "n_incorrectly_oriented": n_wrong,
        "n_still_ambiguous": n_still_amb,
        "orientation_precision": round(n_correct / denom_prec, 4) if denom_prec > 0 else None,
        "orientation_recall": round(n_correct / denom_rec, 4)  if denom_rec > 0 else None,
    }


def compute_all_metrics(
        dag_pred: nx.DiGraph,
        dag_true: nx.DiGraph,
        data: pd.DataFrame,
        config_label: str = "",
        dag_stat_for_cpdag: Optional[nx.DiGraph] = None,
        compute_sid: bool = True,
        compute_bic: bool = True,
        compute_orient: bool = False,
) -> Dict:

    base = evaluate(dag_pred, dag_true, config_label)

    if compute_sid:
        try:
            base["sid"] = sid(dag_pred, dag_true)
        except Exception:
            base["sid"] = None

    if compute_bic:
        try:
            base["bic"] = bic_score(dag_pred, data)
        except Exception:
            base["bic"] = None

    if compute_orient and dag_stat_for_cpdag is not None:
        try:
            og = orientation_gain(dag_pred, dag_true, dag_stat_for_cpdag)
            base.update(og)
        except Exception:
            pass

    return base