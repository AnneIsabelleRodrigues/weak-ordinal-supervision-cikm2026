import math
from typing import Optional

from pgmpy.estimators import HillClimbSearch, BDeu

import networkx as nx
import pandas as pd

from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq

def pc_search(
    data: pd.DataFrame,
    alpha: float = 0.05,
    verbose: bool = True,
) -> nx.DiGraph:

    if verbose:
        print("[Config A - PC Algorithm]")
        print(f"Variables: {len(data.columns)} | alpha: {alpha}")

    columns = list(data.columns)
    X = data.values.astype(float)

    cg = pc(X, alpha=alpha, indep_test=chisq, show_progress=False)

    dag = nx.DiGraph()
    dag.add_nodes_from(columns)

    n = len(columns)
    for i in range(n):
        for j in range(n):
            if i != j:
                if cg.G.graph[i, j] == -1 and cg.G.graph[j, i] == 1:
                    dag.add_edge(columns[i], columns[j])

    if verbose:
        print(f"Done. Edges: {dag.number_of_edges()}")

    return dag

def ges_search(
        data: pd.DataFrame,
        score_func: str = "local_score_BIC",
        verbose: bool = True,
) -> nx.DiGraph:

    if verbose:
        print(f"[Config B - GES | score: {score_func}]")
        print(f"Variables: {len(data.columns)}")

    columns = list(data.columns)
    X = data.values.astype(float)

    record = ges(X, score_func=score_func)

    cpdag_mat = record["G"].graph
    n = len(columns)

    dag = nx.DiGraph()
    dag.add_nodes_from(columns)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if cpdag_mat[i, j] == -1 and cpdag_mat[j, i] == 1:
                dag.add_edge(columns[i], columns[j])
            elif cpdag_mat[i, j] == -1 and cpdag_mat[j, i] == -1 and i < j:
                dag.add_edge(columns[i], columns[j])

    if verbose:
        print(f"Done. Edges: {dag.number_of_edges()}")

    return dag

def ges_cpdag(
    data: pd.DataFrame,
    score_func: str = "local_score_BIC",
    verbose: bool = True,
) -> nx.DiGraph:

    if verbose:
        print(f"[GES - CPDAG | score: {score_func}]")

    columns = list(data.columns)
    X = data.values.astype(float)

    record = ges(X, score_func=score_func)
    cpdag_mat = record["G"].graph

    n = len(columns)
    cpdag = nx.DiGraph()
    cpdag.add_nodes_from(columns)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            if cpdag_mat[i, j] == -1 and cpdag_mat[j, i] == 1:
                cpdag.add_edge(columns[i], columns[j])

            elif cpdag_mat[i, j] == -1 and cpdag_mat[j, i] == -1:
                cpdag.add_edge(columns[i], columns[j])
                cpdag.add_edge(columns[j], columns[i])

    return cpdag


class PriorBDeu(BDeu):

    def __init__(self, data, prior_weights=None, lambda_=1.0,
                 equivalent_sample_size=10):
        super().__init__(
            data=data,
            equivalent_sample_size=equivalent_sample_size
        )
        self.prior_weights = prior_weights or {}
        self.lambda_ = lambda_

    def local_score(self, variable, parents):
        base = super().local_score(variable, parents)

        prior = 0.0
        for p in parents:
            w = self.prior_weights.get((p, variable), 0.50)
            w = min(max(w, 1e-6), 1 - 1e-6)
            prior += math.log(w / (1 - w))

        return base + self.lambda_ * prior


def hill_climb_search_prior(
    data: pd.DataFrame,
    prior_weights: Optional[dict],
    lambda_: float = 1.0,
    max_indegree: int = 4,
    max_iter: int = 2000,
    equivalent_sample_size: int = 10,
):
    scorer = PriorBDeu(
        data=data,
        prior_weights=prior_weights,
        lambda_=lambda_,
        equivalent_sample_size=equivalent_sample_size
    )

    est = HillClimbSearch(data)

    dag = est.estimate(
        scoring_method=scorer,
        max_indegree=max_indegree,
        max_iter=max_iter
    )

    trace = {
        "score": scorer.score(dag),
        "n_edges": len(dag.edges())
    }

    return dag, trace