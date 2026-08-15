import math
import os
import time
from multiprocessing import Pool
from typing import List

import numpy as np
import pandas as pd
from pgmpy.sampling import BayesianModelSampling

from src.causal_algs import hill_climb_search_prior
from src.constants import BENCHMARKS
from src.evaluation import evaluate
from src.graph_fusion import fuse_graphs
from src.metrics import bic_score, sid
from src.prepocess import preprocess
from src.treat_graphs import (
    build_dag_true,
    filter_and_map,
    parse_causal_verifications,
    read_groundtruth_graph,
)

DEFAULT_WEIGHT_GRID = [0.1, 0.2, 0.3, 0.4, 0.5]

def _run_task(args):
    (
        bench,
        bif_path,
        llm_path,
        n_samples,
        rep,
        default_weight_grid,
        theta,
        lambda_,
        compute_sid_flag,
        compute_bic_flag,
        base_data_seed,
        max_iter,
        verbose,
    ) = args

    rows = []

    model = read_groundtruth_graph(bif_path)
    dag_true  = build_dag_true(model)
    n_nodes = len(list(model.nodes()))
    n_true_edges = dag_true.number_of_edges()

    records = parse_causal_verifications(llm_path)
    llm_graphs = filter_and_map(records, list(model.nodes()))
    prior_weights, _, _ = fuse_graphs(llm_graphs, theta=theta)

    data_seed = base_data_seed + rep
    sampler = BayesianModelSampling(model)
    data_raw  = sampler.forward_sample(size=n_samples, seed=data_seed)
    data = preprocess(data_raw)

    if verbose:
        print(f"[START] {bench} n={n_samples} rep={rep}")

    def _record(config, dag, default_weight, extra=None):
        m = evaluate(dag, dag_true, config)

        if compute_sid_flag:
            try:
                m["sid"] = sid(dag, dag_true)
            except:
                m["sid"] = None

        if compute_bic_flag:
            try:
                m["bic"] = bic_score(dag, data)
            except:
                m["bic"] = None

        row = {
            "benchmark": bench,
            "config": config,
            "default_weight": default_weight,
            "rep": rep,
            "n_samples": n_samples,
            "n_nodes": n_nodes,
            "n_true_edges": n_true_edges,
            "lambda_": lambda_,
            "theta": theta,
        }
        row.update(m)
        if extra:
            row.update(extra)
        rows.append(row)


    try:
        t0 = time.perf_counter()
        dag, trace = hill_climb_search_prior(
            data = data,
            prior_weights = None,
            lambda_ = lambda_,
            max_indegree = 5,
            max_iter = max_iter,
            equivalent_sample_size = max(3, int(np.round(np.log(n_samples)))),
        )
        elapsed = time.perf_counter() - t0

        for dw in default_weight_grid:
            _record("HC-C", dag, default_weight=dw, extra={
                "score": round(trace["score"], 4),
                "n_edges": trace["n_edges"],
                "elapsed_s": round(elapsed, 3),
            })
    except Exception as exc:
        if verbose:
            print(f"[ERR] HC-C rep={rep}: {exc}")

    for dw in default_weight_grid:

        prior_with_default = _PriorWithDefault(prior_weights, default_weight=dw)

        try:
            t0 = time.perf_counter()
            dag, trace = hill_climb_search_prior(
                data = data,
                prior_weights = prior_with_default,
                lambda_ = lambda_,
                max_indegree = 5,
                max_iter = max_iter,
                equivalent_sample_size = max(3, int(np.round(np.log(n_samples)))),
            )
            elapsed = time.perf_counter() - t0

            _record("HC-D", dag, default_weight=dw, extra={
                "score": round(trace["score"], 4),
                "n_edges": trace["n_edges"],
                "elapsed_s": round(elapsed, 3),
            })

        except Exception as exc:
            if verbose:
                print(f"[ERR] HC-D dw={dw} rep={rep}: {exc}")

    if verbose:
        print(f"[DONE] {bench} n={n_samples} rep={rep}")

    return rows


class _PriorWithDefault(dict):

    def __init__(self, prior_weights: dict, default_weight: float = 0.5):
        super().__init__(prior_weights)
        self._default_weight = default_weight

    def get(self, key, default=None):
        return super().get(key, self._default_weight)


def run_e10(
    benchmarks: List[str],
    sample_sizes: List[int],
    default_weight_grid: List[float] = DEFAULT_WEIGHT_GRID,
    theta: float = 0.0,
    lambda_: float  = 1.0,
    n_reps: int = 20,
    base_data_seed: int = 0,
    max_iter: int = 2000,
    n_jobs: int = 32,
    compute_sid: bool = True,
    compute_bic: bool = True,
    output_csv: str = "results/experiments/e10_default_weight_raw.csv",
    verbose: bool = False,
) -> pd.DataFrame:

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    rng   = np.random.default_rng(42)
    tasks = []

    for bench in benchmarks:
        if bench not in BENCHMARKS:
            raise ValueError(f"Benchmark '{bench}' unknown.")

        bif_path, llm_path = BENCHMARKS[bench]

        for n_samples in sample_sizes:
            for rep in rng.choice(10000, size=n_reps, replace=False).tolist():
                tasks.append((
                    bench,
                    bif_path,
                    llm_path,
                    n_samples,
                    rep,
                    default_weight_grid,
                    theta,
                    lambda_,
                    compute_sid,
                    compute_bic,
                    base_data_seed,
                    max_iter,
                    verbose,
                ))

    if verbose:
        print(f"Tasks: {len(tasks)} | Workers: {n_jobs}")
        print(f"default_weight_grid: {default_weight_grid}")
        print(f"Benchmarks: {benchmarks}")

    with Pool(processes=n_jobs) as pool:
        results_nested = pool.map(_run_task, tasks)

    all_rows   = [row for rows in results_nested for row in rows]
    df         = pd.DataFrame(all_rows)

    col_order = [
        "benchmark", "config", "default_weight", "rep", "n_samples",
        "n_nodes", "n_true_edges",
        "shd", "f1", "precision", "recall",
        "tp", "fp", "fn",
        "sid", "bic",
        "theta", "lambda_", "score", "n_edges", "elapsed_s",
    ]
    col_order = [c for c in col_order if c in df.columns]
    df        = df[col_order]

    df.to_csv(output_csv, index=False)

    if verbose:
        print(f"Saved: {output_csv}  ({len(df)} rows)")

    return df