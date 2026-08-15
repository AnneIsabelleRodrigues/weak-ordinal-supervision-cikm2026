import itertools
import os
import time
from multiprocessing import Pool
from typing import List

import numpy as np
import pandas as pd
from pgmpy.sampling import BayesianModelSampling

from src.causal_algs import pc_search, ges_search, hill_climb_search_prior
from src.evaluation import evaluate
from src.graph_fusion import fuse_graphs
from src.metrics import sid, bic_score
from src.prepocess import preprocess
from src.utils import build_oracle_prior
from src.treat_graphs import (
    parse_causal_verifications,
    filter_and_map,
    read_groundtruth_graph,
    build_dag_true,
)
from src.constants import BENCHMARKS

def _run_task(args):
    (
        bench,
        bif_path,
        llm_path,
        n_samples,
        rep,
        lambdas,
        thetas,
        run_pc,
        run_ges,
        run_hc_c,
        run_hc_d,
        run_hc_e,
        compute_sid,
        compute_bic,
        base_data_seed,
        max_iter,
        verbose,
    ) = args

    rows = []

    model = read_groundtruth_graph(bif_path)
    dag_true = build_dag_true(model)

    n_true_edges = dag_true.number_of_edges()
    n_nodes = len(list(model.nodes()))

    llm_nodes = list(model.nodes())

    oracle_prior = build_oracle_prior(dag_true, p_correct=0.9)

    data_seed = base_data_seed + rep
    sampler = BayesianModelSampling(model)
    data_raw = sampler.forward_sample(size=n_samples, seed=data_seed)
    data = preprocess(data_raw)

    def _record(config, dag, extra=None):
        m = evaluate(dag, dag_true, config)

        if compute_sid:
            try:
                m["sid"] = sid(dag, dag_true)
            except Exception:
                m["sid"] = None

        if compute_bic:
            try:
                m["bic"] = bic_score(dag, data)
            except Exception:
                m["bic"] = None

        row = {
            "benchmark": bench,
            "config": config,
            "rep": rep,
            "n_samples": n_samples,
            "n_nodes": n_nodes,
            "n_true_edges": n_true_edges,
        }
        row.update(m)
        if extra:
            row.update(extra)
        rows.append(row)

    if verbose:
        print(f"[START] {bench} n={n_samples} rep={rep}")

    # -------------------------------------------------------
    # PC
    # -------------------------------------------------------
    if run_pc:
        try:
            t0  = time.perf_counter()
            dag = pc_search(data, verbose=False)
            elapsed = time.perf_counter() - t0

            _record("PC", dag, {
                "theta": None,
                "lambda_": None,
                "acceptance_rate": None,
                "elapsed_s": round(elapsed, 3),
                "data_seed": data_seed,
                "algorithm_seed": None,
            })
        except Exception:
            pass

    # -------------------------------------------------------
    # GES
    # -------------------------------------------------------
    if run_ges:
        try:
            t0  = time.perf_counter()
            dag = ges_search(data, verbose=False)
            elapsed = time.perf_counter() - t0

            _record("GES", dag, {
                "theta": None,
                "lambda_": None,
                "acceptance_rate": None,
                "elapsed_s": round(elapsed, 3),
                "data_seed": data_seed,
                "algorithm_seed": None,
            })
        except Exception:
            pass

    # ==========================================================
    # HC
    # ==========================================================

    for lam, theta in itertools.product(lambdas, thetas):

        llm_graphs = filter_and_map(
            parse_causal_verifications(llm_path),
            llm_nodes,
        )
        prior_weights, _, _ = fuse_graphs(llm_graphs, theta=theta)

        if run_hc_c:
            try:
                t0 = time.perf_counter()
                dag, trace = hill_climb_search_prior(
                    data = data,
                    prior_weights = None,
                    lambda_ = lam,
                    max_indegree = 5,
                    max_iter = max_iter,
                    equivalent_sample_size = max(3, int(np.round(np.log(n_samples)))),
                )
                elapsed = time.perf_counter() - t0

                _record("HC-C", dag, {
                    "theta": None,
                    "lambda_": lam,
                    "max_indegree": 5,
                    "score": round(trace["score"], 4),
                    "n_edges": trace["n_edges"],
                    "elapsed_s": round(elapsed, 3),
                    "algorithm_seed": data_seed,
                })
            except Exception:
                pass

        if run_hc_d:
            try:
                t0 = time.perf_counter()
                dag, trace = hill_climb_search_prior(
                    data = data,
                    prior_weights = prior_weights,
                    lambda_ = lam,
                    max_indegree = 5,
                    max_iter = max_iter,
                    equivalent_sample_size = max(3, int(np.round(np.log(n_samples)))),
                )
                elapsed = time.perf_counter() - t0

                _record("HC-D", dag, {
                    "theta": theta,
                    "lambda_": lam,
                    "max_indegree": 5,
                    "score": round(trace["score"], 4),
                    "n_edges": trace["n_edges"],
                    "elapsed_s": round(elapsed, 3),
                    "algorithm_seed": data_seed,
                })
            except Exception:
                pass

        if run_hc_e:
            try:
                t0 = time.perf_counter()
                dag, trace = hill_climb_search_prior(
                    data = data,
                    prior_weights = oracle_prior,
                    lambda_ = lam,
                    max_indegree = 5,
                    max_iter = max_iter,
                    equivalent_sample_size = max(3, int(np.round(np.log(n_samples)))),
                )
                elapsed = time.perf_counter() - t0

                _record("HC-E", dag, {
                    "theta": None,
                    "lambda_": lam,
                    "max_indegree": 5,
                    "score": round(trace["score"], 4),
                    "n_edges": trace["n_edges"],
                    "elapsed_s": round(elapsed, 3),
                    "algorithm_seed": data_seed,
                })
            except Exception:
                pass

    if verbose:
        print(f"[DONE]  {bench} n={n_samples} rep={rep}")

    return rows

def run_e1_e4(
        benchmark: str = "CHILD",
        sample_sizes: List[int]  = (50, 100, 200, 400, 800, 1500),
        lambdas: List[float] = (0.5, 1.0, 2.0, 5.0),
        thetas: List[float] = (0.1,),
        run_pc: bool = True,
        run_ges: bool = True,
        run_hc_c: bool = True,
        run_hc_d: bool = True,
        run_hc_e: bool = True,
        compute_sid: bool = True,
        compute_bic: bool = True,
        n_reps: int  = 5,
        base_data_seed: int = 0,
        max_iter: int  = 2000,
        n_jobs: int  = 32,
        output_csv: str  = "results/experiments/experiment_results.csv",
        verbose: bool = False,
) -> pd.DataFrame:

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    if benchmark not in BENCHMARKS:
        raise ValueError(f"Benchmark '{benchmark}' desconhecido.")

    bif_path, llm_path = BENCHMARKS[benchmark]

    rng   = np.random.default_rng(42)
    tasks = []

    for n_samples in sample_sizes:
        for rep in rng.choice(10000, size=n_reps, replace=False).tolist():
            tasks.append((
                benchmark,
                bif_path,
                llm_path,
                n_samples,
                rep,
                lambdas,
                thetas,
                run_pc,
                run_ges,
                run_hc_c,
                run_hc_d,
                run_hc_e,
                compute_sid,
                compute_bic,
                base_data_seed,
                max_iter,
                verbose,
            ))

    if verbose:
        print("=" * 70)
        print(f"Tasks paralelas: {len(tasks)}")
        print(f"Workers: {n_jobs}")
        print("=" * 70)

    with Pool(processes=n_jobs) as pool:
        results_nested = pool.map(_run_task, tasks)

    all_rows = [row for rows in results_nested for row in rows]

    df_results = pd.DataFrame(all_rows)

    col_order = [
        "benchmark", "config", "rep", "n_samples",
        "n_nodes", "n_true_edges",
        "shd", "f1", "precision", "recall",
        "tp", "fp", "fn", "n_pred_edges",
        "sid", "bic",
        "theta", "lambda_",
        "acceptance_rate",
        "data_seed", "mcmc_seed",
        "elapsed_s",
    ]
    col_order  = [c for c in col_order if c in df_results.columns]
    df_results = df_results[col_order]

    df_results.to_csv(output_csv, index=False)

    if verbose:
        print("=" * 70)
        print(f"Saved: {output_csv} ({len(df_results)} rows)")
        print("=" * 70)

    return df_results