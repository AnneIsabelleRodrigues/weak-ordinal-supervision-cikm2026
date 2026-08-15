import os
import time
from typing import List
import numpy as np
import pandas as pd
from pgmpy.sampling import BayesianModelSampling
from multiprocessing import Pool

from src.causal_algs import ges_search, hill_climb_search_prior, ges_cpdag
from src.graph_fusion import fuse_graphs
from src.metrics import orientation_gain, bic_score, sid
from src.evaluation import evaluate
from src.prepocess import preprocess
from src.constants import BENCHMARKS
from src.treat_graphs import (
    parse_causal_verifications,
    filter_and_map,
    read_groundtruth_graph,
    build_dag_true,
)


def _run_task(args):
    (
        bench,
        bif_path,
        llm_path,
        n_samples,
        rep,
        lambda_,
        theta,
        max_indegree,
        max_iter,
        verbose,
    ) = args

    rows = []

    model = read_groundtruth_graph(bif_path)
    dag_true = build_dag_true(model)

    records = parse_causal_verifications(llm_path)
    llm_graphs = filter_and_map(records, list(model.nodes()))

    prior_weights, _, _ = fuse_graphs(
        llm_graphs,
        theta=theta,
        penalize_disagreement=True,
    )

    data_seed = rep
    sampler = BayesianModelSampling(model)
    data_raw = sampler.forward_sample(size=n_samples, seed=data_seed)
    data = preprocess(data_raw)

    try:
        cpdag_mat = ges_cpdag(data)
    except Exception as e:
        print(e)
        cpdag_mat = None

    ess = max(3, int(np.round(np.log(n_samples))))

    def _record(config, dag, trace=None, elapsed=None):
        m = evaluate(dag, dag_true, config)

        try:
            m["sid"] = sid(dag, dag_true)
        except Exception:
            m["sid"] = None

        try:
            m["bic"] = bic_score(dag, data)
        except Exception:
            m["bic"] = None

        og = orientation_gain(dag, dag_true, cpdag_mat) if cpdag_mat is not None else {}

        row = {
            "benchmark": bench,
            "config": config,
            "rep": rep,
            "n_samples": n_samples,
            "lambda_": lambda_,
            "data_seed": data_seed,
            "elapsed_s": round(elapsed, 3) if elapsed is not None else None,
            **m,
            **og,
        }

        if trace:
            row["score"] = round(trace["score"], 4)
            row["n_edges"] = trace["n_edges"]

        rows.append(row)

    if verbose:
        print(f"[START] {bench} n={n_samples} rep={rep}")

    try:
        t0 = time.perf_counter()
        dag_c, trace_c = hill_climb_search_prior(
            data=data,
            prior_weights=None,
            lambda_=0.0,
            max_indegree=max_indegree,
            max_iter=max_iter,
            equivalent_sample_size=ess,
        )
        _record("HC-C", dag_c, trace_c, time.perf_counter() - t0)
    except Exception:
        pass

    try:
        t0 = time.perf_counter()
        dag_d, trace_d = hill_climb_search_prior(
            data=data,
            prior_weights=prior_weights,
            lambda_=lambda_,
            max_indegree=max_indegree,
            max_iter=max_iter,
            equivalent_sample_size=ess,
        )
        _record("HC-D", dag_d, trace_d, time.perf_counter() - t0)
    except Exception:
        pass

    if verbose:
        print(f"[DONE] {bench} n={n_samples} rep={rep}")

    return rows


def run_e7(
        bench: str = "CHILD",
        sample_sizes: List[int] = (100, 250, 500),
        lambda_: float = 2.0,
        theta: float = 0.1,
        n_reps: int = 5,
        max_indegree: int = 4,
        max_iter: int = 2000,
        n_jobs: int = 32,
        output_csv: str = "results/experiments/e7_orientation_gain.csv",
        verbose: bool = False,
) -> pd.DataFrame:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    if bench not in BENCHMARKS:
        raise ValueError(f"Benchmark '{bench}' unknown.")

    bif_path, llm_path = BENCHMARKS[bench]

    rng = np.random.default_rng(42)
    rep_seeds = rng.choice(10_000, size=n_reps, replace=False).tolist()

    tasks = [
        (
            bench,
            bif_path,
            llm_path,
            n_samples,
            rep,
            lambda_,
            theta,
            max_indegree,
            max_iter,
            verbose,
        )
        for n_samples in sample_sizes
        for rep in rep_seeds
    ]

    if verbose:
        print("=" * 70)
        print(f"E7 | bench={bench} | tasks={len(tasks)} | workers={n_jobs}")
        print("=" * 70)

    with Pool(processes=n_jobs) as pool:
        results_nested = pool.map(_run_task, tasks)

    all_rows = [row for rows in results_nested for row in rows]

    df = pd.DataFrame(all_rows)

    col_order = [
        "benchmark", "config", "rep", "n_samples", "lambda_", "data_seed",
        "shd", "f1", "precision", "recall", "tp", "fp", "fn", "n_pred_edges",
        "sid", "bic",
        "n_ambiguous", "orientation_recall", "orientation_precision",
        "score", "n_edges", "elapsed_s",
    ]
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    df.to_csv(output_csv, index=False)

    if verbose:
        print("=" * 70)
        print(f"Saved: {output_csv} ({len(df)} rows)")
        print("=" * 70)

    return df