import os
import time
from multiprocessing import Pool
from typing import Dict, List

import numpy as np
import pandas as pd
from pgmpy.sampling import BayesianModelSampling

from src.causal_algs import hill_climb_search_prior
from src.evaluation import evaluate
from src.graph_fusion import fuse_graphs
from src.prepocess import preprocess
from src.constants import BENCHMARKS, ORDINAL_MAPPINGS
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
        mappings,
        lambda_,
        theta,
        base_data_seed,
        max_indegree,
        max_iter,
        verbose,
    ) = args

    rows = []

    model = read_groundtruth_graph(bif_path)
    dag_true = build_dag_true(model)

    records = parse_causal_verifications(llm_path)
    llm_graphs = filter_and_map(records, list(model.nodes()))

    data_seed = base_data_seed + rep
    sampler = BayesianModelSampling(model)
    data_raw = sampler.forward_sample(size=n_samples, seed=data_seed)
    data = preprocess(data_raw)

    ess = max(3, int(np.round(np.log(n_samples))))

    if verbose:
        print(f"[START] {bench} n={n_samples} rep={rep}")

    # ------------------------------------------------------------------
    # HC-Base
    # ------------------------------------------------------------------
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
        elapsed_c = time.perf_counter() - t0

        m_c = evaluate(dag_c, dag_true, "HC-C")
        rows.append({
            "benchmark": bench,
            "mapping": "baseline_C",
            "n_samples": n_samples,
            "rep": rep,
            "data_seed": data_seed,
            **m_c,
            "score": round(trace_c["score"], 4),
            "n_edges": trace_c["n_edges"],
            "elapsed_s": round(elapsed_c, 3),
        })

    except Exception:
        pass

    # ------------------------------------------------------------------
    # HC-LLM
    # ------------------------------------------------------------------
    for map_name, om in mappings.items():
        try:
            prior_weights, _, _ = fuse_graphs(
                llm_graphs,
                theta=theta,
                penalize_disagreement=True,
                ordinal_map=om,
            )

            t0 = time.perf_counter()
            dag_d, trace_d = hill_climb_search_prior(
                data=data,
                prior_weights=prior_weights,
                lambda_=lambda_,
                max_indegree=max_indegree,
                max_iter=max_iter,
                equivalent_sample_size=ess,
            )
            elapsed = time.perf_counter() - t0

            m = evaluate(dag_d, dag_true, "HC-D")
            rows.append({
                "benchmark": bench,
                "mapping": map_name,
                "n_samples": n_samples,
                "rep": rep,
                "data_seed": data_seed,
                "High_weight": om["HIGH"],
                "Medium_weight": om["MEDIUM"],
                "Low_weight": om["LOW"],
                **m,
                "score": round(trace_d["score"], 4),
                "n_edges": trace_d["n_edges"],
                "elapsed_s": round(elapsed, 3),
            })

        except Exception as e:
            pass

    if verbose:
        print(f"[DONE] {bench} n={n_samples} rep={rep}")

    return rows


def run_e5(
        bench: str = "CHILD",
        n_samples: int = 250,
        lambda_: float = 2.0,
        theta: float = 0.1,
        n_reps: int = 3,
        base_data_seed: int = 0,
        max_indegree: int = 4,
        max_iter: int = 2000,
        n_jobs: int = 16,
        output_csv: str = "results/experiments/e5_mapping_sensitivity.csv",
        verbose: bool = False,
) -> pd.DataFrame:

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    mappings = ORDINAL_MAPPINGS

    if bench not in BENCHMARKS:
        raise ValueError(f"Benchmark '{bench}' desconhecido.")

    bif_path, llm_path = BENCHMARKS[bench]

    rng = np.random.default_rng(42)
    rep_seeds = rng.choice(10000, size=n_reps, replace=False).tolist()

    tasks = [
        (
            bench,
            bif_path,
            llm_path,
            n_samples,
            rep,
            mappings,
            lambda_,
            theta,
            base_data_seed,
            max_indegree,
            max_iter,
            verbose,
        )
        for rep in rep_seeds
    ]

    if verbose:
        print("=" * 60)
        print(f"E5 | bench={bench} | n={n_samples} | tasks={len(tasks)} | workers={n_jobs}")
        print("=" * 60)

    with Pool(processes=n_jobs) as pool:
        results_nested = pool.map(_run_task, tasks)

    all_rows = [row for rows in results_nested for row in rows]

    df = pd.DataFrame(all_rows)

    col_order = [
        "benchmark", "mapping", "rep", "n_samples", "data_seed",
        "High_weight", "Medium_weight", "Low_weight",
        "shd", "f1", "precision", "recall", "tp", "fp", "fn", "n_pred_edges",
        "score", "n_edges", "elapsed_s",
    ]
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    df.to_csv(output_csv, index=False)

    if verbose:
        print("=" * 60)
        print(f"Saved: {output_csv} ({len(df)} rows)")
        print("=" * 60)

    return df