import os
import random
import time
from typing import Dict, List, Tuple
import numpy as np

from multiprocessing import Pool
import pandas as pd
from pgmpy.sampling import BayesianModelSampling

from src.causal_algs import hill_climb_search_prior
from src.evaluation import evaluate
from src.graph_fusion import fuse_graphs
from src.metrics import bic_score, sid
from src.prepocess import preprocess
from src.constants import BENCHMARKS
from src.treat_graphs import (
    parse_causal_verifications,
    filter_and_map,
    read_groundtruth_graph,
    build_dag_true,
)


def _build_adversarial_prior(
        dag_true,
        prior_weights_true: Dict[Tuple[str, str], float],
        mode: str = "inverted",
        noise_rate: float = 0.5,
        rng: random.Random = None,
) -> Dict[Tuple[str, str], float]:

    nodes = list(dag_true.nodes())
    rng = rng or random.Random(0)

    if mode == "inverted":
        adversarial = {}
        for u in nodes:
            for v in nodes:
                if u == v:
                    continue
                if dag_true.has_edge(u, v):
                    adversarial[(u, v)] = 0.1
                else:
                    adversarial[(u, v)] = 0.9
        return adversarial

    elif mode == "noisy":
        adversarial = {}
        for edge, p in prior_weights_true.items():
            if rng.random() < noise_rate:
                adversarial[edge] = round(1.0 - p, 3)
            else:
                adversarial[edge] = p
        return adversarial

    elif mode == "absent":
        return {edge: 0.5 for edge in prior_weights_true}

    else:
        raise ValueError(f"Mode: {mode}. Use 'inverted', 'noisy' ou 'absent'.")


def _run_task(args):
    (
        bench,
        bif_path,
        llm_path,
        n_samples,
        rep,
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

    prior_true, _, _ = fuse_graphs(
        llm_graphs,
        theta=theta,
        penalize_disagreement=True,
    )

    data_seed = 0 + rep
    py_rng = random.Random(base_data_seed + rep + 500)

    sampler = BayesianModelSampling(model)
    data_raw = sampler.forward_sample(size=n_samples, seed=data_seed)
    data = preprocess(data_raw)

    prior_inverted = _build_adversarial_prior(
        dag_true, prior_true, mode="inverted", rng=py_rng,
    )
    prior_noisy50 = _build_adversarial_prior(
        dag_true, prior_true, mode="noisy", noise_rate=0.5, rng=py_rng,
    )
    prior_absent = _build_adversarial_prior(
        dag_true, prior_true, mode="absent", rng=py_rng,
    )

    conditions = [
        ("HC-C", None, 0.0),
        ("HC-D-true", prior_true, lambda_),
        ("HC-D-inverted", prior_inverted, lambda_),
        ("HC-D-noisy50", prior_noisy50, lambda_),
        ("HC-D-absent", prior_absent, lambda_),
    ]

    ess = max(3, int(np.round(np.log(n_samples))))

    if verbose:
        print(f"[START] {bench} n={n_samples} rep={rep}")

    for cond_name, pw, lam in conditions:
        try:
            t0 = time.perf_counter()
            dag, trace = hill_climb_search_prior(
                data=data,
                prior_weights=pw,
                lambda_=lam,
                max_indegree=max_indegree,
                max_iter=max_iter,
                equivalent_sample_size=ess,
            )
            elapsed = time.perf_counter() - t0

            m = evaluate(dag, dag_true, cond_name)

            try:
                sid_val = sid(dag, dag_true)
            except Exception:
                sid_val = None

            try:
                bic_val = bic_score(dag, data)
            except Exception:
                bic_val = None

            rows.append({
                "benchmark": bench,
                "condition": cond_name,
                "n_samples": n_samples,
                "rep": rep,
                "n_prior_edges": 0 if pw is None else len(pw),
                **m,
                "sid": sid_val,
                "bic": bic_val,
                "score": round(trace["score"], 4),
                "n_edges": trace["n_edges"],
                "elapsed_s": round(elapsed, 3),
                "data_seed": data_seed,
            })

        except Exception:
            pass

    if verbose:
        print(f"[DONE] {bench} n={n_samples} rep={rep}")

    return rows

def run_e8(
        bench: str = "child",
        sample_sizes: List[int] = (50, 100, 250, 500),
        lambda_: float = 2.0,
        theta: float = 0.1,
        n_reps: int = 3,
        base_data_seed: int = 0,
        max_indegree: int = 4,
        max_iter: int = 2000,
        n_jobs: int = 32,
        output_csv: str = "results/experiments/e8_prior_conflict.csv",
        verbose: bool = False,
) -> pd.DataFrame:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    if bench not in BENCHMARKS:
        raise ValueError(f"Benchmark '{bench}' unknown.")

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
            lambda_,
            theta,
            base_data_seed,
            max_indegree,
            max_iter,
            verbose,
        )
        for n_samples in sample_sizes
        for rep in rep_seeds
    ]

    if verbose:
        print("=" * 60)
        print(f"E8 | bench={bench} | tasks={len(tasks)} | workers={n_jobs}")
        print("=" * 60)

    with Pool(processes=n_jobs) as pool:
        results_nested = pool.map(_run_task, tasks)

    all_rows = [row for rows in results_nested for row in rows]

    df = pd.DataFrame(all_rows)

    col_order = [
        "benchmark", "condition", "rep", "n_samples", "n_prior_edges", "data_seed",
        "shd", "f1", "precision", "recall", "tp", "fp", "fn", "n_pred_edges",
        "sid", "bic",
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