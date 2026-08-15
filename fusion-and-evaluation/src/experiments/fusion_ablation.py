import itertools
import os
import time
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pgmpy.sampling import BayesianModelSampling
from sklearn.metrics import roc_auc_score

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


def _prior_auroc(
    prior_weights: Dict[Tuple[str, str], float],
    dag_true,
    unseen_score: float = 0.5,
) -> Optional[float]:

    true_edges = set(dag_true.edges())
    seen_edges = set(prior_weights.keys())

    scores, labels = [], []

    for edge, p_ij in prior_weights.items():
        scores.append(p_ij)
        labels.append(1 if edge in true_edges else 0)

    for _ in true_edges - seen_edges:
        scores.append(unseen_score)
        labels.append(1)

    if len(set(labels)) < 2:
        return None

    return float(roc_auc_score(labels, scores))

def _prior_ece(
    prior_weights: Dict[Tuple[str, str], float],
    dag_true,
    n_bins: int = 10,
) -> float:

    true_edges = set(dag_true.edges())
    edges      = list(prior_weights.keys())

    if not edges:
        return float("nan")

    scores = np.array([prior_weights[e] for e in edges])
    labels = np.array([1 if e in true_edges else 0 for e in edges])

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece       = 0.0
    n_total   = len(scores)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() == 0:
            continue
        acc  = labels[mask].mean()
        conf = scores[mask].mean()
        ece += (mask.sum() / n_total) * abs(acc - conf)

    return float(ece)


_ABLATION_FLAGS: List[Dict] = [
    {
        "name": "Mean-Simple",
        "penalize_disagreement": False,
        "penalize_direction": False,
        "penalize_entropy": False,
        "alpha": 1.0,
        "note": "baseline — only mean_weight",
    },
]

for pd_flag, pdir_flag, pe_flag in itertools.product([False, True], repeat=3):
    label_parts = []
    if pd_flag:  label_parts.append("Dis")
    if pdir_flag: label_parts.append("Dir")
    if pe_flag:  label_parts.append("Ent")
    name = "Full" if (pd_flag and pdir_flag and pe_flag) else ("+".join(label_parts) or "None")

    _ABLATION_FLAGS.append({
        "name":  name,
        "penalize_disagreement": pd_flag,
        "penalize_direction": pdir_flag,
        "penalize_entropy": pe_flag,
        "alpha": 1.0,
        "note": "",
    })

_ABLATION_FLAGS = [
    c for c in _ABLATION_FLAGS
    if not (
        c["name"] != "Mean-Simple"
        and not c["penalize_disagreement"]
        and not c["penalize_direction"]
        and not c["penalize_entropy"]
    )
]

_ABLATION_FLAGS.append({
    "name": "Full-NoSmooth",
    "penalize_disagreement": True,
    "penalize_direction": True,
    "penalize_entropy": True,
    "alpha": 1.0,
    "smooth_exp_disagreement": 1.0,
    "smooth_exp_entropy": 1.0,
    "note": "justifica expoentes 0.1/0.3 — sem suavização",
})


_ALPHA_CONFIGS: List[Dict] = [
    {
        "name": f"Full-α={a}",
        "penalize_disagreement": True,
        "penalize_direction": True,
        "penalize_entropy": True,
        "alpha": a,
        "note": "",
    }
    for a in [0.25, 0.5, 1.0, 2.0, 4.0]
]

def _run_task(args):
    (
        bench,
        bif_path,
        llm_path,
        n_samples,
        rep,
        fusion_configs,
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
    dag_true = build_dag_true(model)
    n_nodes = len(list(model.nodes()))
    n_true_edges = dag_true.number_of_edges()

    records = parse_causal_verifications(llm_path)
    llm_graphs = filter_and_map(records, list(model.nodes()))

    data_seed = base_data_seed + rep
    sampler = BayesianModelSampling(model)
    data_raw = sampler.forward_sample(size=n_samples, seed=data_seed)
    data = preprocess(data_raw)

    if verbose:
        print(f"[START] {bench} n={n_samples} rep={rep}")

    for cfg in fusion_configs:

        try:
            prior_weights, _, _ = fuse_graphs(
                llm_graphs,
                theta = theta,
                alpha = cfg["alpha"],
                penalize_disagreement = cfg["penalize_disagreement"],
                penalize_direction = cfg["penalize_direction"],
                penalize_entropy = cfg["penalize_entropy"],
                smooth_exp_disagreement = cfg.get("smooth_exp_disagreement", 0.1),
                smooth_exp_entropy = cfg.get("smooth_exp_entropy", 0.3),
            )

            auroc = _prior_auroc(prior_weights, dag_true)
            ece   = _prior_ece(prior_weights, dag_true)

            t0 = time.perf_counter()
            dag, trace = hill_climb_search_prior(
                data = data,
                prior_weights = prior_weights,
                lambda_ = lambda_,
                max_indegree = 5,
                max_iter = max_iter,
                equivalent_sample_size = max(3, int(np.round(np.log(n_samples)))),
            )
            elapsed = time.perf_counter() - t0

            m = evaluate(dag, dag_true, cfg["name"])

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
                "fusion_config": cfg["name"],
                "penalize_disagreement": cfg["penalize_disagreement"],
                "penalize_direction": cfg["penalize_direction"],
                "penalize_entropy": cfg["penalize_entropy"],
                "alpha": cfg["alpha"],
                "note": cfg["note"],
                "rep": rep,
                "n_samples": n_samples,
                "n_nodes": n_nodes,
                "n_true_edges": n_true_edges,
                "n_prior_edges": len(prior_weights),
                "auroc": round(auroc, 4) if auroc is not None else None,
                "ece": round(ece, 4),
                "elapsed_s": round(elapsed, 3),
                "lambda_": lambda_,
                "theta": theta,
            }
            row.update(m)
            rows.append(row)

        except Exception as exc:
            if verbose:
                print(f"  [ERR] {cfg['name']} rep={rep}: {exc}")

    if verbose:
        print(f"[DONE]  {bench} n={n_samples} rep={rep}")

    return rows

def _run_e9(
    benchmark: str,
    fusion_configs: List[Dict],
    sample_sizes: List[int],
    theta: float,
    lambda_: float,
    n_reps: int,
    base_data_seed: int,
    max_iter: int,
    n_jobs: int,
    compute_sid: bool,
    compute_bic: bool,
    output_csv: str,
    verbose: bool,
) -> pd.DataFrame:

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    if benchmark not in BENCHMARKS:
        raise ValueError(f"Benchmark '{benchmark}' unknown.")

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
                fusion_configs,
                theta,
                lambda_,
                compute_sid,
                compute_bic,
                base_data_seed,
                max_iter,
                verbose,
            ))

    if verbose:
        print(f"  Tasks: {len(tasks)} | Workers: {n_jobs}")
        print(f"  Configs: {[c['name'] for c in fusion_configs]}")

    with Pool(processes=n_jobs) as pool:
        results_nested = pool.map(_run_task, tasks)

    all_rows = [row for rows in results_nested for row in rows]
    df = pd.DataFrame(all_rows)

    col_order = [
        "benchmark", "fusion_config", "rep", "n_samples",
        "n_nodes", "n_true_edges", "n_prior_edges",
        "penalize_disagreement", "penalize_direction", "penalize_entropy", "alpha",
        "auroc", "ece",
        "shd", "f1", "precision", "recall",
        "tp", "fp", "fn",
        "sid", "bic",
        "theta", "lambda_", "elapsed_s", "note",
    ]
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    df.to_csv(output_csv, index=False)

    if verbose:
        print(f"  Saved: {output_csv} ({len(df)} rows)")

    return df

def run_e9a(
    bench: str,
    sample_sizes: List[int],
    theta: float = 0.0,
    lambda_: float = 1.0,
    n_reps: int = 20,
    base_data_seed: int = 0,
    max_iter: int = 2000,
    n_jobs: int = 32,
    compute_sid: bool = True,
    compute_bic: bool = True,
    output_csv: str = "results/experiments/e9a_ablation_raw.csv",
    verbose: bool = False,
) -> pd.DataFrame:

    print(f"\n[E9a] Binary Ablation - {bench} - {n_reps} reps")
    return _run_e9(
        benchmark = bench,
        fusion_configs = _ABLATION_FLAGS,
        sample_sizes = sample_sizes,
        theta = theta,
        lambda_ = lambda_,
        n_reps = n_reps,
        base_data_seed = base_data_seed,
        max_iter = max_iter,
        n_jobs = n_jobs,
        compute_sid = compute_sid,
        compute_bic = compute_bic,
        output_csv = output_csv,
        verbose = verbose,
    )


def run_e9b(
    bench: str,
    sample_sizes: List[int],
    theta: float  = 0.0,
    lambda_: float = 1.0,
    n_reps: int = 20,
    base_data_seed: int = 0,
    max_iter: int = 2000,
    n_jobs: int = 32,
    compute_sid: bool = True,
    compute_bic: bool = True,
    output_csv: str = "results/experiments/e9b_alpha_raw.csv",
    verbose: bool = False,
) -> pd.DataFrame:

    print(f"\n[E9b] Sensibility of α  - {bench} - {n_reps} reps")
    return _run_e9(
        benchmark = bench,
        fusion_configs = _ALPHA_CONFIGS,
        sample_sizes = sample_sizes,
        theta = theta,
        lambda_ = lambda_,
        n_reps = n_reps,
        base_data_seed = base_data_seed,
        max_iter = max_iter,
        n_jobs = n_jobs,
        compute_sid = compute_sid,
        compute_bic = compute_bic,
        output_csv  = output_csv,
        verbose = verbose,
    )