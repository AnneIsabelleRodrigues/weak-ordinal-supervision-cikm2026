from typing import Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def shd(dag_pred: nx.DiGraph, dag_true: nx.DiGraph) -> int:
    pred_edges = set(dag_pred.edges())
    true_edges = set(dag_true.edges())

    missing = true_edges - pred_edges
    extra = pred_edges - true_edges
    reversed_ = {(v, u) for (u, v) in missing if (v, u) in extra}

    return len(missing) + len(extra) - len(reversed_)


def f1_score_dag(dag_pred: nx.DiGraph, dag_true: nx.DiGraph) -> float:
    pred_edges = set(dag_pred.edges())
    true_edges = set(dag_true.edges())

    if not pred_edges and not true_edges:
        return 1.0
    if not pred_edges or not true_edges:
        return 0.0

    tp = len(pred_edges & true_edges)
    precision = tp / len(pred_edges)
    recall = tp / len(true_edges)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(
        dag_pred: nx.DiGraph,
        dag_true: nx.DiGraph,
        config_label: str = "",
) -> Dict:
    pred_edges = set(dag_pred.edges())
    true_edges = set(dag_true.edges())
    tp = len(pred_edges & true_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)
    precision = tp / len(pred_edges) if pred_edges else 0.0
    recall = tp / len(true_edges) if true_edges else 0.0

    return {
        "config": config_label,
        "shd": shd(dag_pred, dag_true),
        "f1": round(f1_score_dag(dag_pred, dag_true), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_pred_edges": dag_pred.number_of_edges(),
        "n_true_edges": dag_true.number_of_edges(),
    }


def compute_shd_gap(df: pd.DataFrame) -> pd.DataFrame:

    has_reps = "rep" in df.columns
    group = ["benchmark", "n_samples"]

    configs = set(df["config"].dropna().unique())

    families = sorted(
        set(cfg.rsplit("-", 1)[0] for cfg in configs if "-" in cfg)
    )

    all_rows = []

    for family in families:

        cfg_c = f"{family}-C"
        cfg_d = f"{family}-D"

        if not {cfg_c, cfg_d}.issubset(configs):
            continue

        c = (
            df[df["config"] == cfg_c]
            .groupby(group)["shd"]
            .mean()
            .rename("shd_c")
        )

        d = (
            df[df["config"] == cfg_d]
            .groupby(group)["shd"]
            .mean()
            .rename("shd_d")
        )

        gap = pd.concat([c, d], axis=1).reset_index()

        gap["config_family"] = family
        gap["config_c"] = cfg_c
        gap["config_d"] = cfg_d

        gap["shd_gap"] = gap["shd_c"] - gap["shd_d"]

        if has_reps:

            ci_rows = []
            rng = np.random.default_rng(0)

            for key, grp_c in df[df["config"] == cfg_c].groupby(group):

                grp_d = df[
                    (df["config"] == cfg_d)
                    & (df["benchmark"] == key[0])
                    & (df["n_samples"] == key[1])
                    ]

                vals_c = grp_c["shd"].values
                vals_d = grp_d["shd"].values

                n = min(len(vals_c), len(vals_d))

                if n < 2:
                    ci_rows.append({
                        "benchmark": key[0],
                        "n_samples": key[1],
                        "gap_ci_low": np.nan,
                        "gap_ci_high": np.nan
                    })
                    continue

                boots = [
                    vals_c[rng.integers(0, n, n)].mean()
                    - vals_d[rng.integers(0, n, n)].mean()
                    for _ in range(1000)
                ]

                ci_rows.append({
                    "benchmark": key[0],
                    "n_samples": key[1],
                    "gap_ci_low": float(np.percentile(boots, 2.5)),
                    "gap_ci_high": float(np.percentile(boots, 97.5)),
                })

            gap = gap.merge(
                pd.DataFrame(ci_rows),
                on=group,
                how="left"
            )

        num_cols = gap.select_dtypes(include="number").columns
        gap[num_cols] = gap[num_cols].round(4)

        all_rows.append(gap)

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)
    
def compute_sid_gap(df: pd.DataFrame) -> pd.DataFrame:

    has_reps = "rep" in df.columns
    group = ["benchmark", "n_samples"]

    configs = set(df["config"].dropna().unique())

    families = sorted(
        set(cfg.rsplit("-", 1)[0] for cfg in configs if "-" in cfg)
    )

    all_rows = []

    for family in families:

        cfg_c = f"{family}-C"
        cfg_d = f"{family}-D"

        if not {cfg_c, cfg_d}.issubset(configs):
            continue

        c = (
            df[df["config"] == cfg_c]
            .groupby(group)["sid"]
            .mean()
            .rename("sid_c")
        )

        d = (
            df[df["config"] == cfg_d]
            .groupby(group)["sid"]
            .mean()
            .rename("sid_d")
        )

        gap = pd.concat([c, d], axis=1).reset_index()

        gap["config_family"] = family
        gap["config_c"] = cfg_c
        gap["config_d"] = cfg_d

        gap["sid_gap"] = gap["sid_c"] - gap["sid_d"]

        if has_reps:

            ci_rows = []
            rng = np.random.default_rng(0)

            for key, grp_c in df[df["config"] == cfg_c].groupby(group):

                grp_d = df[
                    (df["config"] == cfg_d)
                    & (df["benchmark"] == key[0])
                    & (df["n_samples"] == key[1])
                ]

                vals_c = grp_c["sid"].values
                vals_d = grp_d["sid"].values

                n = min(len(vals_c), len(vals_d))

                if n < 2:
                    ci_rows.append({
                        "benchmark": key[0],
                        "n_samples": key[1],
                        "gap_ci_low": np.nan,
                        "gap_ci_high": np.nan
                    })
                    continue

                boots = [
                    vals_c[rng.integers(0, n, n)].mean()
                    - vals_d[rng.integers(0, n, n)].mean()
                    for _ in range(1000)
                ]

                ci_rows.append({
                    "benchmark": key[0],
                    "n_samples": key[1],
                    "gap_ci_low": float(np.percentile(boots, 2.5)),
                    "gap_ci_high": float(np.percentile(boots, 97.5)),
                })

            gap = gap.merge(
                pd.DataFrame(ci_rows),
                on=group,
                how="left"
            )

        num_cols = gap.select_dtypes(include="number").columns
        gap[num_cols] = gap[num_cols].round(4)

        all_rows.append(gap)

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


def compute_pairwise_analysis(df: pd.DataFrame) -> pd.DataFrame:

    results = []

    group_cols = ["benchmark", "n_samples"]

    configs = set(df["config"].dropna().unique())

    families = sorted(
        set(cfg.rsplit("-", 1)[0] for cfg in configs if "-" in cfg)
    )

    for family in families:

        cfg_c = f"{family}-C"
        cfg_d = f"{family}-D"

        if not {cfg_c, cfg_d}.issubset(configs):
            continue

        for key, grp_c in df[df["config"] == cfg_c].groupby(group_cols):

            grp_d = df[
                (df["config"] == cfg_d)
                & (df["benchmark"] == key[0])
                & (df["n_samples"] == key[1])
                ]

            grp_c = grp_c.sort_values("rep")
            grp_d = grp_d.sort_values("rep")

            vals_c = grp_c["shd"].values
            vals_d = grp_d["shd"].values

            n = min(len(vals_c), len(vals_d))

            if n < 10:
                continue

            vals_c = vals_c[:n]
            vals_d = vals_d[:n]

            delta = vals_c - vals_d

            try:
                stat, p_value = wilcoxon(delta)
            except Exception:
                stat, p_value = np.nan, np.nan

            p_d_wins = np.mean(vals_d < vals_c)
            p_c_wins = np.mean(vals_c < vals_d)

            cliffs_delta = p_d_wins - p_c_wins

            n_pairs = n * (n + 1) / 2
            if not np.isnan(stat):
                r_rb = 1 - (2 * stat) / n_pairs
            else:
                r_rb = np.nan

            mean_diff = float(np.mean(delta))
            median_diff = float(np.median(delta))
            std_diff = float(np.std(delta))

            skew_proxy = float(
                (mean_diff - median_diff) / (std_diff + 1e-8)
            )

            results.append({
                "benchmark": key[0],
                "n_samples": key[1],
                "family": family,
                "config_c": cfg_c,
                "config_d": cfg_d,
                "n": n,

                "mean_diff": mean_diff,
                "median_diff": median_diff,
                "std_diff": std_diff,
                "skew_proxy": skew_proxy,

                "p_value": p_value,
                "wilcoxon_stat": stat,

                "p_d_wins": p_d_wins,
                "p_c_wins": p_c_wins,
                "cliffs_delta": cliffs_delta,
                "rank_biserial": r_rb,
            })

    df_out = pd.DataFrame(results)

    num_cols = df_out.select_dtypes(include="number").columns
    df_out[num_cols] = df_out[num_cols].round(4)

    return df_out


def compute_edge_effects(df):
    group = ["benchmark", "n_samples", "config"]

    agg = (
        df.groupby(group)[["tp", "fp", "fn", "n_nodes"]]
        .mean()
        .reset_index()
    )

    agg["total_possible"] = agg["n_nodes"] * (agg["n_nodes"] - 1)
    agg["tn"] = agg["total_possible"] - (agg["tp"] + agg["fp"] + agg["fn"])
    agg["tpr"] = agg["tp"] / (agg["tp"] + agg["fn"])
    agg["fpr"] = agg["fp"] / (agg["fp"] + agg["tn"])
    agg["fdr"] = agg["fp"] / (agg["tp"] + agg["fp"])
    agg["precision"] = agg["tp"] / (agg["tp"] + agg["fp"])

    agg.replace([np.inf, -np.inf], np.nan, inplace=True)

    return agg


def compute_edge_gaps(df):
    group = ["benchmark", "n_samples"]

    c = df[df["config"] == "HC-C"].groupby(group)[["tp", "fp", "fn"]].mean()
    d = df[df["config"] == "HC-D"].groupby(group)[["tp", "fp", "fn"]].mean()

    gap = c - d
    gap.columns = ["tp_gap", "fp_gap", "fn_gap"]

    return gap.reset_index()