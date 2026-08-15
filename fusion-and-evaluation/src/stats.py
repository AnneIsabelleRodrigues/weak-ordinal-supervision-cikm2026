from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import wilcoxon, f_oneway, kruskal

def confidence_interval(
        values: np.ndarray,
        level: float = 0.95,
        method: str = "t",
        n_boot: int = 1000,
        rng_seed: int = 0,
) -> Tuple[float, float]:

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)

    if n < 2:
        m = float(np.mean(values)) if n == 1 else np.nan
        return m, m

    if method == "t":
        se     = st.sem(values)
        t_crit = st.t.ppf((1 + level) / 2, df=n - 1)
        mean   = np.mean(values)
        return float(mean - t_crit * se), float(mean + t_crit * se)

    elif method == "bootstrap":
        rng   = np.random.default_rng(rng_seed)
        boots = [rng.choice(values, size=n, replace=True).mean()
                 for _ in range(n_boot)]
        alpha = (1 - level) / 2
        return (float(np.percentile(boots, alpha * 100)),
                float(np.percentile(boots, (1 - alpha) * 100)))

    else:
        raise ValueError(f"method deve ser 't' ou 'bootstrap', não '{method}'")

def aggregate(
        df: pd.DataFrame,
        group_cols: List[str],
        metric_cols: List[str] = ["shd", "f1", "precision", "recall", "sid", "bic"],
        ci_level: float = 0.95,
        ci_method: str = "t",
) -> pd.DataFrame:

    metric_cols = [c for c in metric_cols if c in df.columns]

    records = []
    for key, grp in df.groupby(group_cols):
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row["n_reps"] = len(grp)

        for m in metric_cols:
            vals = grp[m].dropna().values

            if len(vals) == 0:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_std"] = np.nan
                row[f"{m}_se"] = np.nan
                row[f"{m}_ci_low"] = np.nan
                row[f"{m}_ci_high"] = np.nan
                continue

            mean  = float(np.mean(vals))
            std   = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            se    = float(std / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            ci_lo, ci_hi = confidence_interval(vals, level=ci_level, method=ci_method)

            row[f"{m}_mean"] = round(mean,  4)
            row[f"{m}_std"] = round(std,   4)
            row[f"{m}_se"] = round(se,    4)
            row[f"{m}_ci_low"] = round(ci_lo, 4)
            row[f"{m}_ci_high"] = round(ci_hi, 4)

        records.append(row)

    return pd.DataFrame(records).sort_values(group_cols).reset_index(drop=True)


def bootstrap_ci(
        df: pd.DataFrame,
        group_cols: List[str],
        metric_cols: List[str],
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        seed: int = 42,
) -> pd.DataFrame:

    alpha = 1.0 - ci
    lo_pct = 100 * (alpha / 2)
    hi_pct = 100 * (1 - alpha / 2)
    rng = np.random.default_rng(seed)

    records = []
    for keys, group in df.groupby(group_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row  = dict(zip(group_cols, keys))

        for metric in metric_cols:
            values = group[metric].dropna().values

            if len(values) == 0:
                row[f"{metric}_mean"]    = np.nan
                row[f"{metric}_ci95_lo"] = np.nan
                row[f"{metric}_ci95_hi"] = np.nan
                continue

            boot_means = np.array([
                rng.choice(values, size=len(values), replace=True).mean()
                for _ in range(n_bootstrap)
            ])

            row[f"{metric}_mean"] = round(float(values.mean()), 4)
            row[f"{metric}_ci95_lo"] = round(float(np.percentile(boot_means, lo_pct)), 4)
            row[f"{metric}_ci95_hi"] = round(float(np.percentile(boot_means, hi_pct)), 4)

        records.append(row)

    return pd.DataFrame(records)


def anova_grid(
        df: pd.DataFrame,
        metric: str = "shd",
        factors: List[str] = ["benchmark", "n_samples", "temperature", "lambda_"],
        config_filter: Optional[str] = "MCMC-D",
        config_col: str = "config",
        verbose: bool = True,
) -> pd.DataFrame:

    subset = df.copy()
    if config_filter and config_col in df.columns:
        subset = subset[subset[config_col] == config_filter]

    if metric not in subset.columns:
        raise ValueError(f"Metric '{metric}' not found in DataFrame.")

    rows       = []
    grand_mean = subset[metric].mean()
    ss_total   = ((subset[metric] - grand_mean) ** 2).sum()

    for factor in factors:
        if factor not in subset.columns:
            continue

        groups = [grp[metric].dropna().values
                  for _, grp in subset.groupby(factor)]
        groups = [g for g in groups if len(g) >= 2]

        if len(groups) < 2:
            continue

        try:
            F, p = f_oneway(*groups)
        except Exception:
            F, p = np.nan, np.nan

        group_means = np.array([g.mean() for g in groups])
        group_ns = np.array([len(g) for g in groups])
        ss_between = float(np.sum(group_ns * (group_means - grand_mean) ** 2))
        eta_sq = round(ss_between / ss_total, 4) if ss_total > 0 else np.nan

        rows.append({
            "factor": factor,
            "n_levels": len(groups),
            "F_stat": round(float(F), 4) if not np.isnan(F) else np.nan,
            "p_value": round(float(p), 4) if not np.isnan(p) else np.nan,
            "significant": bool(p < 0.05) if not np.isnan(p) else False,
            "eta_squared": eta_sq,
        })

    df_anova = (pd.DataFrame(rows)
                .sort_values("F_stat", ascending=False)
                .reset_index(drop=True))

    if verbose:
        print(f"\n[ANOVA] config={config_filter}  metric={metric}")
        print(df_anova.to_string(index=False))

    return df_anova


def wilcoxon_table(
        df: pd.DataFrame,
        group_cols: List[str],
        config_col: str = "config",
        config_a: str = "MCMC-C",
        config_b: str = "MCMC-D",
        metric: str = "shd",
        rep_col: str = "rep",
) -> pd.DataFrame:

    rows = []
    mask = df[config_col].isin([config_a, config_b])

    for key, grp in df[mask].groupby(group_cols):
        vals_a = (grp[grp[config_col] == config_a]
                  .sort_values(rep_col)[metric].values)
        vals_b = (grp[grp[config_col] == config_b]
                  .sort_values(rep_col)[metric].values)

        n = min(len(vals_a), len(vals_b))
        vals_a, vals_b = vals_a[:n], vals_b[:n]

        if n < 3:
            stat, p = np.nan, np.nan
        else:
            diff = vals_a - vals_b
            if np.all(diff == 0):
                stat, p = np.nan, 1.0
            else:
                try:
                    stat, p = wilcoxon(vals_a, vals_b, alternative="greater")
                except Exception:
                    stat, p = np.nan, np.nan

        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row.update({
            f"mean_{config_a}": round(float(np.mean(vals_a)), 4) if n > 0 else np.nan,
            f"mean_{config_b}": round(float(np.mean(vals_b)), 4) if n > 0 else np.nan,
            "mean_diff": round(float(np.mean(vals_a - vals_b)), 4) if n > 0 else np.nan,
            "n_reps": n,
            "statistic": round(float(stat), 4) if not np.isnan(stat) else np.nan,
            "p_value": round(float(p), 4) if not np.isnan(p) else np.nan,
            "significant": bool(p < 0.05) if not np.isnan(p) else False,
        })
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def summary_table(
        df_agg: pd.DataFrame,
        df_wilcoxon: Optional[pd.DataFrame],
        group_cols: List[str],
        metric: str = "shd",
        config_col: str = "config",
        config_b: str = "HC-D",
        ci: bool = True,
) -> pd.DataFrame:

    df = df_agg.copy()

    def _cell(row):
        mean = row.get(f"{metric}_mean", np.nan)
        std = row.get(f"{metric}_std",  np.nan)
        lo = row.get(f"{metric}_ci_low",  np.nan)
        hi = row.get(f"{metric}_ci_high", np.nan)
        if np.isnan(mean):
            return "—"
        s = f"{mean:.3f} ± {std:.3f}"
        if ci and not np.isnan(lo):
            s += f" [{lo:.3f}, {hi:.3f}]"
        return s

    df["display"] = df.apply(_cell, axis=1)

    if df_wilcoxon is not None and "significant" in df_wilcoxon.columns:
        sig_keys = set(
            tuple(row[c] for c in group_cols)
            for _, row in df_wilcoxon[df_wilcoxon["significant"]].iterrows()
        )

        def _add_star(row):
            key = tuple(row[c] for c in group_cols)
            if row.get(config_col) == config_b and key in sig_keys:
                return row["display"] + " *"
            return row["display"]

        df["display"] = df.apply(_add_star, axis=1)

    pivot = df.pivot_table(
        index   = group_cols,
        columns = config_col,
        values  = "display",
        aggfunc = "first",
    )
    pivot.columns.name = None
    return pivot.reset_index()


def test_gap_sign(
        df_raw: pd.DataFrame,
        group_cols: List[str],
        config_c: str = "HC-C",
        config_d: str = "HC-D",
        metric: str = "shd",
) -> pd.DataFrame:

    records = []

    for keys, group in df_raw.groupby(group_cols, sort=False):
        keys   = keys if isinstance(keys, tuple) else (keys,)
        row    = dict(zip(group_cols, keys))

        c_vals = group[group["config"] == config_c].set_index("rep")[metric]
        d_vals = group[group["config"] == config_d].set_index("rep")[metric]
        common = c_vals.index.intersection(d_vals.index)

        if len(common) < 5:
            row.update({
                "gap_median": np.nan, "gap_mean":    np.nan,
                "gap_ci95_lo": np.nan, "gap_ci95_hi": np.nan,
                "n_pairs": len(common), "w_stat": np.nan,
                "p_value": np.nan, "significant": False,
            })
            records.append(row)
            continue

        gap = (c_vals[common] - d_vals[common]).values

        rng  = np.random.default_rng(42)
        boot = np.array([
            rng.choice(gap, size=len(gap), replace=True).mean()
            for _ in range(1000)
        ])

        try:
            w_stat, p_value = wilcoxon(gap, alternative="greater")
        except Exception:
            w_stat, p_value = np.nan, np.nan

        row.update({
            "gap_median": round(float(np.median(gap)), 4),
            "gap_mean": round(float(gap.mean()), 4),
            "gap_ci95_lo": round(float(np.percentile(boot, 2.5)), 4),
            "gap_ci95_hi": round(float(np.percentile(boot, 97.5)), 4),
            "n_pairs": len(common),
            "w_stat": round(float(w_stat),  4) if not np.isnan(w_stat)  else np.nan,
            "p_value": round(float(p_value), 6) if not np.isnan(p_value) else np.nan,
            "significant": bool(p_value < 0.05) if not np.isnan(p_value) else False,
        })
        records.append(row)

    return pd.DataFrame(records)


def kruskal_posthoc(
        df_raw: pd.DataFrame,
        group_cols: List[str],
        condition_col: str,
        metric: str,
        reference: Optional[str] = None,
) -> pd.DataFrame:

    records = []

    for keys, group in df_raw.groupby(group_cols, sort=False):
        keys     = keys if isinstance(keys, tuple) else (keys,)
        base_row = dict(zip(group_cols, keys))

        groups = {
            c: group[group[condition_col] == c][metric].dropna().values
            for c in group[condition_col].dropna().unique()
        }
        groups = {c: v for c, v in groups.items() if len(v) >= 3}

        if len(groups) < 2:
            continue

        try:
            _, kw_p = kruskal(*groups.values())
        except Exception:
            kw_p = np.nan

        pairs = list(combinations(groups.keys(), 2))
        if reference:
            pairs = [(a, b) for a, b in pairs if a == reference or b == reference]
        n_pairs = len(pairs)

        for cond_a, cond_b in pairs:
            row    = base_row.copy()
            va, vb = groups[cond_a], groups[cond_b]

            try:
                u_stat, p_raw = st.mannwhitneyu(va, vb, alternative="two-sided")
                p_bonf        = min(p_raw * n_pairs, 1.0)
            except Exception:
                u_stat, p_raw, p_bonf = np.nan, np.nan, np.nan

            row.update({
                "condition_a": cond_a,
                "condition_b": cond_b,
                "median_a": round(float(np.median(va)), 4),
                "median_b": round(float(np.median(vb)), 4),
                "delta_median": round(float(np.median(va) - np.median(vb)), 4),
                "n_a": len(va),
                "n_b": len(vb),
                "u_stat": round(float(u_stat), 2) if not np.isnan(u_stat) else np.nan,
                "p_raw": round(float(p_raw),  6) if not np.isnan(p_raw) else np.nan,
                "p_bonferroni": round(float(p_bonf), 6) if not np.isnan(p_bonf) else np.nan,
                "significant": bool(p_bonf < 0.05) if not np.isnan(p_bonf) else False,
                "kruskal_p": round(float(kw_p), 6) if not np.isnan(kw_p) else np.nan,
            })
            records.append(row)

    return pd.DataFrame(records)


def wilcoxon_pairs(
        df_raw: pd.DataFrame,
        group_cols: List[str],
        condition_col: str,
        metric: str,
        pairs: List[Tuple[str, str]],
) -> pd.DataFrame:

    records = []

    for keys, group in df_raw.groupby(group_cols, sort=False):
        keys     = keys if isinstance(keys, tuple) else (keys,)
        base_row = dict(zip(group_cols, keys))

        for cond_a, cond_b in pairs:
            row = base_row.copy()

            va = group[group[condition_col] == cond_a].set_index("rep")[metric]
            vb = group[group[condition_col] == cond_b].set_index("rep")[metric]

            common = va.index.intersection(vb.index)

            if len(common) < 5:
                row.update({
                    "condition_a": cond_a, "condition_b": cond_b,
                    "median_a": np.nan, "median_b": np.nan,
                    "delta_median": np.nan, "n_pairs": len(common),
                    "w_stat": np.nan, "p_value": np.nan, "significant": False,
                })
                records.append(row)
                continue

            diff = (va[common] - vb[common]).values

            try:
                w_stat, p_value = wilcoxon(diff, alternative="two-sided")
            except Exception:
                w_stat, p_value = np.nan, np.nan

            row.update({
                "condition_a": cond_a,
                "condition_b": cond_b,
                "median_a": round(float(np.median(va[common].values)), 4),
                "median_b": round(float(np.median(vb[common].values)), 4),
                "delta_median": round(float(np.median(diff)), 4),
                "n_pairs": len(common),
                "w_stat": round(float(w_stat), 4) if not np.isnan(w_stat)  else np.nan,
                "p_value": round(float(p_value), 6) if not np.isnan(p_value) else np.nan,
                "significant": bool(p_value < 0.05) if not np.isnan(p_value) else False,
            })
            records.append(row)

    return pd.DataFrame(records)


def jonckheere_trend(
        df_raw: pd.DataFrame,
        group_cols: List[str],
        order_col: str,
        metric: str,
        ordered_values: Optional[List] = None,
        alternative: str = "decreasing",
) -> pd.DataFrame:

    records = []

    for keys, group in df_raw.groupby(group_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base_row = dict(zip(group_cols, keys))

        vals = ordered_values or sorted(group[order_col].dropna().unique())
        groups_ordered = [
            (v, group[group[order_col] == v][metric].dropna().values)
            for v in vals
        ]
        groups_ordered = [(v, g) for v, g in groups_ordered if len(g) >= 2]

        if len(groups_ordered) < 3:
            continue

        jt_stat = 0.0
        for i, (_, gi) in enumerate(groups_ordered):
            for j, (_, gj) in enumerate(groups_ordered):
                if i >= j:
                    continue
                u, _ = st.mannwhitneyu(gi, gj, alternative="less")
                jt_stat += u

        ns     = np.array([len(g) for _, g in groups_ordered])
        n      = ns.sum()
        e_jt   = (n**2 - (ns**2).sum()) / 4
        var_jt = (
            n**2 * (2*n + 3) - sum(ni**2 * (2*ni + 3) for ni in ns)
        ) / 72

        if var_jt <= 0:
            z_stat, p_value = np.nan, np.nan
        else:
            z_stat  = (jt_stat - e_jt) / np.sqrt(var_jt)
            p_value = (float(st.norm.cdf(z_stat)) if alternative == "decreasing"
                       else float(st.norm.sf(z_stat)))

        ordered_medians = {
            str(v): round(float(np.median(g)), 4)
            for v, g in groups_ordered
        }

        base_row.update({
            "jt_stat": round(float(jt_stat), 2),
            "z_stat": round(float(z_stat),  4) if not np.isnan(z_stat)  else np.nan,
            "p_value": round(float(p_value), 6) if not np.isnan(p_value) else np.nan,
            "significant": bool(p_value < 0.05) if not np.isnan(p_value) else False,
            "trend_direction": alternative,
            "ordered_medians": str(ordered_medians),
        })
        records.append(base_row)

    return pd.DataFrame(records)


def print_ci_summary(
        df_ci: pd.DataFrame,
        experiment: str,
        metric: str,
        group_cols: List[str],
        condition_col: Optional[str] = None,
        max_rows: int = 12,
) -> None:

    mean_col = f"{metric}_mean"

    if f"{metric}_ci95_lo" in df_ci.columns:
        lo_col, hi_col = f"{metric}_ci95_lo", f"{metric}_ci95_hi"
    elif f"{metric}_ci_low" in df_ci.columns:
        lo_col, hi_col = f"{metric}_ci_low",  f"{metric}_ci_high"
    else:
        print(f"[{experiment}] IC summary: columns not found for '{metric}'")
        return

    print(f"\n[{experiment}] {metric.upper()}  —  média [IC 95%]")

    display = df_ci.copy()
    display["ci_str"] = (
        display[mean_col].map("{:.3f}".format)
        + " ["
        + display[lo_col].map("{:.3f}".format)
        + ", "
        + display[hi_col].map("{:.3f}".format)
        + "]"
    )

    if condition_col and condition_col in display.columns:
        try:
            pivot_cols = [c for c in group_cols if c != condition_col]
            pivot = display.pivot_table(
                index = pivot_cols if pivot_cols else condition_col,
                columns = condition_col,
                values = "ci_str",
                aggfunc = "first",
            )
            print(pivot.to_string())
            return
        except Exception:
            pass

    print(display[group_cols + ["ci_str"]].head(max_rows).to_string(index=False))