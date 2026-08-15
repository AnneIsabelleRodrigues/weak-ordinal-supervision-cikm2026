import argparse
import os

import pandas as pd

from config.env import DIR_RESULTS
from src.constants import GRAPH_DEF
from src.evaluation import (
    compute_edge_effects,
    compute_edge_gaps,
    compute_pairwise_analysis,
    compute_shd_gap,
    compute_sid_gap
)
from src.experiments.config_grid   import run_e1_e4
from src.experiments.ordinal_validity import run_e2
from src.experiments.mapping_sensitivity import run_e5
from src.experiments.orientation_gain import run_e7
from src.experiments.prior_conflict   import run_e8
from src.experiments.fusion_ablation import run_e9a, run_e9b
from src.experiments.default_weight_sensitivity import run_e10
from src.stats import (
    aggregate,
    anova_grid,
    bootstrap_ci,
    jonckheere_trend,
    kruskal_posthoc,
    print_ci_summary,
    summary_table,
    test_gap_sign,
    wilcoxon_pairs,
    wilcoxon_table,
)
from src.utils import get_regime_samples, get_sample_sizes

def main(benchmark_name: str = "CHILD"):

    RUN_E1_E4 = True
    RUN_E2    = True
    RUN_E4    = True
    RUN_E5    = True
    RUN_E7    = True
    RUN_E8    = True
    RUN_E9A   = True
    RUN_E9B   = True
    RUN_E10   = True

    LAMBDA         = 0.9
    THETA          = 0.0
    THETAS_FAT     = [0.0, 0.3]
    BASE_DATA_SEED = 0

    N_REPS_E1 = 100
    N_REPS_E5 = 20
    N_REPS_E7 = 20
    N_REPS_E8 = 20
    N_REPS_E9 = 20

    n_nodes      = GRAPH_DEF[benchmark_name]['n_nodes']
    n_params     = GRAPH_DEF[benchmark_name]['n_params']

    # ==========================================================================
    #  Grid - Data regime analysis
    # ==========================================================================

    if RUN_E1_E4:
        print("\n" + "=" * 70)
        print(f"E1 / E4  —  {N_REPS_E1} rep  |  Configs: PC, GES, HC-C, HC-D, HC-E")

        sample_sizes = get_sample_sizes(n_nodes)

        df_e1_raw = run_e1_e4(
            benchmark      = benchmark_name,
            sample_sizes   = sample_sizes,
            lambdas        = [0.90],
            thetas         = [THETA],
            run_pc         = True,
            run_ges        = True,
            run_hc_c       = True,
            run_hc_d       = True,
            run_hc_e       = True,
            compute_sid    = True,
            compute_bic    = True,
            n_reps         = N_REPS_E1,
            base_data_seed = BASE_DATA_SEED,
            max_iter       = int(10 * n_nodes^2),
            output_csv     = f"{DIR_RESULTS}/e1_e4_raw.csv",
            verbose        = False,
        )

        df_e1_fat = run_e1_e4(
            benchmark      = benchmark_name,
            sample_sizes   = [int(10*n_nodes), int(100*n_nodes)],
            lambdas        = [0.5, 1.0],
            thetas         = THETAS_FAT,
            run_pc         = False,
            run_ges        = False,
            run_hc_c       = True,
            run_hc_d       = True,
            run_hc_e       = False,
            compute_sid    = True,
            compute_bic    = True,
            n_reps         = N_REPS_E1,
            base_data_seed = BASE_DATA_SEED,
            max_iter       = int(10 * n_nodes^2),
            output_csv     = f"{DIR_RESULTS}/e1_e4_raw_fat.csv",
            verbose        = False,
        )

        group_e1 = ["benchmark", "config", "n_samples", "lambda_"]

        df_e1_agg = aggregate(
            df_e1_raw,
            group_cols  = group_e1,
            metric_cols = ["shd", "f1", "precision", "recall", "sid", "bic"],
        )

        df_e1_agg.to_csv(f"{DIR_RESULTS}/e1_e4_aggregated.csv", index=False)
        print(f"\n[E1] Aggregated → e1_e4_aggregated.csv ({len(df_e1_agg)} rows)")

        configs_raw = set(df_e1_raw["config"].dropna().unique())
        configs_agg = set(df_e1_agg["config"].dropna().unique())

        cfg_c = "HC-C"
        cfg_d = "HC-D"

        if {cfg_c, cfg_d}.issubset(configs_raw):

            df_wx = wilcoxon_table(
                df_e1_raw,
                group_cols = ["benchmark", "n_samples"],
                config_a   = cfg_c,
                config_b   = cfg_d,
                metric     = "shd",
            )
            df_wx.to_csv(f"{DIR_RESULTS}/e1_wilcoxon_C_vs_D.csv", index=False)
            print(f"[E1] Wilcoxon {cfg_c} ↔ {cfg_d}")

            df_wx_sid = wilcoxon_table(
                df_e1_raw,
                group_cols = ["benchmark", "n_samples"],
                config_a   = cfg_c,
                config_b   = cfg_d,
                metric     = "sid",
            )
            df_wx_sid.to_csv(f"{DIR_RESULTS}/e1_wilcoxon_C_vs_D_SID.csv", index=False)
            print(f"[E1] Wilcoxon SID {cfg_c} ↔ {cfg_d}")

        else:
            df_wx     = None

        if cfg_d in configs_raw:

            df_anova = anova_grid(
                df_e1_fat,
                metric        = "shd",
                factors       = ["benchmark", "config", "n_samples", "lambda_", "theta"],
                config_filter = cfg_d,
                verbose       = True,
            )
            df_anova.to_csv(f"{DIR_RESULTS}/e1_anova.csv", index=False)
            print(f"[E1] {cfg_d} ANOVA → e1_anova.csv")

        if {cfg_c, cfg_d}.issubset(configs_agg):

            df_paper = summary_table(
                df_e1_agg,
                df_wilcoxon = df_wx,
                group_cols  = ["benchmark", "n_samples"],
                metric      = "shd",
                config_col  = "config",
                config_b    = cfg_d,
            )
            df_paper.to_csv(f"{DIR_RESULTS}/e1_table_paper.csv", index=False)
            print(f"[E1] Paper table → e1_table_paper.csv")

            df_paper_sid = summary_table(
                df_e1_agg,
                df_wilcoxon = df_wx,
                group_cols  = ["benchmark", "n_samples"],
                metric      = "sid",
                config_col  = "config",
                config_b    = cfg_d,
            )
            df_paper_sid.to_csv(f"{DIR_RESULTS}/e1_table_paper_sid.csv", index=False)
            print(f"[E1] Paper table SID → e1_table_paper_sid.csv")

        if RUN_E4:
            print("\n[E4] SHD gap com IC 95% bootstrapped...")
            df_gap = compute_shd_gap(df_e1_raw)
            df_gap.to_csv(f"{DIR_RESULTS}/e4_shd_gap.csv", index=False)

            print("\n[E4] SID gap com IC 95% bootstrapped...")
            df_sid_gap = compute_sid_gap(df_e1_raw)
            df_sid_gap.to_csv(f"{DIR_RESULTS}/e4_sid_gap.csv", index=False)
            print("[E4] → e4_sid_gap.csv")

            for fam in sorted(df_gap["config_family"].unique()):
                sub = df_gap[df_gap["config_family"] == fam]
                print(f"\n--- GAP {fam} ---")
                print(
                    sub.pivot(
                        index   = "benchmark",
                        columns = "n_samples",
                        values  = "shd_gap",
                    ).round(2).to_string()
                )

            analysis_df = compute_pairwise_analysis(df_e1_raw)
            analysis_df.to_csv(f"{DIR_RESULTS}/pairwise_analysis.csv", index=False)
            print("[E4] → pairwise_analysis.csv")

            df_effect = compute_edge_effects(df_e1_raw)
            df_effect.to_csv(f"{DIR_RESULTS}/edge_effect.csv", index=False)
            print("[E4] → edge_effect.csv")

            df_gaps = compute_edge_gaps(df_e1_raw)
            df_gaps.to_csv(f"{DIR_RESULTS}/edge_gaps.csv", index=False)
            print("[E4] → edge_gaps.csv")

            df_gap_sign = test_gap_sign(
                df_e1_raw,
                group_cols = ["benchmark", "n_samples"],
                config_c   = "HC-C",
                config_d   = "HC-D",
                metric     = "shd",
            )
            df_gap_sign.to_csv(f"{DIR_RESULTS}/e4_gap_sign_test.csv", index=False)
            print("[E4] Gap sign test → e4_gap_sign_test.csv")

            df_e4_ci = bootstrap_ci(
                df_e1_raw,
                group_cols  = ["benchmark", "config", "n_samples"],
                metric_cols = ["shd", "sid"],
            )
            df_e4_ci.to_csv(f"{DIR_RESULTS}/e4_ci95.csv", index=False)
            print_ci_summary(df_e4_ci, "E4", "shd",
                             group_cols=["benchmark", "n_samples"],
                             condition_col="config")

    # ==========================================================================
    #  Ordinal ranking + LLM Calibration + Ancestral Precision
    # ==========================================================================

    if RUN_E2:
        print("\n" + "=" * 70)
        print("E2 — Ordinal validity, LLM calibration, ancestral precision")

        df_e2_ord, df_e2_cal, df_e2_ece = run_e2(
            bench                   = benchmark_name,
            theta                   = 0.0,
            output_csv_ordering     = f"{DIR_RESULTS}/e2_ordinal_ordering.csv",
            output_csv_calibration  = f"{DIR_RESULTS}/e2_calibration.csv",
            output_csv_ece          = f"{DIR_RESULTS}/e2_ece.csv",
            verbose                 = False,
        )

    # ==========================================================================
    #  Sensitivity to ordinal mapping
    # ==========================================================================

    if RUN_E5:
        print("\n" + "=" * 70)
        print(f"E5 — Mapping Sensitivity  —  {N_REPS_E5} rep")

        df_e5_raw = run_e5(
            bench          = benchmark_name,
            n_samples      = 100*n_nodes,
            lambda_        = LAMBDA,
            theta          = THETA,
            n_reps         = N_REPS_E5,
            max_indegree   = 5,
            max_iter       = int(10 * n_nodes^2),
            base_data_seed = BASE_DATA_SEED,
            output_csv     = f"{DIR_RESULTS}/e5_raw.csv",
            verbose        = False,
        )

        df_e5_agg = aggregate(
            df_e5_raw,
            group_cols  = ["benchmark", "mapping"],
            metric_cols = ["shd", "f1", "sid", "bic"],
        )
        df_e5_agg.to_csv(f"{DIR_RESULTS}/e5_aggregated.csv", index=False)
        print(f"[E5] Aggregated → e5_aggregated.csv")

        df_e5_ci = bootstrap_ci(
            df_e5_raw,
            group_cols  = ["benchmark", "mapping"],
            metric_cols = ["shd", "f1"],
        )
        df_e5_ci.to_csv(f"{DIR_RESULTS}/e5_ci95.csv", index=False)
        print_ci_summary(df_e5_ci, "E5", "shd",
                         group_cols=["benchmark"],
                         condition_col="mapping")

        df_e5_kw = kruskal_posthoc(
            df_e5_raw,
            group_cols    = ["benchmark"],
            condition_col = "mapping",
            metric        = "shd",
        )
        df_e5_kw.to_csv(f"{DIR_RESULTS}/e5_kruskal_posthoc.csv", index=False)

        n_sig = df_e5_kw["significant"].sum() if not df_e5_kw.empty else 0
        print(f"[E5] Kruskal post-hoc → e5_kruskal_posthoc.csv  "
              f"({n_sig}/{len(df_e5_kw)} pares significativos)")

    # ==========================================================================
    #  CPDAG Orientation Gain
    # ==========================================================================

    if RUN_E7:
        print("\n" + "=" * 70)
        print(f"E7 — CPDAG Orientation Gain  —  {N_REPS_E7} rep")

        sample_sizes = get_regime_samples(n_params)

        df_e7_raw = run_e7(
            bench          = benchmark_name,
            sample_sizes   = sample_sizes,
            lambda_        = LAMBDA,
            theta          = THETA,
            n_reps         = N_REPS_E7,
            max_indegree   = 5,
            max_iter       = int(10 * n_nodes^2),
            output_csv     = f"{DIR_RESULTS}/e7_orientation_gain.csv",
            verbose        = False,
        )

        df_e7_agg = aggregate(
            df_e7_raw,
            group_cols  = ["benchmark", "config", "n_samples"],
            metric_cols = [
                "shd", "f1", "sid", "bic",
                "n_ambiguous", "n_correctly_oriented",
                "orientation_recall", "orientation_precision",
            ],
        )
        df_e7_agg.to_csv(f"{DIR_RESULTS}/e7_aggregated.csv", index=False)

        df_e7_filtered = df_e7_raw[df_e7_raw["config"].isin(["HC-C", "HC-D"])]

        df_e7_wx = wilcoxon_table(
            df_e7_filtered,
            group_cols = ["benchmark", "n_samples"],
            config_a   = "HC-C",
            config_b   = "HC-D",
            metric     = "orientation_recall",
        )
        df_e7_wx.to_csv(f"{DIR_RESULTS}/e7_wilcoxon.csv", index=False)

        print("[E7] Aggregated → e7_aggregated.csv  |  Wilcoxon → e7_wilcoxon.csv")

    # ==========================================================================
    #  Prior–Data Conflict
    # ==========================================================================

    if RUN_E8:
        print("\n" + "=" * 70)
        print(f"E8 — Prior–Data Conflict  —  {N_REPS_E8} rep")

        sample_sizes = get_regime_samples(n_params)

        df_e8_raw = run_e8(
            bench          = benchmark_name,
            sample_sizes   = sample_sizes,
            lambda_        = LAMBDA,
            theta          = THETA,
            n_reps         = N_REPS_E8,
            base_data_seed = BASE_DATA_SEED,
            output_csv     = f"{DIR_RESULTS}/e8_prior_conflict.csv",
            verbose        = False,
        )

        df_e8_agg = aggregate(
            df_e8_raw,
            group_cols  = ["benchmark", "condition", "n_samples"],
            metric_cols = ["shd", "f1", "sid", "bic"],
        )
        df_e8_agg.to_csv(f"{DIR_RESULTS}/e8_aggregated.csv", index=False)

        df_e8_filtered = df_e8_raw[
            df_e8_raw["condition"].isin(["HC-C", "HC-D-inverted"])
        ]

        df_e8_wx = wilcoxon_table(
            df_e8_filtered,
            group_cols = ["n_samples"],
            config_col = "condition",
            config_a   = "HC-D-inverted",
            config_b   = "HC-C",
            metric     = "shd",
        )
        df_e8_wx.to_csv(f"{DIR_RESULTS}/e8_wilcoxon_damage.csv", index=False)

        print("[E8] Aggregated → e8_aggregated.csv  |  Damage test → e8_wilcoxon_damage.csv")

        dam = df_e8_wx[df_e8_wx["significant"]]

        if len(dam):
            print(
                f"Cells where prior WRONG significantly worsens:"
                f"{len(dam)}/{len(df_e8_wx)}"
            )
        else:
            print("Resilient HC: wrong prior doesn’t get significantly worse")

    # ==========================================================================
    #  Fusion Ablation: Penalities
    # ==========================================================================

    if RUN_E9A:
        print("\n" + "=" * 70)
        print(f"E9a — Fusion Ablation (binária)  —  {N_REPS_E9} rep")

        sample_sizes = get_regime_samples(n_params)

        df_e9a_raw = run_e9a(
            bench          = benchmark_name,
            sample_sizes   = sample_sizes,
            theta          = 0.0,
            lambda_        = LAMBDA,
            n_reps         = N_REPS_E9,
            base_data_seed = BASE_DATA_SEED,
            max_iter       = int(10 * n_nodes^2),
            compute_sid    = True,
            compute_bic    = True,
            output_csv     = f"{DIR_RESULTS}/e9a_ablation_raw.csv",
            verbose        = False,
        )

        df_e9a_agg = aggregate(
            df_e9a_raw,
            group_cols  = ["benchmark", "fusion_config", "n_samples"],
            metric_cols = ["auroc", "ece", "shd", "f1", "sid", "bic"],
        )
        df_e9a_agg.to_csv(f"{DIR_RESULTS}/e9a_ablation_aggregated.csv", index=False)
        print("[E9a] Aggregated → e9a_ablation_aggregated.csv")

        df_e9a_ci = bootstrap_ci(
            df_e9a_raw,
            group_cols  = ["benchmark", "fusion_config", "n_samples"],
            metric_cols = ["auroc", "ece", "shd", "f1"],
        )
        df_e9a_ci.to_csv(f"{DIR_RESULTS}/e9a_ci95.csv", index=False)
        print_ci_summary(df_e9a_ci, "E9a", "shd",
                         group_cols=["benchmark", "n_samples"],
                         condition_col="fusion_config")

        df_e9a_wx = wilcoxon_pairs(
            df_e9a_raw,
            group_cols    = ["benchmark", "n_samples"],
            condition_col = "fusion_config",
            metric        = "shd",
            pairs         = [
                ("Full", "Mean-Simple"),
                ("Full", "Full-NoSmooth"),
            ],
        )
        df_e9a_wx.to_csv(f"{DIR_RESULTS}/e9a_wilcoxon_pairs.csv", index=False)

        n_sig = df_e9a_wx["significant"].sum() if not df_e9a_wx.empty else 0
        print(f"[E9a] Wilcoxon pairs → e9a_wilcoxon_pairs.csv  "
              f"({n_sig}/{len(df_e9a_wx)} células significativas)")

    # ==========================================================================
    #  Fusion Ablation: α sensitivity
    # ==========================================================================

    if RUN_E9B:
        print("\n" + "=" * 70)
        print(f"E9b — Fusion Ablation (α sensitivity)  —  {N_REPS_E9} rep")

        sample_sizes = get_regime_samples(n_params)

        df_e9b_raw = run_e9b(
            bench          = benchmark_name,
            sample_sizes   = sample_sizes,
            theta          = 0.0,
            lambda_        = LAMBDA,
            n_reps         = N_REPS_E9,
            base_data_seed = BASE_DATA_SEED,
            max_iter       = int(10 * n_nodes^2),
            compute_sid    = True,
            compute_bic    = True,
            output_csv     = f"{DIR_RESULTS}/e9b_alpha_raw.csv",
            verbose        = False,
        )

        df_e9b_agg = aggregate(
            df_e9b_raw,
            group_cols  = ["benchmark", "fusion_config", "alpha", "n_samples"],
            metric_cols = ["auroc", "ece", "shd", "f1", "sid", "bic"],
        )
        df_e9b_agg.to_csv(f"{DIR_RESULTS}/e9b_alpha_aggregated.csv", index=False)
        print("[E9b] Aggregated → e9b_alpha_aggregated.csv")

    # ==========================================================================
    #  Default weight sensitivity
    # ==========================================================================

    if RUN_E10:
        print("\n" + "=" * 70)
        print(f"E10 — Default Weight Sensitivity  —  {N_REPS_E9} rep")
        print(f"      Benchmarks: ALARM, CHILD  |  grid: {[0.1, 0.2, 0.3, 0.4, 0.5]}")

        df_e10_raw = run_e10(
            benchmarks          = ["ALARM", "CHILD"],
            sample_sizes        = get_regime_samples(n_params),
            default_weight_grid = [0.1, 0.2, 0.3, 0.4, 0.5],
            theta               = THETA,
            lambda_             = LAMBDA,
            n_reps              = N_REPS_E9,
            base_data_seed      = BASE_DATA_SEED,
            max_iter            = int(10 * n_nodes^2),
            compute_sid         = True,
            compute_bic         = True,
            output_csv          = f"{DIR_RESULTS}/e10_default_weight_raw.csv",
            verbose             = False,
        )

        df_e10_agg = aggregate(
            df_e10_raw,
            group_cols  = ["benchmark", "config", "default_weight", "n_samples"],
            metric_cols = ["shd", "f1", "sid", "bic"],
        )
        df_e10_agg.to_csv(f"{DIR_RESULTS}/e10_default_weight_aggregated.csv", index=False)
        print("[E10] Aggregated → e10_default_weight_aggregated.csv")

        df_e10_ci = bootstrap_ci(
            df_e10_raw[df_e10_raw["config"] == "HC-D"],
            group_cols  = ["benchmark", "config", "default_weight"],
            metric_cols = ["shd", "f1"],
        )
        df_e10_ci.to_csv(f"{DIR_RESULTS}/e10_ci95.csv", index=False)
        print_ci_summary(df_e10_ci, "E10", "shd",
                         group_cols      = ["benchmark"],
                         condition_col   = "default_weight")

        df_e10_jt = jonckheere_trend(
            df_e10_raw[df_e10_raw["config"] == "HC-D"],
            group_cols     = ["benchmark"],
            order_col      = "default_weight",
            metric         = "shd",
            ordered_values = sorted(df_e10_raw["default_weight"].unique()),
            alternative    = "decreasing",
        )
        df_e10_jt.to_csv(f"{DIR_RESULTS}/e10_jonckheere.csv", index=False)

        if not df_e10_jt.empty:
            for _, r in df_e10_jt.iterrows():
                sig = "✓ significativo" if r["significant"] else "✗ não significativo"
                print(f"[E10] JT {r['benchmark']}: z={r['z_stat']:.3f}  "
                      f"p={r['p_value']:.4f}  {sig}")
                print(f"      medians: {r['ordered_medians']}")

        hc_d = df_e10_agg[df_e10_agg["config"] == "HC-D"].copy()
        if not hc_d.empty:
            pivot = (
                hc_d.groupby(["benchmark", "default_weight"])["shd_mean"]
                .mean()
                .unstack("default_weight")
                .round(3)
            )
            print("\n[E10] SHD médio por default_weight (HC-D):")
            print(pivot.to_string())

    # ==========================================================================
    #  SUMÁRIO FINAL
    # ==========================================================================

    print("\n" + "=" * 70)
    print("SUMÁRIO — results/experiments/")

    files = sorted(os.listdir(f"{DIR_RESULTS}"))
    for fname in files:
        fpath = os.path.join(f"{DIR_RESULTS}", fname)
        try:
            df          = pd.read_csv(fpath)
            n_reps_col  = df["rep"].nunique() if "rep" in df.columns else "—"
            print(f"  {fname:<50}  {len(df):>6} linhas  reps={n_reps_col}")
        except Exception:
            print(f"  {fname}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run benchmarks")
    parser.add_argument(
        "--benchmark_name",
        type     = str,
        required = True,
        help     = "Name of the benchmark (e.g. CHILD, ASIA, ALARM, HEPAR)",
    )
    args = parser.parse_args()

    main(args.benchmark_name)