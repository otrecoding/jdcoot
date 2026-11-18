# Generate the Quarto report's alpha-sweep and synthetic-experiment comparison plots.
# Convert JSON Lines results in --data-dir to tab-delimited CSV, then save PNGs to
# --output-dir. Run from the repository root, for example:
# python experiments/plot_experiments.py --data-dir data --output-dir plots

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FIGSIZE = (8, 6)


def convert_json_results(data_dir):
    """Convert each JSON Lines results file to the tab-delimited schema used below."""
    for json_file in sorted(data_dir.glob("*.json")):
        with json_file.open(encoding="utf-8") as file:
            rows = [json.loads(line) for line in file]
        data = pd.DataFrame(rows)
        if "learning" not in data or "recoding" not in data:
            raise ValueError(
                f"{json_file} must include 'learning' and 'recoding' fields"
            )

        data.loc[data.learning == "unsupervised", "prop_source"] = 1.0
        data.loc[data.learning == "unsupervised", "prop_target"] = 0.0
        data.loc[data.learning == "semisupervised", "prop_source"] = 1.0
        data.loc[:, "recoding"] = data.recoding.replace(
            {"coot": "COOT", "jdcoot": "JDCOOT"}
        )

        csv_file = json_file.with_suffix(".csv")
        data.to_csv(csv_file, sep="\t", encoding="utf-8", index=False, header=True)
        print(f"Converted {json_file} -> {csv_file}")


def load_results(data_dir, filename, required_columns):
    """Read one tab-delimited results file and validate its plotting columns."""
    path = data_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Results file not found: {path}")

    data = pd.read_csv(path, sep="\t")
    missing_columns = sorted(set(required_columns) - set(data.columns))
    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns: {', '.join(missing_columns)}"
        )
    return data


def save_figure(fig, output_dir, filename):
    """Save and close a plot in the requested output directory."""
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_alpha(data, variable, learning, title, ylabel, output_dir, filename, log_x=False):
    """Plot mean test-target score by alpha for one variable/learning setting."""
    subset = data[
        (data["variable"] == variable) & (data["learning"] == learning)
    ]
    scores = subset.groupby("alpha", sort=True)["test_target"].mean()
    if scores.empty:
        print(f"Skipping {title}: no matching results.")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(scores.index, scores.values)
    ax.set_title(title)
    ax.set_xlabel("alpha")
    ax.set_ylabel(ylabel)
    if log_x:
        ax.set_xscale("log")
    save_figure(fig, output_dir, filename)


def print_sample_counts(data, x_column):
    """Print the number of test-target scores in each plotted group."""
    counts = data.groupby([x_column, "recoding"])["test_target"].count()
    print(f"Samples = {counts}")


def plot_boxplot(
    data,
    *,
    x,
    title,
    xlabel,
    ylabel,
    output_dir,
    filename,
    y_limits=None,
    tick_labels=None,
    axis_title="target",
):
    """Draw a result-comparison boxplot and print its per-group sample counts."""
    if data.empty:
        print(f"Skipping {title}: no matching results.")
        return

    print_sample_counts(data, x)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    order = sorted(data[x].dropna().unique())
    sns.boxplot(
        data=data,
        x=x,
        y="test_target",
        hue="recoding",
        order=order,
        showfliers=False,
        ax=ax,
    )
    fig.suptitle(title)
    ax.set_title(axis_title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title="Methods")
    if y_limits is not None:
        ax.set_ylim(y_limits)
    if tick_labels is not None:
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(tick_labels)
    save_figure(fig, output_dir, filename)


def generate_plots(data_dir, output_dir):
    """Generate all figures from the tab-separated Quarto result files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    convert_json_results(data_dir)

    discrete_alpha = load_results(
        data_dir,
        "alpha_optimization_discrete.csv",
        ["variable", "learning", "alpha", "test_target"],
    )
    continuous_alpha = load_results(
        data_dir,
        "alpha_optimization_continuous.csv",
        ["variable", "learning", "alpha", "test_target"],
    )
    for learning in ("unsupervised", "semisupervised", "partial"):
        plot_alpha(
            discrete_alpha,
            "discrete",
            learning,
            f"discrete-{learning}-jdcoot",
            "accuracy",
            output_dir,
            f"alpha_discrete_{learning}_jdcoot.png",
        )
        plot_alpha(
            continuous_alpha,
            "continuous",
            learning,
            f"continuous-{learning}-jdcoot",
            "MSE",
            output_dir,
            f"alpha_continuous_{learning}_jdcoot.png",
            log_x=True,
        )

    mean_shift = load_results(
        data_dir,
        "mean_shift_variation.csv",
        ["variable", "mean_x_source", "mean_x_target", "test_target", "recoding"],
    )
    continuous = mean_shift[mean_shift["variable"] == "continuous"].copy()
    plot_boxplot(
        continuous,
        x="mean_x_target",
        title="Covariable shift for continuous outcome",
        xlabel=r"$\delta$",
        ylabel="MSE on test dataset",
        output_dir=output_dir,
        filename="mean_shift_continuous.png",
        tick_labels=[str(round(value, 1)) for value in sorted(continuous["mean_x_target"].unique())],
        axis_title="target accuracy",
    )

    discrete = mean_shift[
        (mean_shift["variable"] == "discrete")
        & (mean_shift["mean_x_source"] == 0)
    ].copy()
    discrete["recoding"] = discrete["recoding"].replace(
        {"coot": "COOT", "jdcoot": "JDCOOT"}
    )
    plot_boxplot(
        discrete,
        x="mean_x_target",
        title="Covariate shift for discrete outcome",
        xlabel=r"$\delta$",
        ylabel="Accuracy on test dataset",
        output_dir=output_dir,
        filename="mean_shift_discrete.png",
        y_limits=(0.5, 1),
        tick_labels=[str(round(value, 1)) for value in sorted(discrete["mean_x_target"].unique())],
        axis_title="target accuracy",
    )

    sample_sizes = load_results(
        data_dir,
        "sample_size_variation.csv",
        ["variable", "size_source", "size_target", "test_target", "recoding"],
    )
    sample_sizes["size_ratio"] = sample_sizes["size_source"] / sample_sizes["size_target"]
    for variable, ylabel, title, filename, axis_title in (
        (
            "continuous",
            "MSE on test target dataset ",
            "Sample size ratio variation for continuous outcome",
            "sample_size_ratio_continuous.png",
            "target",
        ),
        (
            "discrete",
            "Accuracy on test target dataset",
            "Sample size variation for discrete outcome",
            "sample_size_ratio_discrete.png",
            "accuracy on test target",
        ),
    ):
        plot_boxplot(
            sample_sizes[sample_sizes["variable"] == variable].copy(),
            x="size_ratio",
            title=title,
            xlabel=r"$n_S/n_T$",
            ylabel=ylabel,
            output_dir=output_dir,
            filename=filename,
            axis_title=axis_title,
        )

    correlations = load_results(
        data_dir,
        "correlation_variation.csv",
        [
            "variable",
            "active_autocorr_source",
            "active_autocorr_target",
            "test_target",
            "recoding",
        ],
    )
    for variable, ylabel, title, filename in (
        (
            "continuous",
            "MSE on test dataset",
            "Correlation matrix shift variation for continuous outcome",
            "correlation_variation_continuous.png",
        ),
        (
            "discrete",
            "Accuracy on test dataset",
            "Correlation matrix shift variation for discrete outcome",
            "correlation_variation_discrete.png",
        ),
    ):
        subset = correlations[
            (correlations["variable"] == variable)
            & (correlations["active_autocorr_source"] == 0.2)
        ].copy()
        plot_boxplot(
            subset,
            x="active_autocorr_target",
            title=title,
            xlabel=r"$\delta$",
            ylabel=ylabel,
            output_dir=output_dir,
            filename=filename,
        )

    sparse_rates = load_results(
        data_dir,
        "sparse_rate_variation.csv",
        ["variable", "sparse_rate", "test_target", "recoding"],
    )
    for variable, ylabel, filename in (
        ("discrete", "Accuracy on test dataset", "sparse_rate_discrete.png"),
        ("continuous", "MSEs on test dataset", "sparse_rate_continuous.png"),
    ):
        plot_boxplot(
            sparse_rates[sparse_rates["variable"] == variable].copy(),
            x="sparse_rate",
            title=f"Sparse rate variation for {variable} outcome",
            xlabel="sp",
            ylabel=ylabel,
            output_dir=output_dir,
            filename=filename,
            y_limits=(0, 1) if variable == "discrete" else None,
        )

    odds_ratios = load_results(
        data_dir,
        "odds_ratio_variations.csv",
        ["odds_ratio_target", "test_target", "recoding"],
    )
    plot_boxplot(
        odds_ratios,
        x="odds_ratio_target",
        title="Odds ratio variation for discrete outcome",
        xlabel="OR",
        ylabel="Accuracy on test dataset",
        output_dir=output_dir,
        filename="odds_ratio_variation_discrete.png",
    )

    r2_results = load_results(
        data_dir,
        "r2_variations.csv",
        ["r2_target", "test_target", "recoding"],
    )
    plot_boxplot(
        r2_results,
        x="r2_target",
        title=r"$R^2$ variation for continuous outcome",
        xlabel="$R^2$",
        ylabel="MSE",
        output_dir=output_dir,
        filename="r2_variation_continuous.png",
    )

    covariates = load_results(
        data_dir,
        "observed_covariates_proportion.csv",
        [
            "variable",
            "obs_covar_prop_source",
            "obs_covar_prop_target",
            "test_target",
            "recoding",
        ],
    )
    for variable, ylabel, title, filename in (
        (
            "continuous",
            "MSE",
            "Proportion of observed active covariate variation for continuous outcome",
            "observed_covariates_continuous.png",
        ),
        (
            "discrete",
            "Accuracy",
            "Proportion of observed active covariate variation for discrete outcome",
            "observed_covariates_discrete.png",
        ),
    ):
        subset = covariates[
            (covariates["variable"] == variable)
            & (
                covariates["obs_covar_prop_source"]
                == covariates["obs_covar_prop_target"]
            )
        ].copy()
        plot_boxplot(
            subset,
            x="obs_covar_prop_source",
            title=title,
            xlabel="$p_x$",
            ylabel=ylabel,
            output_dir=output_dir,
            filename=filename,
        )

    observed_labels = load_results(
        data_dir,
        "observed_labels_proportions_variation.csv",
        ["variable", "learning", "prop_source", "prop_target", "test_target", "recoding"],
    )
    continuous_partial_csv = data_dir / (
        "observed_labels_proportions_variation_continuous.csv"
    )
    if continuous_partial_csv.is_file():
        continuous_partial = load_results(
            data_dir,
            continuous_partial_csv.name,
            [
                "variable",
                "learning",
                "prop_source",
                "prop_target",
                "test_target",
                "recoding",
            ],
        )
    else:
        continuous_partial = observed_labels
        print(
            "Using observed_labels_proportions_variation.csv for continuous "
            "partial-label plots; the separate continuous results file is absent."
        )
    label_plots = (
        (
            observed_labels,
            "discrete",
            "semisupervised",
            None,
            "Proportion of target data observed for discrete outcome",
            "Accuracy",
            "observed_labels_semisupervised_discrete.png",
        ),
        (
            observed_labels,
            "continuous",
            "semisupervised",
            None,
            "Proportion of target data observed for continuous outcome",
            "MSE",
            "observed_labels_semisupervised_continuous.png",
        ),
        (
            observed_labels,
            "discrete",
            "partial",
            0.05,
            "Proportion of target and source data observed for discrete outcome (prop_source = 0.05)",
            "Accuracy",
            "observed_labels_partial_discrete_source_0.05.png",
        ),
        (
            observed_labels,
            "discrete",
            "partial",
            0.5,
            "Proportion of target and source data observed for discrete outcome (prop_source = 0.5)",
            "Accuracy",
            "observed_labels_partial_discrete_source_0.5.png",
        ),
        (
            continuous_partial,
            "continuous",
            "partial",
            0.05,
            "Proportion of target data observed for continuous outcome (prop_source = 0.05)",
            "MSE",
            "observed_labels_partial_continuous_source_0.05.png",
        ),
        (
            continuous_partial,
            "continuous",
            "partial",
            0.5,
            "Proportion of target data observed for continuous outcome (prop_source = 0.5)",
            "MSE",
            "observed_labels_partial_continuous_source_0.5.png",
        ),
    )
    for (
        results,
        variable,
        learning,
        prop_source,
        title,
        ylabel,
        filename,
    ) in label_plots:
        subset = results[
            (results["variable"] == variable)
            & (results["learning"] == learning)
        ].copy()
        if prop_source is not None:
            subset = subset[subset["prop_source"] == prop_source]
        plot_boxplot(
            subset,
            x="prop_target",
            title=title,
            xlabel="$p_o^T$",
            ylabel=ylabel,
            output_dir=output_dir,
            filename=filename,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert JSON Lines results in the data directory, then generate "
            "experiment plots from the resulting tab-delimited CSV files."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help=(
            "Directory containing result CSVs and/or JSON Lines files "
            "(default: ./data). JSON files are converted in place."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory where PNG plots will be saved (default: ./plots).",
    )
    args = parser.parse_args()
    generate_plots(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
