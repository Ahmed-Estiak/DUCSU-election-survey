import sys

import numpy as np
import pandas as pd


def _try_pearsonr(x: np.ndarray, y: np.ndarray):
    try:
        from scipy.stats import pearsonr  # type: ignore

        r, p = pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        r = np.corrcoef(x, y)[0, 1]
        return float(r), float("nan")


def main() -> int:
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "Fore py1 - Sheet1.csv"
    df = pd.read_csv(csv_path)
    if "Division" not in df.columns and "Party (Division)" in df.columns:
        df["Division"] = df["Party (Division)"].str.extract(r"\(([^)]+)\)")
    if "Division" not in df.columns:
        print("Missing 'Division' column for clustered SE.")
        return 1
    if "Party (Division)" in df.columns:
        df = df[~df["Party (Division)"].isin(["BNP (Rangpur)", "BNP (Khulna)"])]

    # Linear regression and scatter plot
    divisions = ["Rajshahi", "Chittagong", "Barisal", "Dhaka", "Mymensingh", "Sylhet"]
    reg_df = df[df["Division"].isin(divisions)].copy()
    if reg_df.empty:
        print("No matching divisions found for regression plot.")
        return 0

    x = reg_df["Survey BAL"].to_numpy()
    y = reg_df["Actual-Survey BNP"].to_numpy()
    pearson_r, pearson_p = _try_pearsonr(x, y)
    pearson_r2 = pearson_r * pearson_r if np.isfinite(pearson_r) else float("nan")

    from scipy.stats import linregress, spearmanr  # type: ignore

    import statsmodels.api as sm

    X = sm.add_constant(reg_df["Survey BAL"])
    y_sm = reg_df["Actual-Survey BNP"]
    cluster_groups = reg_df["Division"]
    clustered_model = sm.OLS(y_sm, X).fit(
        cov_type="cluster", cov_kwds={"groups": cluster_groups}
    )
    print()
    print("Clusters (by Division):")
    for division, group in reg_df.groupby("Division"):
        rows = ", ".join(group["Party (Division)"].astype(str).tolist())
        print(f"- {division}: {rows}")
    print()
    print()
    print("Clustered SE regression summary (grouped by Division):")
    print(clustered_model.summary())

    result = linregress(x, y)
    slope = float(result.slope)
    intercept = float(result.intercept)
    r_value = float(result.rvalue)
    p_value = float(result.pvalue)
    r_squared = r_value * r_value
    y_pred = intercept + slope * x
    mae = float(np.mean(np.abs(y - y_pred)))
    see = float(np.sqrt(np.sum((y - y_pred) ** 2) / (len(x) - 2)))
    spearman_rho, _spearman_p = spearmanr(x, y)

    print("X Y     r        R^2       p-value")
    print(
        f"Survey BAL Actual-Survey BNP     {pearson_r:.4f}   {pearson_r2:.4f}   {pearson_p:.6f}"
    )
    print()
    print("Metric   Value")
    print(f"MAE      {mae:.6f}")
    print(f"Spearman rho   {float(spearman_rho):.6f}")
    print(f"SEE      {see:.6f}")

    print()
    print(f"Slope (Survey BAL -> Actual-Survey BNP): {slope:.6f}")
    print(f"P-value (linregress): {p_value:.6g}")

    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import t  # type: ignore

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.scatterplot(
        x="Survey BAL",
        y="Actual-Survey BNP",
        data=reg_df,
        ax=ax,
        s=70,
        alpha=0.95,
        hue="Division",
        palette=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
        edgecolor="black",
        linewidth=0.4,
    )

    # Regression line + 95% CI based on standard error (mean response)
    x_mean = x.mean()
    ssx = np.sum((x - x_mean) ** 2)
    y_hat = intercept + slope * x
    s_err = np.sqrt(np.sum((y - y_hat) ** 2) / (len(x) - 2))

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = intercept + slope * x_line
    se_fit = s_err * np.sqrt(1 / len(x) + ((x_line - x_mean) ** 2) / ssx)
    t_val = t.ppf(1 - (1 - 0.95) / 2, df=len(x) - 2)
    ci_upper = y_line + t_val * se_fit
    ci_lower = y_line - t_val * se_fit

    ax.plot(x_line, y_line, color="black", lw=2)
    ax.fill_between(x_line, ci_lower, ci_upper, color="#1f77b4", alpha=0.2)

    # Highlight Dhaka
    dhaka = reg_df[reg_df["Division"] == "Dhaka"]
    if not dhaka.empty:
        ax.scatter(
            dhaka["Survey BAL"],
            dhaka["Actual-Survey BNP"],
            s=120,
            color="#d62728",
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )

    for _, row in reg_df.iterrows():
        label = row["Division"]
        dx = 0.3
        dy = 0.3
        ax.text(
            row["Survey BAL"] + dx,
            row["Actual-Survey BNP"] + dy,
            label,
            fontsize=9,
            color="#d62728" if label == "Dhaka" else "#222222",
            weight="bold" if label == "Dhaka" else "normal",
        )

    eq_text = f"y = {slope:.3f}x + {intercept:.3f}"
    stats_text = f"$R^2$ = {r_squared:.3f}, p = {p_value:.3g}"
    ax.set_title(f"{eq_text} | {stats_text}")
    ax.set_xlabel("Survey BAL")
    ax.set_ylabel("Actual-Survey BNP")
    ax.text(
        0.98,
        0.02,
        "95% CI",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=27,
        color="#444444",
    )

    plt.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
