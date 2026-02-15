import sys
from itertools import combinations

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


def _binarize_by_median(series: pd.Series) -> np.ndarray:
    return (series >= series.median()).astype(int).to_numpy()


def _f1_score(a: np.ndarray, b: np.ndarray) -> float:
    tp = int(((a == 1) & (b == 1)).sum())
    fp = int(((a == 0) & (b == 1)).sum())
    fn = int(((a == 1) & (b == 0)).sum())
    denom = (2 * tp + fp + fn)
    return float(0.0 if denom == 0 else (2 * tp) / denom)


def main() -> int:
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "Fore py - Sheet1.csv"
    df = pd.read_csv(csv_path)
    df = df[~df["Party (Division)"].isin(["BNP (Rangpur)", "BNP (Khulna)"])]
    df["Division"] = df["Party (Division)"].str.extract(r"\(([^)]+)\)")
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty or num_df.shape[1] < 2:
        print("Need at least two numeric columns to compute metrics.")
        return 1

    print("Correlation matrix (Pearson r):")
    print(num_df.corr().round(4).to_string())
    print()

    rows = []
    for col_a, col_b in combinations(num_df.columns, 2):
        x = num_df[col_a].to_numpy()
        y = num_df[col_b].to_numpy()
        r, p = _try_pearsonr(x, y)
        f1 = _f1_score(_binarize_by_median(num_df[col_a]), _binarize_by_median(num_df[col_b]))
        rows.append(
            {
                "Column A": col_a,
                "Column B": col_b,
                "Pearson r": r,
                "P score (p-value)": p,
                "F1 (median-binary)": f1,
            }
        )

    out = pd.DataFrame(rows)
    out["Pearson r"] = out["Pearson r"].round(4)
    out["P score (p-value)"] = out["P score (p-value)"].round(6)
    out["F1 (median-binary)"] = out["F1 (median-binary)"].round(4)

    print("Pairwise metrics:")
    print(out.to_string(index=False))
    if out["P score (p-value)"].isna().any():
        print()
        print("Note: p-values are NaN because scipy is not available.")

    # Linear regression and scatter plot
    divisions = ["Rajshahi", "Chittagong", "Barisal", "Dhaka", "Mymensingh", "Sylhet"]
    reg_df = df[df["Division"].isin(divisions)].copy()
    if reg_df.empty:
        print("No matching divisions found for regression plot.")
        return 0

    x = reg_df["Survey BAL"].to_numpy()
    y = reg_df["Actual-Survey BNP"].to_numpy()

    from scipy.stats import linregress  # type: ignore

    result = linregress(x, y)
    slope = float(result.slope)
    intercept = float(result.intercept)
    r_value = float(result.rvalue)
    p_value = float(result.pvalue)
    r_squared = r_value * r_value

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
        color="#1f77b4",
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
    ax.text(
        0.98,
        0.02,
        "95% CI",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    ax.set_xlabel("Survey BAL")
    ax.set_ylabel("Actual-Survey BNP")
    ax.set_ylim(0, 25)
    plt.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
