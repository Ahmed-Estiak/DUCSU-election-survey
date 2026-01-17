import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    import statsmodels.api as sm
except ImportError as exc:
    raise SystemExit("statsmodels is required. Install it and re-run.") from exc


def render_table(title: str, df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        edges="closed",
    )
    header_color = "#1f2937"
    row_colors = ["#f8fafc", "#eef2f7"]
    text_color = "#111827"
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor(row_colors[(row - 1) % 2])
            cell.set_text_props(color=text_color)
    ax.set_title(title, pad=12, fontweight="bold")
    fig.set_facecolor("white")
    plt.tight_layout()
    plt.show()


def plot_trend(
    df: pd.DataFrame,
    value_col: str,
    label: str,
    exclude_surveys: list[str] | None = None,
) -> tuple[float, float]:
    plot_df = df
    if exclude_surveys:
        plot_df = df[~df["Survey Name"].isin(exclude_surveys)]
    if plot_df.empty:
        print(f"Skipping plot for {label}: no data after exclusions.")
        return float("nan"), float("nan")
    x = mdates.date2num(plot_df["Time"])
    y = plot_df[value_col].to_numpy()
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const).fit()
    x_grid = np.linspace(x.min(), x.max(), 200)
    pred = model.get_prediction(sm.add_constant(x_grid)).summary_frame(alpha=0.05)
    latest_x = x.max()
    latest_pred = float(model.predict(np.array([[1.0, latest_x]]))[0])
    slope = float(model.params[1])

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-whitegrid")
    ax.scatter(
        plot_df["Time"],
        y,
        s=70,
        color="#1d4ed8",
        edgecolors="white",
        linewidths=0.7,
        zorder=3,
        label="Survey",
    )
    for _, row in plot_df.iterrows():
        ax.annotate(
            row["Survey Name"],
            (row["Time"], row[value_col]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )

    ax.plot(
        mdates.num2date(x_grid),
        pred["mean"],
        color="#f97316",
        linewidth=2.5,
        label="Linear trend",
    )
    ax.fill_between(
        mdates.num2date(x_grid),
        pred["mean_ci_lower"],
        pred["mean_ci_upper"],
        color="#f97316",
        alpha=0.2,
        label="95% CI",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(lambda val, pos: f"{val:.0f}%")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    y_min = y.min()
    y_max = y.max()
    padding = max((y_max - y_min) * 0.12, 1.5)
    ax.set_ylim(y_min - padding, y_max + padding)
    plt.xticks(rotation=30, ha="right")
    ax.set_title("Linear trend + 95% CI")
    ax.set_xlabel("Time")
    ax.set_ylabel(label)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()
    return latest_pred, slope


def build_filtered_df(
    df: pd.DataFrame, label: str, mask: pd.Series, party_label: str
) -> pd.DataFrame:
    filtered = df[mask].copy().sort_values("Time")
    if filtered.empty:
        print(f"No {party_label} entries for {label}.")
    return filtered


def render_section(
    df: pd.DataFrame,
    title: str,
    value_col: str,
    label: str,
    exclude_surveys: list[str] | None = None,
) -> None:
    if df.empty:
        return
    plot_df = df.copy()
    if exclude_surveys:
        plot_df = plot_df[~plot_df["Survey Name"].isin(exclude_surveys)]
    if plot_df.empty:
        print(f"No {label} entries after exclusions.")
        return
    latest_time = plot_df["Time"].max()
    if isinstance(latest_time, str):
        latest_time = pd.to_datetime(latest_time, errors="coerce")
    latest_time_str = (
        latest_time.strftime("%m/%d/%Y") if pd.notna(latest_time) else "Unknown"
    )
    table_df = plot_df.copy()
    table_df[value_col] = table_df[value_col].map(lambda v: f"{v:.2f}%")
    table_df["Time"] = table_df["Time"].dt.strftime("%m/%d/%Y")
    render_table(title, table_df[["Survey Name", value_col, "Time"]])
    trend_value, slope = plot_trend(
        plot_df, value_col, label, exclude_surveys=exclude_surveys
    )
    if not np.isnan(trend_value):
        print(f"Linear trend {label} ({latest_time_str}): {trend_value:.2f}%")
        print(f"Slope {label}: {slope:.4f} percentage points/day")


def main() -> None:
    df = pd.read_csv("All survey - Sheet1.csv")
    time_clean = df["Time"].astype(str).str.strip()
    time_clean = time_clean.str.replace(
        r"^(\d{1,2})/(\d{4})$", r"\1/01/\2", regex=True
    )
    df["Time"] = pd.to_datetime(time_clean, format="%m/%d/%Y", errors="coerce")
    df["BNP percentage"] = pd.to_numeric(df["BNP percentage"], errors="coerce")
    df["Jamat percentage"] = pd.to_numeric(df["Jamat percentage"], errors="coerce")
    df["IAB+ percentage"] = pd.to_numeric(df["IAB+ percentage"], errors="coerce")
    df["Jatiyo Party Percentage"] = pd.to_numeric(
        df["Jatiyo Party Percentage"], errors="coerce"
    )
    df["BAL percentage"] = pd.to_numeric(df["BAL percentage"], errors="coerce")
    df = df.dropna(subset=["Survey Name", "Time"])

    between_mask = (df["BNP percentage"] > 30) & (df["BNP percentage"] < 45)
    between_df = build_filtered_df(df, "30% < BNP < 45%", between_mask, "BNP")
    render_section(
        between_df,
        "BNP survey data (30% < BNP < 45%)",
        "BNP percentage",
        "BNP percentage",
    )

    above_mask = df["BNP percentage"] > 55
    above_df = build_filtered_df(df, "BNP > 55%", above_mask, "BNP")
    render_section(above_df, "BNP survey data (BNP > 55%)", "BNP percentage", "BNP percentage")

    jamat_mask = (df["Jamat percentage"] > 20) & (df["Jamat percentage"] < 45)
    jamat_df = build_filtered_df(df, "20% < Jamat < 45%", jamat_mask, "Jamat")
    render_section(
        jamat_df,
        "Jamat survey data (20% < Jamat < 45%)",
        "Jamat percentage",
        "Jamat percentage",
    )

    iab_df = df.dropna(subset=["IAB+ percentage"]).copy().sort_values("Time")
    render_section(
        iab_df,
        "IAB+ survey data (full range, Prothom Alo excluded)",
        "IAB+ percentage",
        "IAB+ percentage",
    )

    jatiyo_df = df.dropna(subset=["Jatiyo Party Percentage"]).copy().sort_values("Time")
    render_section(
        jatiyo_df,
        "Jatiyo Party survey data (full range, Prothom Alo excluded)",
        "Jatiyo Party Percentage",
        "Jatiyo Party percentage",
    )

    bal_df = df.dropna(subset=["BAL percentage"]).copy().sort_values("Time")
    render_section(
        bal_df,
        "BAL survey data (full range, Prothom Alo excluded)",
        "BAL percentage",
        "BAL percentage",
        exclude_surveys=["Prothom Alo"],
    )


if __name__ == "__main__":
    main()
