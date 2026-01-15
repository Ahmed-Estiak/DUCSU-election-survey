import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    import statsmodels.api as sm
except ImportError as exc:
    raise SystemExit(
        "statsmodels is required. Install it and re-run."
    ) from exc


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


def base_scatter(ax: plt.Axes, df: pd.DataFrame) -> None:
    ax.scatter(
        df["Time"],
        df["Percentage"],
        s=70,
        color="#1d4ed8",
        edgecolors="white",
        linewidths=0.7,
        zorder=3,
        label="Survey",
    )
    for _, row in df.iterrows():
        ax.annotate(
            row["Survey Name"],
            (row["Time"], row["Percentage"]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(lambda val, pos: f"{val:.0f}%")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    y = df["Percentage"].to_numpy()
    y_min = y.min()
    y_max = y.max()
    padding = max((y_max - y_min) * 0.12, 1.5)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_xlabel("Time")
    ax.set_ylabel("Percentage")
    plt.xticks(rotation=30, ha="right")


def plot_linear_with_ci(df: pd.DataFrame) -> None:
    plot_df = df[~df["Survey Name"].str.contains("Prothom Alo", case=False, na=False)]
    if plot_df.empty:
        print("Skipping plot: no data after excluding Prothom Alo.")
        return
    x = mdates.date2num(plot_df["Time"])
    y = plot_df["Percentage"].to_numpy()
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const).fit()
    x_grid = np.linspace(x.min(), x.max(), 200)
    pred = model.get_prediction(sm.add_constant(x_grid)).summary_frame(alpha=0.05)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-whitegrid")
    base_scatter(ax, plot_df)
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
    ax.set_title("Linear trend with 95% CI")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()



def main() -> None:
    df = pd.read_csv("NCP survey - Sheet1.csv")
    df["Time"] = pd.to_datetime(df["Time"], format="%m/%d/%Y", errors="coerce")
    df["Percentage"] = pd.to_numeric(df["Percentage"], errors="coerce")
    df = df.dropna(subset=["Survey Name", "Time", "Percentage"]).sort_values("Time")

    table_df = df.copy()
    table_df["Percentage"] = table_df["Percentage"].map(lambda v: f"{v:.2f}%")
    table_df["Time"] = table_df["Time"].dt.strftime("%m/%d/%Y")
    render_table("NCP survey data", table_df[["Survey Name", "Percentage", "Time"]])

    plot_linear_with_ci(df)


if __name__ == "__main__":
    main()
