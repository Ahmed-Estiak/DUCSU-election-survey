import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    df = pd.read_csv("DUCSU AGS - Sheet1.csv")

    survey_cols = ["Dessent", "Narrativ"]
    common_df = df[df[survey_cols].notna().all(axis=1)][
        ["AGS candidate", "Actual"] + survey_cols
    ].copy()

    if common_df.empty:
        raise SystemExit("No common candidates between Dessent and Narrativ.")

    def mov_on_actual_top_two(df_slice: pd.DataFrame, pred_col: str) -> float:
        """
        Margin of victory using the two candidates with highest actual results.
        Uses predicted values for those two to compute predicted gap; NaN if fewer than two.
        """
        top_two_actual = df_slice.nlargest(2, "Actual")
        if len(top_two_actual) < 2:
            return np.nan
        values = top_two_actual[pred_col].to_numpy()
        return float(values[0] - values[1])

    results = []
    for survey in survey_cols:
        errors = common_df[survey] - common_df["Actual"]
        abs_error = errors.abs()
        mae = abs_error.mean()
        rmse = np.sqrt((errors ** 2).mean())
        mape = (abs_error / common_df["Actual"]).mean() * 100
        wape = abs_error.sum() / common_df["Actual"].sum() * 100

        mov_df = common_df[common_df["AGS candidate"] != "Others+"]
        actual_mov = mov_on_actual_top_two(mov_df, "Actual")
        pred_mov = mov_on_actual_top_two(mov_df, survey)
        mov_error = (
            abs(pred_mov - actual_mov)
            if not np.isnan(pred_mov) and not np.isnan(actual_mov)
            else np.nan
        )

        results.append(
            {
                "Survey": survey,
                "Candidates Used": len(common_df),
                "MAE": round(mae, 3),
                "RMSE": round(rmse, 3),
                "MAPE (%)": round(mape, 3),
                "WAPE (%)": round(wape, 3),
                "Margin of Victory Error": round(mov_error, 3)
                if not np.isnan(mov_error)
                else np.nan,
            }
        )

    results_df = pd.DataFrame(results)
    print("Common candidates:", ", ".join(common_df["AGS candidate"]))
    print(results_df)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")
    table = ax.table(
        cellText=results_df.values,
        colLabels=results_df.columns,
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
    ax.set_title("DUCSU AGS election", pad=12, fontweight="bold")
    fig.set_facecolor("white")
    plt.tight_layout()
    plt.show()
    plot_df = (
        common_df.sort_values("Actual", ascending=False)
        .set_index("AGS candidate")[["Actual"] + survey_cols]
    )
    plt.figure(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-whitegrid")
    for col in plot_df.columns:
        plt.plot(
            plot_df.index,
            plot_df[col],
            marker="o",
            markersize=7,
            linewidth=2,
            label=col,
        )
    plt.title("AGS Common Candidates: Actual vs Survey Results")
    plt.xlabel("AGS candidate")
    plt.ylabel("Vote Share")
    plt.xticks(rotation=45, ha="right")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
