import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def mov_on_actual_top_two(df_slice: pd.DataFrame, pred_col: str, actual_col: str) -> float:
    """
    Margin of victory using the two candidates with highest actual results.
    Uses predicted values for those two to compute predicted gap; NaN if fewer than two.
    """
    top_two_actual = df_slice.nlargest(2, actual_col)
    if len(top_two_actual) < 2:
        return np.nan
    values = top_two_actual[pred_col].to_numpy()
    return float(values[0] - values[1])


def compute_metrics(
    df_slice: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    candidate_col: str = "Candidate",
) -> dict[str, float]:
    errors = df_slice[pred_col] - df_slice[actual_col]
    abs_error = errors.abs()
    mae = abs_error.mean()
    rmse = np.sqrt((errors**2).mean())
    mape = (abs_error / df_slice[actual_col]).replace([np.inf, -np.inf], np.nan).mean() * 100
    wape = abs_error.sum() / df_slice[actual_col].sum() * 100

    mov_df = df_slice
    if candidate_col in df_slice.columns:
        mov_df = df_slice[
            ~df_slice[candidate_col].str.contains("Others", case=False, na=False)
        ]

    actual_mov = mov_on_actual_top_two(mov_df, pred_col=actual_col, actual_col=actual_col)
    pred_mov = mov_on_actual_top_two(mov_df, pred_col=pred_col, actual_col=actual_col)
    mov_error = (
        abs(pred_mov - actual_mov)
        if not np.isnan(pred_mov) and not np.isnan(actual_mov)
        else np.nan
    )

    return {
        "Candidates Used": len(df_slice),
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "MAPE (%)": round(mape, 3),
        "WAPE (%)": round(wape, 3),
        "Margin of Victory Error": round(mov_error, 3) if not np.isnan(mov_error) else np.nan,
    }


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


def render_line_plot(title: str, df: pd.DataFrame, candidate_col: str) -> None:
    plot_df = df.sort_values("Actual", ascending=False).set_index(candidate_col)
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
    plt.title(title)
    plt.xlabel(f"{candidate_col} candidate")
    plt.ylabel("Vote Share")
    plt.xticks(rotation=45, ha="right")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def main() -> None:
    df = pd.read_csv("Jncsu election - Sheet1.csv")

    sections = [
        {
            "label": "VP",
            "candidate_col": "Candidate (VP)",
            "actual_col": "Actual (VP)",
            "survey_cols": {
                "Socchar 1": "Socchar 1 (VP)",
                "Socchar 2": "Socchar 2 (VP)",
                "Narrativ": "Narrativ (VP)",
            },
        },
        {
            "label": "GS",
            "candidate_col": "Candidate (GS)",
            "actual_col": "Actual (GS)",
            "survey_cols": {
                "Socchar 1": "Socchar 1 (GS)",
                "Socchar 2": "Socchar 2 (GS)",
                "Narrativ": "Narrativ (GS)",
            },
        },
        {
            "label": "AGS",
            "candidate_col": "Candidate (AGS)",
            "actual_col": "Actual (AGS)",
            "survey_cols": {
                "Socchar 1": "Socchar 1 (AGS)",
                "Socchar 2": "Socchar 2 (AGS)",
                "Narrativ": "Narrativ (AGS)",
            },
        },
    ]

    all_rows = []
    for section in sections:
        base_cols = [section["candidate_col"], section["actual_col"]]
        section_df = df[base_cols].rename(
            columns={
                section["candidate_col"]: "Candidate",
                section["actual_col"]: "Actual",
            }
        )
        if section_df.dropna(subset=["Actual"]).empty:
            print(f"{section['label']}: no data after filtering.")
            continue

        section_results = []
        for survey_label, survey_col in section["survey_cols"].items():
            survey_df = section_df.copy()
            survey_df["Predicted"] = df[survey_col]
            survey_df = survey_df.dropna(subset=["Predicted", "Actual"])
            if survey_df.empty:
                continue
            metrics = compute_metrics(survey_df, pred_col="Predicted", actual_col="Actual")
            section_results.append({"Survey": survey_label, **metrics})

            all_rows.append(
                {
                    "Office": section["label"],
                    "Survey": survey_label,
                    "Data": survey_df,
                }
            )

        if section_results:
            out_df = pd.DataFrame(section_results)
            print(f"{section['label']} candidates:", ", ".join(section_df["Candidate"]))
            print(out_df)
            render_table(f"{section['label']} election", out_df)

            plot_cols = ["Actual"] + list(section["survey_cols"].values())
            plot_df = df[
                [section["candidate_col"], section["actual_col"]] + plot_cols[1:]
            ].rename(
                columns={
                    section["candidate_col"]: "Candidate",
                    section["actual_col"]: "Actual",
                    **{col: label for label, col in section["survey_cols"].items()},
                }
            )
            plot_df = plot_df.dropna(subset=["Actual"])
            if not plot_df.empty:
                render_line_plot(
                    f"{section['label']} candidates: Actual vs Survey Results",
                    plot_df,
                    "Candidate",
                )

    if all_rows:
        total_rows = []
        for survey_label in {row["Survey"] for row in all_rows}:
            combined = pd.concat(
                [row["Data"] for row in all_rows if row["Survey"] == survey_label],
                ignore_index=True,
            )
            metrics = compute_metrics(combined, pred_col="Predicted", actual_col="Actual")
            metrics.pop("Margin of Victory Error", None)
            total_rows.append({"Survey": survey_label, **metrics})
        total_df = pd.DataFrame(total_rows)
        print("Total candidates:", ", ".join(pd.concat([row["Data"] for row in all_rows])["Candidate"]))
        print(total_df)
        render_table("Total candidates", total_df)

if __name__ == "__main__":
    main()
