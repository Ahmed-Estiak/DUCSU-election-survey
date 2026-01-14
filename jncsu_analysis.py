import pandas as pd
import numpy as np


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


def compute_metrics(df_slice: pd.DataFrame, pred_col: str, actual_col: str) -> dict[str, float]:
    errors = df_slice[pred_col] - df_slice[actual_col]
    abs_error = errors.abs()
    mae = abs_error.mean()
    rmse = np.sqrt((errors**2).mean())
    variance = errors.var(ddof=0)
    mape = (abs_error / df_slice[actual_col]).replace([np.inf, -np.inf], np.nan).mean() * 100
    wape = abs_error.sum() / df_slice[actual_col].sum() * 100

    actual_mov = mov_on_actual_top_two(df_slice, pred_col=actual_col, actual_col=actual_col)
    pred_mov = mov_on_actual_top_two(df_slice, pred_col=pred_col, actual_col=actual_col)
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
        "Variance": round(variance, 3),
    }


def compute_mov_error(df_slice: pd.DataFrame, pred_col: str, actual_col: str) -> float:
    actual_mov = mov_on_actual_top_two(df_slice, pred_col=actual_col, actual_col=actual_col)
    pred_mov = mov_on_actual_top_two(df_slice, pred_col=pred_col, actual_col=actual_col)
    if np.isnan(actual_mov) or np.isnan(pred_mov):
        return np.nan
    return round(abs(pred_mov - actual_mov), 3)


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
    mov_rows = []
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

            no_others_df = survey_df[
                ~survey_df["Candidate"].str.contains("Others", case=False, na=False)
            ].copy()
            mov_error = (
                compute_mov_error(no_others_df, pred_col="Predicted", actual_col="Actual")
                if not no_others_df.empty
                else np.nan
            )
            mov_rows.append(
                {
                    "Office": section["label"],
                    "Survey": survey_label,
                    "Margin of Victory Error": mov_error,
                }
            )

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

    if all_rows:
        total_rows = []
        for survey_label in {row["Survey"] for row in all_rows}:
            combined = pd.concat(
                [row["Data"] for row in all_rows if row["Survey"] == survey_label],
                ignore_index=True,
            )
            metrics = compute_metrics(combined, pred_col="Predicted", actual_col="Actual")
            total_rows.append({"Survey": survey_label, **metrics})
        total_df = pd.DataFrame(total_rows)
        print("Total candidates:", ", ".join(pd.concat([row["Data"] for row in all_rows])["Candidate"]))
        print(total_df)

    mov_df = pd.DataFrame(mov_rows)
    print("Margin of Victory Error without Others/Others+:")
    print(mov_df)


if __name__ == "__main__":
    main()
