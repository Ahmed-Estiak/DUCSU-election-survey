import pandas as pd
import numpy as np


def main() -> None:
    df = pd.read_csv("DUCSU VP - Sheet1.csv")

    # Use the three common candidates and add an Others+ row
    core_candidates = ["Sadik", "Abid", "Kader"]
    survey_cols = ["Dessent", "Narrativ", "Socchar", "BPA"]

    def build_others_plus() -> pd.DataFrame:
        """Fill Others+ with existing values; otherwise sum Shamim, Umama, Others."""
        base_row = df[df["VP Candidate"] == "Others+"]
        sums = df[df["VP Candidate"].isin(["Shamim", "Umama", "Others"])]
        row = {"VP Candidate": "Others+"}
        for col in survey_cols + ["Actual"]:
            if not base_row.empty and pd.notna(base_row.iloc[0][col]):
                row[col] = float(base_row.iloc[0][col])
            else:
                row[col] = float(sums[col].sum(skipna=True))
        return pd.DataFrame([row])

    others_plus_df = build_others_plus()
    # Show the constructed Others+ values for transparency
    others_plus_row = others_plus_df.iloc[0]
    print("Others+ values:")
    for col in survey_cols + ["Actual"]:
        print(f"  {col}: {others_plus_row[col]}")

    def mov_on_actual_top_two(df_slice: pd.DataFrame, pred_col: str) -> float:
        """
        Margin of victory using the two candidates with highest actual results.
        Others+ is excluded from this calculation. Uses predicted values for those two
        to compute predicted gap; NaN if fewer than two remain.
        """
        filtered = df_slice[df_slice["VP Candidate"] != "Others+"]
        top_two_actual = filtered.nlargest(2, "Actual")
        if len(top_two_actual) < 2:
            return np.nan
        values = top_two_actual[pred_col].to_numpy()
        return float(values[0] - values[1])

    common_df = pd.concat(
        [df[df["VP Candidate"].isin(core_candidates)], others_plus_df],
        ignore_index=True,
    )

    results = []
    for survey in survey_cols:
        errors = common_df[survey] - common_df["Actual"]
        abs_error = errors.abs()
        mae = abs_error.mean()
        rmse = np.sqrt((errors ** 2).mean())
        variance = errors.var(ddof=0)
        mape = (abs_error / common_df["Actual"]).mean() * 100
        wape = abs_error.sum() / common_df["Actual"].sum() * 100

        actual_mov = mov_on_actual_top_two(common_df, "Actual")
        pred_mov = mov_on_actual_top_two(common_df, survey)
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
                "Variance": round(variance, 3),
            }
        )

    results_df = pd.DataFrame(results)
    print("Candidates:", ", ".join(common_df["VP Candidate"]))
    print(results_df)


if __name__ == "__main__":
    main()
