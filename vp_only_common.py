import pandas as pd
import numpy as np


def main() -> None:
    df = pd.read_csv("DUCSU VP - Sheet1.csv")

    core_candidates = ["Sadik", "Abid", "Kader", "Umama", "Shamim"]
    survey_cols = ["Dessent", "Narrativ", "Socchar", "BPA"]

    # Keep only the actual candidates and restrict to those present in every survey
    core_df = df[df["VP Candidate"].isin(core_candidates)].copy()
    common_df = core_df[core_df[survey_cols].notna().all(axis=1)]

    if common_df.empty:
        raise SystemExit("No common candidates across all surveys.")

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
    print("Common candidates:", ", ".join(common_df["VP Candidate"]))
    print(results_df)


if __name__ == "__main__":
    main()
