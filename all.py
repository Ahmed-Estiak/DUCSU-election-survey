import pandas as pd
import numpy as np


def main() -> None:
    df = pd.read_csv("DUCSU ALL - Sheet1.csv")

    survey_cols = ["Dessent", "Narrativ"]
    common_df = df[df[survey_cols].notna().all(axis=1)][
        ["Candidate", "Actual"] + survey_cols
    ].copy()

    if common_df.empty:
        raise SystemExit("No common candidates between Dessent and Narrativ.")

    results = []
    for survey in survey_cols:
        errors = common_df[survey] - common_df["Actual"]
        abs_error = errors.abs()
        mae = abs_error.mean()
        rmse = np.sqrt((errors ** 2).mean())
        mape = (abs_error / common_df["Actual"]).mean() * 100
        wape = abs_error.sum() / common_df["Actual"].sum() * 100

        results.append(
            {
                "Survey": survey,
                "Candidates Used": len(common_df),
                "MAE": round(mae, 3),
                "RMSE": round(rmse, 3),
                "MAPE (%)": round(mape, 3),
                "WAPE (%)": round(wape, 3),
            }
        )

    results_df = pd.DataFrame(results)
    print("Common candidates:", ", ".join(common_df["Candidate"]))
    print(results_df)

    excl_mask = ~common_df["Candidate"].str.contains("Others", case=False, na=False)
    excl_df = common_df[excl_mask].copy()
    if excl_df.empty:
        print("Error metrics without Others/Others+: no candidates after exclusion.")
        return

    excl_results = []
    for survey in survey_cols:
        errors = excl_df[survey] - excl_df["Actual"]
        abs_error = errors.abs()
        mae = abs_error.mean()
        rmse = np.sqrt((errors ** 2).mean())
        mape = (abs_error / excl_df["Actual"]).mean() * 100
        wape = abs_error.sum() / excl_df["Actual"].sum() * 100
        excl_results.append(
            {
                "Survey": survey,
                "Candidates Used": len(excl_df),
                "MAE": round(mae, 3),
                "RMSE": round(rmse, 3),
                "MAPE (%)": round(mape, 3),
                "WAPE (%)": round(wape, 3),
            }
        )

    excl_results_df = pd.DataFrame(excl_results)
    print("Error metrics without Others/Others+:")
    print(excl_results_df)


if __name__ == "__main__":
    main()
