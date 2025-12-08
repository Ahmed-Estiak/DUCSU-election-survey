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

    results = []
    for survey in survey_cols:
        errors = common_df[survey] - common_df["Actual"]
        mae = errors.abs().mean()
        rmse = np.sqrt((errors ** 2).mean())
        variance = errors.var(ddof=0)

        results.append(
            {
                "Survey": survey,
                "Candidates Used": len(common_df),
                "MAE": round(mae, 3),
                "RMSE": round(rmse, 3),
                "Variance": round(variance, 3),
            }
        )

    results_df = pd.DataFrame(results)
    print("Common candidates:", ", ".join(common_df["VP Candidate"]))
    print(results_df)


if __name__ == "__main__":
    main()
