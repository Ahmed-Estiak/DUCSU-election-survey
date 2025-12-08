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

    common_df = pd.concat(
        [df[df["VP Candidate"].isin(core_candidates)], others_plus_df],
        ignore_index=True,
    )

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
    print("Candidates:", ", ".join(common_df["VP Candidate"]))
    print(results_df)


if __name__ == "__main__":
    main()
