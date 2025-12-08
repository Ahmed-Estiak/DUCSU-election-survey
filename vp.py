import pandas as pd
import numpy as np

# === 1. Load data ===
df = pd.read_csv("DUCSU VP - Sheet1.csv")

# === 2. Define core candidates (actual individuals) ===
core = ["Sadik", "Abid", "Kader", "Umama", "Shamim"]
surveys = ["Dessent", "Narrativ", "Socchar", "BPA"]

results = []

# === 4. Loop through each survey ===
for s in surveys:
    poll_df = df[df["VP Candidate"].isin(core)][["VP Candidate", "Actual", s]].copy()

    # Remove candidates missing in that survey (e.g., Shamim missing in BPA)
    poll_df = poll_df.dropna(subset=[s])

    # Error = predicted - actual
    poll_df["error"] = poll_df[s] - poll_df["Actual"]
    abs_error = poll_df["error"].abs()

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

    # Metrics
    mae = abs_error.mean()
    rmse = np.sqrt((poll_df["error"]**2).mean())
    variance = poll_df["error"].var(ddof=0)
    mape = (abs_error / poll_df["Actual"]).mean() * 100
    wape = abs_error.sum() / poll_df["Actual"].sum() * 100

    actual_mov = mov_on_actual_top_two(poll_df, "Actual")
    pred_mov = mov_on_actual_top_two(poll_df, s)
    mov_error = abs(pred_mov - actual_mov) if not np.isnan(pred_mov) and not np.isnan(actual_mov) else np.nan

    results.append({
        "Survey": s,
        "Candidates Used": len(poll_df),
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "MAPE (%)": round(mape, 3),
        "WAPE (%)": round(wape, 3),
        "Margin of Victory Error": round(mov_error, 3) if not np.isnan(mov_error) else np.nan,
        "Variance": round(variance, 3)
    })

# === 5. Final result table ===
results_df = pd.DataFrame(results)
print(results_df)
