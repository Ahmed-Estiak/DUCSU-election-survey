import pandas as pd
import numpy as np

# === 1. Load data ===
df = pd.read_csv("DUCSU VP - Sheet1.csv")

# === 2. Define core candidates (actual individuals) ===
core = ["Sadik", "Abid", "Kader", "Umama", "Shamim"]
surveys = ["Dessent", "Narrativ", "Socchar", "BPA"]

results = []

# === 3. Compute actual ranks ===
actual_df = df[df["VP Candidate"].isin(core)][["VP Candidate", "Actual"]].copy()
actual_df["actual_rank"] = actual_df["Actual"].rank(ascending=False, method="dense")

# === 4. Loop through each survey ===
for s in surveys:
    poll_df = df[df["VP Candidate"].isin(core)][["VP Candidate", "Actual", s]].copy()

    # Remove candidates missing in that survey (e.g., Shamim missing in BPA)
    poll_df = poll_df.dropna(subset=[s])

    # Error = predicted - actual
    poll_df["error"] = poll_df[s] - poll_df["Actual"]

    # Metrics
    mae = poll_df["error"].abs().mean()
    rmse = np.sqrt((poll_df["error"]**2).mean())
    variance = poll_df["error"].var(ddof=0)

    # Ranks
    poll_df["pred_rank"] = poll_df[s].rank(ascending=False, method="dense")

    merged = poll_df.merge(actual_df[["VP Candidate", "actual_rank"]], on="VP Candidate")

    # Sum of absolute rank differences
    roe = (merged["pred_rank"] - merged["actual_rank"]).abs().sum()

    results.append({
        "Survey": s,
        "Candidates Used": len(poll_df),
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "Variance": round(variance, 3),
        "Rank Order Error": int(roe)
    })

# === 5. Final result table ===
results_df = pd.DataFrame(results)
print(results_df)
