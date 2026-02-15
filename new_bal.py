import sys

import numpy as np
import pandas as pd
from scipy.stats import t  # type: ignore


def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    w = w.astype(float)
    w = w / w.sum()
    mx = np.sum(w * x)
    my = np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my))
    vx = np.sum(w * (x - mx) ** 2)
    vy = np.sum(w * (y - my) ** 2)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return float(cov / np.sqrt(vx * vy))


def weighted_corr_pvalue(r: float, w: np.ndarray) -> float:
    # Approximate p-value using effective sample size.
    w = w.astype(float)
    n_eff = (w.sum() ** 2) / np.sum(w ** 2)
    if not np.isfinite(r) or n_eff <= 2:
        return float("nan")
    t_stat = r * np.sqrt((n_eff - 2) / (1 - r**2))
    return float(2 * t.sf(np.abs(t_stat), df=n_eff - 2))


def main() -> int:
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "Fore py - Sheet1.csv"
    df = pd.read_csv(csv_path)
    df = df.head(6)

    x = df["Survey BAL"].to_numpy()
    y = df["Actual-Survey BNP"].to_numpy()
    w = df[" Seat Number"].to_numpy()

    r = weighted_corr(x, y, w)
    r2 = r * r if np.isfinite(r) else float("nan")
    p_value = weighted_corr_pvalue(r, w)

    print(f"Weighted Pearson r (Survey BAL vs Actual-Survey BNP): {r:.6f}")
    print(f"Weighted R^2: {r2:.6f}")
    print(f"Weighted p-value (approx): {p_value:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
