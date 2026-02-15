import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Dhaka-Chittagong vs Total differences.")
    parser.add_argument(
        "--degree",
        type=int,
        default=1,
        help="Polynomial degree for trend line (e.g., 1=linear, 2=quadratic).",
    )
    parser.add_argument(
        "--fit",
        choices=["poly", "power", "linear", "both", "all"],
        default="poly",
        help="Trend line model: poly, power (y = a * x^b), linear, both (poly+power), or all.",
    )
    parser.add_argument("--a", type=float, help="Power-law parameter a (y = a * x^b).")
    parser.add_argument("--b", type=float, help="Power-law parameter b (y = a * x^b).")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for inputs instead of relying only on CLI args.",
    )
    args = parser.parse_args()

    if args.interactive or len(sys.argv) == 1:
        raw_fit = input("কোন ফিট চান? poly / power / linear / both / all: ").strip().lower()
        args.fit = raw_fit if raw_fit in {"poly", "power", "linear", "both", "all"} else "both"

        if args.fit in {"poly", "both", "all"}:
            raw_degree = input("পলিনোমিয়াল ডিগ্রি দিন (1/2/3...): ").strip()
            try:
                args.degree = int(raw_degree)
            except ValueError as exc:
                raise ValueError("degree must be an integer >= 1") from exc

        if args.fit in {"power", "both", "all"}:
            raw_a = input("পাওয়ার ল এর a মান দিন (y = a * x^b): ").strip()
            raw_b = input("পাওয়ার ল এর b মান দিন (y = a * x^b): ").strip()
            try:
                args.a = float(raw_a)
                args.b = float(raw_b)
            except ValueError as exc:
                raise ValueError("a and b must be numbers") from exc

    if args.fit in {"poly", "both", "all"} and args.degree < 1:
        raise ValueError("degree must be >= 1")

    if args.fit in {"power", "both", "all"} and (args.a is None or args.b is None):
        raise ValueError("power-law fit requires a and b values (use --a and --b or interactive input).")

    df = pd.read_csv("Dhakanadchittagong - Sheet1.csv")
    x_col = "Dhaka and Chittagong difference"
    y_col = "Total Difference"

    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()

    plt.style.use("seaborn-v0_8-whitegrid")
    years = df["Election Year"].astype(str)
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.15, 0.85, len(df)))

    order_by_x = np.argsort(x)
    x_sorted = x[order_by_x]
    y_sorted = y[order_by_x]
    years_sorted = years.iloc[order_by_x].to_numpy()

    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0


    def format_poly_equation(coeffs: np.ndarray) -> str:
        degree = len(coeffs) - 1
        terms = []
        for i, coef in enumerate(coeffs):
            power = degree - i
            if power == 0:
                terms.append(f"{coef:.3f}")
            elif power == 1:
                terms.append(f"{coef:.3f}x")
            else:
                terms.append(f"{coef:.3f}x^{power}")
        return "y = " + " + ".join(terms)

    # 1) Scatter + polynomial trendline
    if args.fit in {"poly", "both", "all"}:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x, y, s=120, c=colors, edgecolor="white", linewidth=1.2, zorder=3)

        poly_r2 = None
        if len(df) >= args.degree + 1:
            coeffs = np.polyfit(x, y, args.degree)
            poly = np.poly1d(coeffs)
            y_hat = poly(x)
            poly_r2 = r2_score(y, y_hat)
            x_line = np.linspace(min(x) - 5, max(x) + 5, 200)
            y_line = poly(x_line)
            ax.plot(x_line, y_line, color="#1f2937", linewidth=2, alpha=0.8, zorder=2)
            eqn = format_poly_equation(coeffs)
            print(f"Polynomial equation (degree {args.degree}): {eqn}")
            ax.text(
                0.02,
                0.98,
                eqn,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        for xi, yi, label in zip(x, y, years):
            ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(6, 6), fontsize=9)

        title = "Polynomial Fit: Dhaka-Chittagong vs Total"
        if poly_r2 is not None:
            title = f"{title} (R²={poly_r2:.3f})"
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_xlabel(x_col, fontsize=11)
        ax.set_ylabel(y_col, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig("dhaka_chittagong_vs_total_scatter_poly.png", dpi=200)
        plt.show()

    # 2) Scatter + power-law trendline
    if args.fit in {"power", "both", "all"}:
        if np.any(x <= 0) or np.any(y <= 0):
            raise ValueError("Power-law fit requires all x and y values to be positive.")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x, y, s=120, c=colors, edgecolor="white", linewidth=1.2, zorder=3)

        power_r2 = None
        if len(df) >= 2:
            y_hat = args.a * np.power(x, args.b)
            power_r2 = r2_score(y, y_hat)
            x_line = np.linspace(min(x), max(x), 200)
            y_line = args.a * np.power(x_line, args.b)
            ax.plot(x_line, y_line, color="#1f2937", linewidth=2, alpha=0.8, zorder=2)
            eqn = f"y = {args.a:.3f}x^{args.b:.3f}"
            print(f"Power-law equation: {eqn}")
            ax.text(
                0.02,
                0.98,
                eqn,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        for xi, yi, label in zip(x, y, years):
            ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(6, 6), fontsize=9)

        title = "Power-Law Fit: Dhaka-Chittagong vs Total"
        if power_r2 is not None:
            title = f"{title} (R²={power_r2:.3f})"
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_xlabel(x_col, fontsize=11)
        ax.set_ylabel(y_col, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig("dhaka_chittagong_vs_total_scatter_power.png", dpi=200)
        plt.show()

    # 3) Scatter + linear trendline
    if args.fit in {"linear", "all"}:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x, y, s=120, c=colors, edgecolor="white", linewidth=1.2, zorder=3)

        linear_r2 = None
        if len(df) >= 2:
            coeffs = np.polyfit(x, y, 1)
            poly = np.poly1d(coeffs)
            y_hat = poly(x)
            linear_r2 = r2_score(y, y_hat)
            x_line = np.linspace(min(x) - 5, max(x) + 5, 200)
            y_line = poly(x_line)
            ax.plot(x_line, y_line, color="#1f2937", linewidth=2, alpha=0.8, zorder=2)
            eqn = format_poly_equation(coeffs)
            print(f"Linear equation: {eqn}")
            ax.text(
                0.02,
                0.98,
                eqn,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        for xi, yi, label in zip(x, y, years):
            ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(6, 6), fontsize=9)

        title = "Linear Fit: Dhaka-Chittagong vs Total"
        if linear_r2 is not None:
            title = f"{title} (R²={linear_r2:.3f})"
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_xlabel(x_col, fontsize=11)
        ax.set_ylabel(y_col, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig("dhaka_chittagong_vs_total_scatter_linear.png", dpi=200)
        plt.show()

    # 4) Line plot (both series over years)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(years_sorted, x_sorted, marker="o", linewidth=2, color="#2563eb", label=x_col)
    ax.plot(years_sorted, y_sorted, marker="o", linewidth=2, color="#16a34a", label=y_col)
    ax.set_title("Year-wise Differences", fontsize=14, pad=12)
    ax.set_xlabel("Election Year", fontsize=11)
    ax.set_ylabel("Difference", fontsize=11)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig("dhaka_chittagong_vs_total_lines.png", dpi=200)
    plt.show()



if __name__ == "__main__":
    main()
