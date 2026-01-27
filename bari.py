import pandas as pd
import matplotlib.pyplot as plt


def render_table(title: str, df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        edges="closed",
    )
    header_color = "#1f2937"
    row_colors = ["#f8fafc", "#eef2f7"]
    text_color = "#111827"
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor(row_colors[(row - 1) % 2])
            cell.set_text_props(color=text_color)
    ax.set_title(title, pad=12, fontweight="bold")
    fig.set_facecolor("white")
    plt.tight_layout()
    plt.show()


def main() -> None:
    df = pd.read_csv("Barisal Division - Sheet1.csv")
    display_df = df.copy()
    for col in display_df.columns[1:]:
        display_df[col] = pd.to_numeric(display_df[col], errors="coerce").map(
            lambda v: f"{v:.2f}%" if pd.notna(v) else ""
        )
    render_table("Barisal Division survey data", display_df)


if __name__ == "__main__":
    main()
