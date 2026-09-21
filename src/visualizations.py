import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def save_figure(fig, path):
    path = Path(path)

    # Skapa mappen om den inte finns
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    

    plt.close(fig)

def create_correlation_heatmap_and_save_heatmap(df,path):
    # Bara numeriska kolumner
    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(12, 10))

    heatmap = ax.imshow(
        corr,
        vmin=-1,
        vmax=1,
        cmap="coolwarm"
    )

    # Feature names
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))

    ax.set_xticklabels(
        corr.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(corr.columns)

    # Correlation values inside cells
    for row in range(len(corr.columns)):
        for col in range(len(corr.columns)):
            value = corr.iloc[row, col]

            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center"
            )

    fig.colorbar(
        heatmap,
        ax=ax,
        label="Correlation"
    )

    ax.set_title("Feature Correlation Heatmap")

    fig.tight_layout()

    save_figure(fig,path)