# src/utils.py


from pathlib import Path
import json
import matplotlib.pyplot as plt

from src.logger import get_logger

logger = get_logger(__name__)


def save_json(data, path):
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                indent=4
            )

        logger.info("JSON saved successfully: %s", path)

    except Exception:
        logger.exception("Failed to save JSON: %s", path)
        raise
def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        logger.info("JSON loaded successfully: %s", path)

        return data

    except Exception:
        logger.exception("Failed to load JSON: %s", path)
        raise



def artifact_exists(path: str | Path) -> bool:
    """
    Check if an artifact already exists.

    Returns True if it exists, otherwise False.
    """
    path = Path(path)

    if path.exists():
        logger.info(
            "Artifact already exists, skipping: %s",
            path
        )
        return True

    return False


def save_json_if_missing(data: dict | tuple, path: str | Path) -> None:
    """
    Save JSON only if the file does not already exist.
    """
    path = Path(path)

    if artifact_exists(path):
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    try:
        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                data,
                file,
                indent=4
            )

        logger.info(
            "JSON saved successfully: %s",
            path
        )

    except Exception:
        logger.exception(
            "Failed to save JSON: %s",
            path
        )
        raise


def save_figure_if_missing(
    fig,
    path: str | Path,
    dpi: int = 200
) -> None:
    """
    Save a matplotlib figure only if it does not already exist.
    """
    path = Path(path)

    if artifact_exists(path):
        plt.close(fig)
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    try:
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight"
        )

        logger.info(
            "Figure saved successfully: %s",
            path
        )

    except Exception:
        logger.exception(
            "Failed to save figure: %s",
            path
        )
        raise

    finally:
        plt.close(fig)