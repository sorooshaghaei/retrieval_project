"""CLI entrypoint for the retrieval project pipeline."""

from src.pipeline import run_pipeline


if __name__ == "__main__":
    # Keep main.py minimal: all logic lives in src/pipeline.py.
    run_pipeline()
