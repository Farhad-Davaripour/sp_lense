"""Publish recorded pilot results to an existing Prefect server; no server provisioning."""

import argparse
import json
from pathlib import Path


def publish(output):
    from prefect import flow
    from prefect.artifacts import (
        create_markdown_artifact,
        create_progress_artifact,
        create_table_artifact,
    )

    @flow(name="research-2-feasibility", retries=0, persist_result=False)
    def snapshot():
        output_path = Path(output)
        state = json.loads((output_path / "STATUS.json").read_text())
        create_progress_artifact(
            key="research-2-progress", progress=state["percent"], description=json.dumps(state)
        )
        if (output_path / "comparison.json").exists():
            create_table_artifact(
                key="research-2-comparison",
                table=json.loads((output_path / "comparison.json").read_text()),
            )
        summary = output_path / "RESULT.md"
        create_markdown_artifact(
            key="research-2-status",
            markdown=summary.read_text() if summary.exists() else json.dumps(state),
        )

    snapshot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    publish(parser.parse_args().output)
