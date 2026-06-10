"""Integration tests against the real IonQ Direct API simulator.

Skipped unless IONQ_API_KEY is set. Run locally before opening a PR:

    $env:IONQ_API_KEY = "your-key"
    pytest tests/integration/test_ionq_executor.py -v
"""

from __future__ import annotations

import os

import pytest

from marqov.circuits import bell_state
from marqov.executors.ionq import IonQExecutor, IonQExecutorConfig

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("IONQ_API_KEY"),
        reason="Requires IONQ_API_KEY",
    ),
]


@pytest.mark.asyncio
async def test_execute_bell_state_on_simulator() -> None:
    """Run bell_state on IonQ Direct cloud simulator."""
    config = IonQExecutorConfig(
        backend="simulator",
        api_key=os.environ["IONQ_API_KEY"],
        project_id=os.environ.get("IONQ_PROJECT_ID"),
        job_name="marqov-sdk-integration-test",
    )
    executor = IonQExecutor(config)
    result = await executor.execute(bell_state(), shots=100)

    assert result.shots == 100
    assert result.backend == "simulator"
    assert result.counts
    assert sum(result.counts.values()) > 0
    assert "00" in result.counts or "11" in result.counts
    assert result.metadata.get("job_id")
    assert result.metadata.get("status") == "completed"
