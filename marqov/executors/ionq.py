"""IonQ executor for running circuits on IonQ devices.

This module provides IonQExecutor for executing quantum circuits on IonQ devices.

Example:
    >>> from marqov.circuits import bell_state
    >>> from marqov.executors import IonQExecutor, IonQExecutorConfig
    >>>
    >>> config = IonQExecutorConfig(
    ...     backend="simulator",
    ...     api_key="your-api-key",
    ...     project_id="your-project-id",
    ...     name="bell-state",
    ... )
    >>> executor = IonQExecutor(config)
    >>> result = await executor.execute(bell_state(), shots=1000)
    >>> print(result.counts)  # {"00": ~500, "11": ~500}
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any
from marqov.executors.base import BaseExecutor, ExecutionResult

if TYPE_CHECKING:
    from marqov.circuits import Circuit

@dataclass
class IonQExecutorConfig:
    """Configuration for IonQ executor.

    Attributes:
        backend: IonQ backend name (e.g., "simulator").
        api_key: IonQ API key.
        project_id: IonQ project ID.
        job_name: Name of the job.
        poll_interval_seconds: Polling interval for job completion.
        timeout_seconds: Maximum time to wait for job completion. None for no timeout.
    """

    backend: str
    api_key: str | None = None
    project_id: str | None = None
    job_name: str | None = None
    poll_interval_seconds: float = 2.0
    timeout_seconds: float | None = None
    api_version: str = "v0.4"
    dry_run: bool = False

class IonQExecutor(BaseExecutor):
    """Executor for running circuits on IonQ devices.

    Supports running circuits on IonQ devices using the IonQ Direct API.

    Example:
        >>> config = IonQExecutorConfig(
        ...     backend="simulator",
        ...     api_key="your-api-key",
        ...     project_id="your-project-id",
        ...     name="job_name",
        ... )
        >>> executor = IonQExecutor(config)
        >>> result = await executor.execute(bell_state(), shots=1000)
        >>> print(result.counts)  # {"00": ~500, "11": ~500}
    """

    def __init__(self, config: IonQExecutorConfig) -> None:
        """Initialize IonQExecutor.

        Args:
            config: Executor configuration including backend details.
        """
        self.config = config
        self._current_job_id: str | None = None

    async def execute(self, circuit: Circuit, shots: int = 1000, **kwargs: Any) -> ExecutionResult:
            """ Execute a circuit on the IonQ device.

            Args:
                circuit: The circuit to execute.
                shots: Number of measurement shots.
                **kwargs: Additional options.

            Returns:
                ExecutionResult with measurement counts and metadata.
            """
            circuit = self._validate_circuit(circuit)
            loop = asyncio.get_running_loop()
            start_time = time.perf_counter()
            if self.config.timeout_seconds is not None:
                counts, job = await asyncio.wait_for(
                    loop.run_in_executor(None, partial(self._run_sync, circuit, shots, **kwargs)),
                    timeout=self.config.timeout_seconds,
                )
            else:
               counts, job = await loop.run_in_executor(None, partial(self._run_sync, circuit, shots, **kwargs))
            
            wall_time = time.perf_counter() - start_time
            self._current_job_id = job["id"]
        
            return ExecutionResult(
                counts=counts,
                backend=self.config.backend,
                execution_time_ms= job.get("execution_duration_ms") or wall_time * 1000,
                shots=shots,
                raw_result=job,
                metadata={
                    "job_id": job["id"],
                    "status": job["status"],
                    "wall_time_ms": wall_time * 1000,
                },
            )
    
    def _build_job_payload(self, circuit: Circuit, shots: int) -> dict:
        qasm = circuit.to_openqasm()   # needs marqov[openqasm]
        payload = {
            "backend": self.config.backend,   # "simulator", "qpu.forte-1", etc.
            "shots": shots,
            "name": self.config.job_name,
            "dry_run": self.config.dry_run,
            "input": {
                "format": "openqasm",
                "data": qasm,
            },
        }
        if self.config.project_id:
            payload["project_id"] = self.config.project_id
        return payload

    def _run_sync(self, circuit: Circuit, shots: int, **kwargs: Any) -> tuple[dict, dict]:
        payload = self._build_job_payload(circuit, shots)
        created = self._api_post("/jobs", payload)
        job_id = created["id"]

        while True:
            job = self._api_get(f"/jobs/{job_id}")
            if job["status"] == "completed":
                counts = self._fetch_counts(job, shots)
                return counts, job
            if job["status"] in ("failed", "canceled"):
                raise RuntimeError(job.get("failure", job["status"]))
            time.sleep(self.config.poll_interval_seconds)

    def _api_post(self, path: str, payload: dict) -> dict:
        import requests
        headers = {
            "Authorization": f"apiKey {self._get_api_key()}",
            "Content-Type": "application/json",
        }
        response = requests.post(self._resolve_url(path), headers=headers, json=payload)
        response.raise_for_status()
        return response.json()

    def _api_get(self, path: str) -> dict:
        import requests
        headers = {
            "Authorization": f"apiKey {self._get_api_key()}",
            "Content-Type": "application/json",
        }
        response = requests.get(self._resolve_url(path), headers=headers)
        response.raise_for_status()
        return response.json()

    def _fetch_counts(self, job: dict, shots: int) -> dict:
        url = job["results"]["probabilities"]["url"]
        # url may be relative — prepend https://api.ionq.co if needed
        probs = self._api_get(url)  # or full URL GET
        return {k: round(v * shots) for k, v in probs.items()}

    def _get_api_key(self) -> str:
        import os
        key = self.config.api_key or os.environ.get("IONQ_API_KEY")
        if not key:
            raise ValueError("IonQ API key required: set api_key or IONQ_API_KEY")
        return key

    def _api_base_url(self) -> str:
        return f"https://api.ionq.co/{self.config.api_version}"

    def _resolve_url(self, path: str) -> str:
        if path.startswith("http"):
            return path
        if path.startswith("/v0."):
            return f"https://api.ionq.co{path}"
        return self._api_base_url() + path