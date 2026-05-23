"""Dataclass models for notebook orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .status import JobStatus


ParameterType = Literal["str", "int", "float", "bool", "choice", "date"]


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    type: ParameterType
    default: Any = None
    description: str = ""
    required: bool = False
    choices: list[Any] = field(default_factory=list)
    minimum: float | None = None
    maximum: float | None = None

    def validate(self, value: Any) -> Any:
        if value in (None, ""):
            if self.required:
                raise ValueError(f"{self.name} is required")
            return self.default
        if self.type == "int":
            value = int(value)
        elif self.type == "float":
            value = float(value)
        elif self.type == "bool":
            value = bool(value)
        elif self.type in {"str", "date"}:
            value = str(value)
        elif self.type == "choice":
            if value not in self.choices:
                raise ValueError(f"{self.name} must be one of {self.choices}")
        if self.minimum is not None and isinstance(value, (int, float)) and value < self.minimum:
            raise ValueError(f"{self.name} must be >= {self.minimum}")
        if self.maximum is not None and isinstance(value, (int, float)) and value > self.maximum:
            raise ValueError(f"{self.name} must be <= {self.maximum}")
        return value


@dataclass(frozen=True)
class ArtifactSpec:
    label: str
    relative_path: str
    required: bool = False
    kind: str = "file"
    freshness_hours: int | None = None


@dataclass(frozen=True)
class NotebookJob:
    job_id: str
    label: str
    notebook_path: Path | None
    runner_type: str
    output_notebook_template: str
    timeout_seconds: int
    parameters: list[ParameterSpec] = field(default_factory=list)
    expected_artifacts: list[ArtifactSpec] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    enabled: bool = True
    disabled_reason: str = ""
    output_domain: str = "workspace"
    description: str = ""

    def validate_parameters(self, values: dict[str, Any]) -> dict[str, Any]:
        return {spec.name: spec.validate(values.get(spec.name, spec.default)) for spec in self.parameters}


@dataclass
class JobRun:
    run_id: str
    job_id: str
    status: JobStatus
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    runner_type: str = ""
    notebook_path: str = ""
    output_notebook_path: str = ""
    artifacts_detected: list[dict[str, Any]] = field(default_factory=list)
    log_path: str = ""
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobRun":
        payload = dict(data)
        payload["status"] = JobStatus(str(payload.get("status", JobStatus.PENDING)).replace("JobStatus.", ""))
        return cls(**payload)
