"""Shared data models used by forensic modules."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ForensicResult(BaseModel):
    """Standard result returned by an individual forensic module."""

    module: str = Field(description="Name of the module that produced the result.")
    status: Literal["success", "error"] = Field(description="Module execution status.")
    model: str | None = Field(
        default=None,
        description="Optional model or algorithm used by the module.",
    )
    score: float | None = Field(
        default=None,
        description="Optional module score, typically normalized by that module.",
    )
    label: str | None = Field(
        default=None,
        description="Optional human-readable classification label.",
    )
    confidence: str | None = Field(
        default=None,
        description="Optional module-specific confidence description.",
    )
    evidence: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured observations supporting the result.",
    )
    artifacts: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Paths or identifiers for generated forensic artifacts.",
    )
    error: str | None = Field(
        default=None,
        description="Recoverable module error, present only when status is 'error'.",
    )
