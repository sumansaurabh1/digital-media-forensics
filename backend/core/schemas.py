"""Shared data models used by forensic modules."""

from typing import Any

from pydantic import BaseModel, Field


class ForensicResult(BaseModel):
    """Standard result returned by an individual forensic module."""

    module: str = Field(description="Name of the module that produced the result.")
    status: str = Field(description="Module execution status, such as 'completed'.")
    score: float | None = Field(
        default=None,
        description="Optional module score, typically normalized by that module.",
    )
    label: str | None = Field(
        default=None,
        description="Optional human-readable classification label.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Optional confidence from 0 to 1.",
    )
    evidence: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured observations supporting the result.",
    )
    artifacts: list[str] = Field(
        default_factory=list,
        description="Paths or identifiers for generated forensic artifacts.",
    )
