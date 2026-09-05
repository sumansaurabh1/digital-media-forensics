"""Minimal orchestration layer for forensic investigation modules."""

from collections.abc import Callable

from backend.core.schemas import ForensicResult


ForensicModule = Callable[[str], ForensicResult]


class InvestigationPipeline:
    """Runs registered forensic modules against a supplied media reference.

    Modules will be added later for AI-generation detection, metadata,
    manipulation analysis, fingerprinting, and traceability.
    """

    def __init__(self) -> None:
        self._modules: list[ForensicModule] = []

    def register(self, module: ForensicModule) -> None:
        """Add a forensic module to the investigation sequence."""
        self._modules.append(module)

    def run(self, media_reference: str) -> list[ForensicResult]:
        """Run registered modules and return their results in registration order."""
        return [module(media_reference) for module in self._modules]
