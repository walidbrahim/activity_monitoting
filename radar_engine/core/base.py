"""
radar_engine.core.base
======================
RadarModule — the common abstract interface for all pipeline stages.

Design rationale (refactor.md F6):
  A unified interface makes it trivial to:
  - assemble different processing pipelines at runtime
    (e.g. ActivityOnlyEngine, BedMonitoringEngine, OfflineReplayEngine),
  - swap or skip individual modules for testing,
  - add new processing stages without changing the orchestrator.

Contract:
  Every concrete RadarModule must implement:
    ``reset() -> None``
        Clear all internal state so the module behaves as if freshly
        constructed. Called on track loss, radar pose update, or pipeline reset.

    ``process(context: RadarContext) -> RadarContext``
        Read the fields this module depends on from ``context``, compute
        outputs, write results back into ``context``, and return it.
        Modules must NOT raise exceptions for expected edge-cases (empty
        frame, off-zone, warmup) — they should leave their output fields as
        None / inactive and set a diagnostics entry instead.

Optional override:
    ``name -> str``
        Human-readable module identifier used in diagnostics and logging.
        Defaults to the class name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from radar_engine.core.context import RadarContext


class RadarModule(ABC):
    """Abstract base class for all radar processing pipeline stages.

    Subclasses own one well-defined processing responsibility and maintain
    only the persistent state they need for that responsibility.

    See module docstring for the full contract.
    """

    # ── Interface ────────────────────────────────────────────────────────────

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal state.

        After reset(), the module must produce the same outputs as a freshly
        constructed instance on the next process() call.
        """

    @abstractmethod
    def process(self, context: RadarContext) -> RadarContext:
        """Run this stage for one radar frame.

        Args:
            context: Shared pipeline state.  Read upstream fields; write
                     this module's output fields.

        Returns:
            The same ``context`` object with output fields populated.

        Raises:
            Nothing for expected edge-cases.  Unexpected errors (e.g. malformed
            frame shape) may raise ValueError or RuntimeError.
        """

    # ── Optional overrides ───────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Human-readable module identifier for diagnostics."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"<{self.name}>"


class NullModule(RadarModule):
    """A no-op module that passes the context through unchanged.

    Useful as a placeholder when a pipeline stage is intentionally disabled,
    e.g. when running an activity-only pipeline without respiration analysis.
    """

    def __init__(self, label: str = "NullModule"):
        self._label = label

    @property
    def name(self) -> str:
        return self._label

    def reset(self) -> None:
        pass  # nothing to clear

    def process(self, context: RadarContext) -> RadarContext:
        return context  # pass-through
