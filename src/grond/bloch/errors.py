"""Explicit physical-feasibility errors carried across worker processes."""

from __future__ import annotations


class LocalSgCrossingError(ValueError):
    """A requested target Sg has no crossing inside the local geometry bound."""


class InfeasibleSgJitterProfileError(ValueError):
    """An active Sg-jitter candidate requests an unreachable quadrature point."""


class NonPassiveAbsorptionError(ValueError):
    """An optical-potential candidate would amplify some incident state.

    ``details`` is JSON-compatible and survives multiprocessing exception
    transfer, so an optional grid candidate can be rejected with evidence.
    """

    def __init__(self, message, *, details=None):
        """Keep the message and a plain-dict copy of ``details`` (empty when None)."""

        super().__init__(message)
        self.details = dict(details or {})


class MissingObservedCentroidError(ValueError):
    """A requested centroid-based profile has no measured spot position."""
