"""Small, dependency-free reference for graph-stabilized DG recruitment.

The production learner lives in the NEMO2 ``SF_hipposlam`` checkout.  This
module deliberately contains only the state machine that must be shared by
that learner and its tests, so its boundary conditions can be tested in this
vault checkout as well.

The graph is updated from exclusive DG events observed by the actor.  CA3 is
not an input to any method here.  A caller using the policy-buffer HRL path
passes the policy graph matrices to :meth:`eligible_victim`; flat agents pass
the passive matrices instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pow
from typing import List, Optional, Sequence, Tuple


def _square(n: int, value: float = 0.0) -> List[List[float]]:
    return [[value for _ in range(n)] for _ in range(n)]


@dataclass
class GraphRecruitmentState:
    """Checkpointable passive graph and actor-side event history.

    ``birth_support`` is initialized to one, and is decayed on each accepted
    transition.  A new row therefore cannot be immediately recycled.  Matrix
    confidence is decayed before each accepted transition and elapsed steps
    are maintained as a confidence-weighted mean.
    """

    n_units: int
    L: int
    connectivity_threshold: float = 0.25
    redundancy_max_steps: int = 4
    half_life_events: float = 5000.0
    confidence: List[List[float]] = field(init=False)
    elapsed: List[List[float]] = field(init=False)
    birth_support: List[float] = field(init=False)
    generation: List[int] = field(init=False)
    last_exclusive_id: Optional[int] = field(default=None, init=False)
    decisions_since_last_active: int = field(default=0, init=False)
    last_generation: Optional[int] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.n_units <= 0:
            raise ValueError("n_units must be positive")
        if self.L <= 0:
            raise ValueError("L must be positive")
        if not 0.0 < self.connectivity_threshold < 1.0:
            raise ValueError("connectivity_threshold must be in (0, 1)")
        if self.redundancy_max_steps <= 0:
            raise ValueError("redundancy_max_steps must be positive")
        if self.half_life_events <= 0:
            raise ValueError("half_life_events must be positive")
        self.confidence = _square(self.n_units)
        self.elapsed = _square(self.n_units)
        self.birth_support = [1.0] * self.n_units
        self.generation = [0] * self.n_units

    @property
    def decay_factor(self) -> float:
        return pow(0.5, 1.0 / self.half_life_events)

    def reset_episode(self) -> None:
        """Drop only the actor history at a physical-episode boundary."""

        self.last_exclusive_id = None
        self.decisions_since_last_active = 0
        self.last_generation = None

    def _decay_after_accepted_transition(self) -> None:
        gamma = self.decay_factor
        for i in range(self.n_units):
            self.birth_support[i] *= gamma
            for j in range(self.n_units):
                self.confidence[i][j] *= gamma

    def record_transition(self, source: int, destination: int, gap: int,
                          source_generation: int, destination_generation: int
                          ) -> bool:
        """Apply one learner-side rollout transition if its generations match.

        This is the stale-evidence guard used when a rollout was collected
        before a structural reassignment.  It intentionally does not mutate
        actor history; callers use :meth:`observe` for that state.
        """

        if not (0 <= source < self.n_units and 0 <= destination < self.n_units):
            raise ValueError("transition endpoint out of range")
        if not 1 <= gap <= self.L or source == destination:
            return False
        if (source_generation != self.generation[source]
                or destination_generation != self.generation[destination]):
            return False
        self._decay_after_accepted_transition()
        old_confidence = self.confidence[source][destination]
        new_confidence = old_confidence + 1.0
        self.confidence[source][destination] = new_confidence
        self.elapsed[source][destination] = (
            (old_confidence * self.elapsed[source][destination] + gap)
            / new_confidence
        )
        return True

    def observe(self, exclusive_id: Optional[int], generation: Optional[int], *,
                episode_start: bool = False) -> bool:
        """Observe one behavior-time decision.

        Returns whether this observation accepted a passive transition.  A
        transition is recorded only for different exclusive units in the same
        episode with a gap no larger than ``L``.  Generation mismatches are
        stale rollout evidence and are rejected.  ``exclusive_id=None`` is a
        silent/non-exclusive decision and advances the history age.
        """

        if episode_start:
            self.reset_episode()

        if exclusive_id is None:
            if self.last_exclusive_id is not None:
                self.decisions_since_last_active += 1
            return False
        if not 0 <= exclusive_id < self.n_units:
            raise ValueError("exclusive_id out of range")
        if generation is None:
            raise ValueError("generation is required for an active DG event")

        accepted = False
        if self.last_exclusive_id is not None:
            gap = self.decisions_since_last_active + 1
            same_generation = self.last_generation == self.generation[self.last_exclusive_id]
            current_generation = generation == self.generation[exclusive_id]
            if (same_generation and current_generation
                    and self.last_exclusive_id != exclusive_id
                    and gap <= self.L):
                source = self.last_exclusive_id
                accepted = self.record_transition(
                    source,
                    exclusive_id,
                    gap,
                    self.last_generation,
                    generation,
                )

        self.last_exclusive_id = exclusive_id
        self.decisions_since_last_active = 0
        self.last_generation = generation
        return accepted

    def adjacency(self, confidence: Optional[Sequence[Sequence[float]]] = None,
                  elapsed: Optional[Sequence[Sequence[float]]] = None
                  ) -> List[List[bool]]:
        confidence = self.confidence if confidence is None else confidence
        elapsed = self.elapsed if elapsed is None else elapsed
        return [[
            i != j and confidence[i][j] > self.connectivity_threshold
            and elapsed[i][j] > 0.0
            for j in range(self.n_units)
        ] for i in range(self.n_units)]

    def incident_support(self, confidence: Optional[Sequence[Sequence[float]]] = None
                         ) -> List[float]:
        confidence = self.confidence if confidence is None else confidence
        return [
            sum(confidence[i][j] for j in range(self.n_units) if j != i)
            + sum(confidence[j][i] for j in range(self.n_units) if j != i)
            for i in range(self.n_units)
        ]

    def redundant_losers(self, confidence: Optional[Sequence[Sequence[float]]] = None,
                         elapsed: Optional[Sequence[Sequence[float]]] = None
                         ) -> set[int]:
        confidence = self.confidence if confidence is None else confidence
        elapsed = self.elapsed if elapsed is None else elapsed
        adjacent = self.adjacency(confidence, elapsed)
        support = self.incident_support(confidence)
        losers: set[int] = set()
        for i in range(self.n_units):
            for j in range(i + 1, self.n_units):
                mutual_close = (
                    adjacent[i][j] and adjacent[j][i]
                    and elapsed[i][j] <= self.redundancy_max_steps
                    and elapsed[j][i] <= self.redundancy_max_steps
                )
                if not mutual_close:
                    continue
                if support[i] < support[j]:
                    losers.add(i)
                elif support[j] < support[i]:
                    losers.add(j)
                else:
                    # The higher index is the deterministic loser.
                    losers.add(max(i, j))
        return losers

    def eligible_victim(self, *, confidence: Optional[Sequence[Sequence[float]]] = None,
                        elapsed: Optional[Sequence[Sequence[float]]] = None
                        ) -> Tuple[Optional[int], str]:
        """Return ``(row, reason)`` using the graph eligibility rule."""

        confidence = self.confidence if confidence is None else confidence
        elapsed = self.elapsed if elapsed is None else elapsed
        adjacent = self.adjacency(confidence, elapsed)
        support = self.incident_support(confidence)
        expired = {
            i for i, value in enumerate(self.birth_support)
            if value <= self.connectivity_threshold
        }
        isolated = {
            i for i in expired
            if not any(adjacent[i]) and not any(row[i] for row in adjacent)
        }
        if isolated:
            return min(isolated, key=lambda i: (support[i], i)), "isolated"
        redundant = self.redundant_losers(confidence, elapsed) & expired
        if redundant:
            return min(redundant, key=lambda i: (support[i], i)), "redundant"
        return None, "protected_or_supported"

    def invalidate_row(self, row: int) -> None:
        """Invalidate incident evidence after a structural reassignment."""

        if not 0 <= row < self.n_units:
            raise ValueError("row out of range")
        for i in range(self.n_units):
            self.confidence[row][i] = 0.0
            self.confidence[i][row] = 0.0
            self.elapsed[row][i] = 0.0
            self.elapsed[i][row] = 0.0
        self.birth_support[row] = 1.0
        self.generation[row] += 1
        if self.last_exclusive_id == row:
            self.reset_episode()

    def telemetry(self, *, confidence: Optional[Sequence[Sequence[float]]] = None,
                  elapsed: Optional[Sequence[Sequence[float]]] = None
                  ) -> dict:
        adjacent = self.adjacency(confidence, elapsed)
        isolated = [
            not any(adjacent[i]) and not any(row[i] for row in adjacent)
            for i in range(self.n_units)
        ]
        eligible, reason = self.eligible_victim(confidence=confidence, elapsed=elapsed)
        pair_count = len(self.redundant_losers(confidence, elapsed))
        return {
            "connected_fraction": 1.0 - sum(isolated) / self.n_units,
            "isolated_fraction": sum(isolated) / self.n_units,
            "redundant_loser_count": pair_count,
            "eligible_vertex": eligible,
            "eligible_reason": reason if eligible is not None else None,
            "birth_protected_count": sum(
                value > self.connectivity_threshold for value in self.birth_support
            ),
            "passive_edge_density": (
                sum(sum(row) for row in adjacent) / (self.n_units * (self.n_units - 1))
                if self.n_units > 1 else 0.0
            ),
        }


def select_graph_evidence(*, policy_graph_available: bool,
                          policy_confidence: Optional[Sequence[Sequence[float]]],
                          policy_elapsed: Optional[Sequence[Sequence[float]]],
                          passive_confidence: Sequence[Sequence[float]],
                          passive_elapsed: Sequence[Sequence[float]]) -> Tuple[
                              Sequence[Sequence[float]], Sequence[Sequence[float]]]:
    """Select Tctrl evidence for policy-buffer HRL, passive evidence otherwise."""

    if policy_graph_available:
        if policy_confidence is None or policy_elapsed is None:
            raise ValueError("policy graph was declared available without matrices")
        return policy_confidence, policy_elapsed
    return passive_confidence, passive_elapsed
