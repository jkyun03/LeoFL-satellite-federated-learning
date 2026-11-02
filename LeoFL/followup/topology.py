# LeoFL/followup/topology.py
"""
Constellation / leader / neighbor-graph construction.

Reuses LH-FL's existing orbit and leader-selection utilities UNCHANGED, so
that for the same ``constellation_configs`` this follow-up study selects the
exact same leaders and inter-orbit neighbor graph as the existing LH-FL
experiments (spec section 2: "동일한 ... leader 구성 ... 사용해야 한다").
Only the functions needed to build that static topology are imported;
training/scheduling/aggregation (engine.py, policies.py, fednova.py) are new
code with no further dependency on LeoFL.Simulator.scenarios.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from LeoFL.Simulator.fl_core import (
    build_fixed_constellation,
    compute_orbits_for_rounds,
    reindex_satellites_by_orbital_phase,
    precompute_meetings_and_select_leaders,
)
from LeoFL.Simulator.scenarios import (
    _build_topk_leader_plane_neighbors,
    ring_hop_distance,
)

DEFAULT_LEADER_SELECT_SECONDS_PER_ROUND = 60
# Matches the max_dist_km used for the headline LH-FL run in
# simulation_eurosat.py's run_contact_aware_lhfl(..., max_dist_km=5000.0)
DEFAULT_LEADER_SELECT_MAX_DIST_KM = 5000.0


@dataclass
class Topology:
    sats: list  # LEOSat objects; index == global satellite id
    plane_ids: List[int]  # plane_ids[sat_id] -> orbit id
    plane_to_sats: Dict[int, List[int]]  # orbit id -> member satellite ids, ring order
    leaders_per_plane: Dict[int, int]  # orbit id -> leader satellite id
    leader_neighbors: Dict[
        int, Set[int]
    ]  # orbit id -> neighbor orbit ids (static top-k graph)
    meeting_counts: object  # kept only for logging/debugging

    def local_index(self, plane_id: int, sat_id: int) -> int:
        return self.plane_to_sats[plane_id].index(sat_id)

    def ring_hops_to_leader(self, plane_id: int, sat_id: int) -> int:
        """Intra-orbit ISL ring hop distance from ``sat_id`` to its orbit's
        leader (always reachable, no contact-window check -- same
        always-connected-ring assumption every existing LH-FL scenario
        makes)."""
        members = self.plane_to_sats[plane_id]
        leader = self.leaders_per_plane[plane_id]
        return ring_hop_distance(
            self.local_index(plane_id, sat_id),
            self.local_index(plane_id, leader),
            len(members),
        )

    def leader_edges(self) -> List[Tuple[int, int]]:
        edges = set()
        for p, neigh in self.leader_neighbors.items():
            for q in neigh:
                edges.add((min(p, q), max(p, q)))
        return sorted(edges)

    def leader_sats_by_plane(self) -> Dict[int, object]:
        return {p: self.sats[sid] for p, sid in self.leaders_per_plane.items()}


def build_topology(
    constellation_configs: List[dict],
    horizon_s: float,
    leader_top_k: int = 2,
    leader_select_max_dist_km: float = DEFAULT_LEADER_SELECT_MAX_DIST_KM,
    leader_select_seconds_per_round: int = DEFAULT_LEADER_SELECT_SECONDS_PER_ROUND,
    log_dir: Optional[str] = None,
) -> Topology:
    sats = build_fixed_constellation(constellation_configs)
    plane_ids = [c.get("plane_id", 0) for c in constellation_configs]

    # Leader selection reuses the existing tick-based meeting-count
    # machinery as-is (a one-time setup step); the followup engine's own
    # continuous-time event clock (engine.py) is independent of this.
    num_rounds = int(horizon_s // leader_select_seconds_per_round) + 2
    compute_orbits_for_rounds(
        sats, num_rounds=num_rounds, seconds_per_round=leader_select_seconds_per_round
    )
    sats, plane_ids = reindex_satellites_by_orbital_phase(sats, plane_ids, round_idx=0)

    meeting_counts, leaders_per_plane, plane_to_sats = (
        precompute_meetings_and_select_leaders(
            sats,
            plane_ids=plane_ids,
            num_rounds=num_rounds,
            seconds_per_round=leader_select_seconds_per_round,
            max_dist_km=leader_select_max_dist_km,
            log_dir=log_dir,
            filename="followup_meeting_counts.csv",
        )
    )

    leader_neighbors = _build_topk_leader_plane_neighbors(
        leaders_per_plane=leaders_per_plane,
        meeting_counts=meeting_counts,
        top_k=leader_top_k,
    )

    return Topology(
        sats=sats,
        plane_ids=plane_ids,
        plane_to_sats={int(p): list(v) for p, v in plane_to_sats.items()},
        leaders_per_plane={int(p): int(v) for p, v in leaders_per_plane.items()},
        leader_neighbors={
            int(p): set(int(x) for x in v) for p, v in leader_neighbors.items()
        },
        meeting_counts=meeting_counts,
    )


def _topology_cache_key(
    constellation_configs: List[dict],
    horizon_s: float,
    leader_top_k: int,
    leader_select_max_dist_km: float,
) -> str:
    payload = json.dumps(
        {
            "constellation_configs": constellation_configs,
            "horizon_s": horizon_s,
            "leader_top_k": leader_top_k,
            "leader_select_max_dist_km": leader_select_max_dist_km,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def build_topology_cached(
    constellation_configs: List[dict],
    horizon_s: float,
    cache_dir: str,
    leader_top_k: int = 2,
    leader_select_max_dist_km: float = DEFAULT_LEADER_SELECT_MAX_DIST_KM,
    leader_select_seconds_per_round: int = DEFAULT_LEADER_SELECT_SECONDS_PER_ROUND,
    log_dir: Optional[str] = None,
) -> Topology:
    """Disk-cached ``build_topology``. Leader selection
    (``precompute_meetings_and_select_leaders``) is an O(n^2) Python loop
    over every round of the horizon and dominates a single run's wall time
    at the real 220-satellite scale (measured ~564s of a ~631s total run) --
    but its result depends only on the arguments here, not on ``method`` or
    ``seed``-driven training randomness, so every method config at a given
    scale/seed shares one identical topology. Caching it here turns "N
    methods x topology_build" into "1 x topology_build", the dominant cost
    in a multi-method sweep.

    Cache key = sha256 of the exact (constellation_configs, horizon_s,
    leader_top_k, leader_select_max_dist_km) that determine the result --
    any change to those invalidates the cache automatically (no manual
    cache-busting needed).
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = _topology_cache_key(
        constellation_configs, horizon_s, leader_top_k, leader_select_max_dist_km
    )
    cache_path = os.path.join(cache_dir, f"topology_{key}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    topology = build_topology(
        constellation_configs,
        horizon_s=horizon_s,
        leader_top_k=leader_top_k,
        leader_select_max_dist_km=leader_select_max_dist_km,
        leader_select_seconds_per_round=leader_select_seconds_per_round,
        log_dir=log_dir,
    )
    with open(cache_path, "wb") as f:
        pickle.dump(topology, f)
    return topology
