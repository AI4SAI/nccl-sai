#!/usr/bin/env python3
"""Verify NCCL-SAI AlltoAll schedule and island-bulk invariants offline."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable


P2P_BATCH_ROUNDS = 8


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def pow2_up(value: int) -> int:
    result = 1
    while result < value:
        result <<= 1
    return result


def quadratic_permutation(limit: int) -> list[int]:
    modulus = pow2_up(limit)
    round_id = 0
    delta = 0
    result: list[int] = []
    while True:
        if delta < limit:
            result.append(delta)
        round_id += 1
        delta = (delta + round_id) & (modulus - 1)
        if round_id == modulus:
            break
    require(sorted(result) == list(range(limit)), f"invalid quadratic permutation for {limit}")
    return result


@dataclass(frozen=True)
class ScheduleLayout:
    topology_nodes: int
    local_ranks: int = 4
    lane_group_nodes: int = 4
    fabric_nodes: int = 16
    lanes: int = 8
    p2p_group_size: int = 4

    @property
    def ranks(self) -> int:
        return self.topology_nodes * self.local_ranks

    @property
    def fabric_groups(self) -> int:
        return self.topology_nodes // self.fabric_nodes

    def validate_lane(self) -> None:
        require(self.topology_nodes > 0, "topology_nodes must be positive")
        require(self.local_ranks > 0, "local_ranks must be positive")
        require(self.lane_group_nodes > 0, "lane_group_nodes must be positive")
        require(self.topology_nodes % self.lane_group_nodes == 0, "incomplete lane group")
        require(self.lanes >= 2, "lanes must be at least two")

    def validate_fabric_schedule(self) -> None:
        require(self.topology_nodes % self.fabric_nodes == 0, "incomplete fabric group")
        require(self.local_ranks % self.p2p_group_size == 0, "incomplete P2P rank group")


@dataclass(frozen=True)
class RoundDescriptor:
    group_delta: int
    local_delta: int


def fabric_rounds(layout: ScheduleLayout) -> list[RoundDescriptor]:
    layout.validate_fabric_schedule()
    groups_per_node = layout.local_ranks // layout.p2p_group_size
    rank_groups = layout.ranks // layout.p2p_group_size
    groups_per_fabric = layout.fabric_nodes * groups_per_node
    fabric_groups = rank_groups // groups_per_fabric
    descriptors: list[RoundDescriptor] = []
    for local_delta in range(layout.p2p_group_size):
        for group_skew in range(groups_per_fabric):
            for fabric_delta in quadratic_permutation(fabric_groups):
                descriptors.append(
                    RoundDescriptor(
                        group_delta=fabric_delta * groups_per_fabric + group_skew,
                        local_delta=local_delta,
                    )
                )
    require(len(descriptors) == layout.ranks, "fabric schedule length does not match rank count")
    return descriptors


def p2p_peers(layout: ScheduleLayout, rank: int, descriptor: RoundDescriptor) -> tuple[int, int]:
    group_size = layout.p2p_group_size
    groups_per_node = layout.local_ranks // group_size
    rank_groups = layout.ranks // group_size
    node, local_rank = divmod(rank, layout.local_ranks)
    local = local_rank % group_size
    group = node * groups_per_node + local_rank // group_size

    send_group = (group + descriptor.group_delta) % rank_groups
    recv_group = (group - descriptor.group_delta) % rank_groups
    send_node, send_node_group = divmod(send_group, groups_per_node)
    recv_node, recv_node_group = divmod(recv_group, groups_per_node)
    send_local = send_node_group * group_size + (local + descriptor.local_delta) % group_size
    recv_local = recv_node_group * group_size + (local - descriptor.local_delta) % group_size
    return (
        send_node * layout.local_ranks + send_local,
        recv_node * layout.local_ranks + recv_local,
    )


def fabric_group(layout: ScheduleLayout, rank: int) -> int:
    node = rank // layout.local_ranks
    return node // layout.fabric_nodes


def verify_fabric_schedule(layout: ScheduleLayout) -> dict[str, object]:
    descriptors = fabric_rounds(layout)
    rank_count = layout.ranks
    for rank in range(rank_count):
        send_peers: set[int] = set()
        recv_peers: set[int] = set()
        for round_id, descriptor in enumerate(descriptors):
            send_peer, recv_peer = p2p_peers(layout, rank, descriptor)
            send_peers.add(send_peer)
            recv_peers.add(recv_peer)
            peer_recv = p2p_peers(layout, send_peer, descriptor)[1]
            peer_send = p2p_peers(layout, recv_peer, descriptor)[0]
            require(peer_recv == rank, f"send/recv mismatch at rank {rank}, round {round_id}")
            require(peer_send == rank, f"recv/send mismatch at rank {rank}, round {round_id}")
        require(len(send_peers) == rank_count, f"rank {rank} has duplicate send peers")
        require(len(recv_peers) == rank_count, f"rank {rank} has duplicate recv peers")

    cross_edges = layout.fabric_groups * (layout.fabric_groups - 1)
    cumulative_edges: set[tuple[int, int]] = set()
    first_full_undirected_batch = 0
    batch_load_spread: list[dict[str, int]] = []
    total_load: Counter[tuple[int, int]] = Counter()
    for batch_start in range(0, rank_count, P2P_BATCH_ROUNDS):
        batch_load: Counter[tuple[int, int]] = Counter()
        for rank in range(rank_count):
            source_group = fabric_group(layout, rank)
            for descriptor in descriptors[batch_start : batch_start + P2P_BATCH_ROUNDS]:
                send_peer, _ = p2p_peers(layout, rank, descriptor)
                destination_group = fabric_group(layout, send_peer)
                if source_group == destination_group:
                    continue
                edge = (source_group, destination_group)
                batch_load[edge] += 1
                total_load[edge] += 1
                cumulative_edges.add(tuple(sorted(edge)))
        if batch_load:
            values = list(batch_load.values())
            batch_load_spread.append(
                {
                    "batch": batch_start // P2P_BATCH_ROUNDS,
                    "active_directed_edges": len(batch_load),
                    "minimum_messages": min(values),
                    "maximum_messages": max(values),
                }
            )
        if (
            first_full_undirected_batch == 0
            and len(cumulative_edges) == layout.fabric_groups * (layout.fabric_groups - 1) // 2
        ):
            first_full_undirected_batch = batch_start // P2P_BATCH_ROUNDS + 1

    require(len(total_load) == cross_edges, "fabric schedule did not cover every directed fabric edge")
    if total_load:
        require(len(set(total_load.values())) == 1, "total fabric edge load is not balanced")
    return {
        "kind": "p2p_fabric_schedule",
        "layout": asdict(layout),
        "ranks": rank_count,
        "rounds": len(descriptors),
        "directed_edge_count": len(total_load),
        "messages_per_directed_edge": next(iter(total_load.values()), 0),
        "batches_to_full_undirected_coverage": first_full_undirected_batch,
        "first_four_batch_loads": batch_load_spread[:4],
    }


def lane_group_local_rank(layout: ScheduleLayout, rank: int) -> int:
    node, local_rank = divmod(rank, layout.local_ranks)
    return (node % layout.lane_group_nodes) * layout.local_ranks + local_rank


def lane_phase(layout: ScheduleLayout, rank: int, peer: int) -> int:
    return (lane_group_local_rank(layout, rank) + lane_group_local_rank(layout, peer)) % layout.lanes


def verify_lane_schedule(layout: ScheduleLayout) -> dict[str, object]:
    layout.validate_lane()
    rank_count = layout.ranks
    rank_phase_loads: list[int] = []
    edge_phase_load: Counter[tuple[int, int, int]] = Counter()
    for rank in range(rank_count):
        phase_load = [0] * layout.lanes
        for peer in range(rank_count):
            if peer == rank:
                continue
            phase = lane_phase(layout, rank, peer)
            require(phase == lane_phase(layout, peer, rank), f"lane pair mismatch for {rank}<->{peer}")
            phase_load[phase] += 1
            if layout.topology_nodes >= layout.fabric_nodes:
                source_group = fabric_group(layout, rank)
                destination_group = fabric_group(layout, peer)
                if source_group != destination_group:
                    edge_phase_load[(source_group, destination_group, phase)] += 1
        require(max(phase_load) - min(phase_load) <= 1, f"rank {rank} lane load is imbalanced")
        rank_phase_loads.extend(phase_load)

    if edge_phase_load:
        require(len(set(edge_phase_load.values())) == 1, "lane fabric edge load is not phase-balanced")
    return {
        "kind": "a2a_lane_schedule",
        "layout": asdict(layout),
        "ranks": rank_count,
        "minimum_rank_phase_peers": min(rank_phase_loads),
        "maximum_rank_phase_peers": max(rank_phase_loads),
        "messages_per_directed_edge_phase": next(iter(edge_phase_load.values()), 0),
    }


def expected_alltoall_value(source_rank: int, destination_rank: int) -> tuple[int, int]:
    return source_rank, destination_rank


def verify_island_bulk(n_islands: int, island_size: int, in_place: bool) -> dict[str, object]:
    rank_count = n_islands * island_size
    send_buffers = [
        [expected_alltoall_value(source, destination) for destination in range(rank_count)]
        for source in range(rank_count)
    ]
    recv_buffers = send_buffers if in_place else [[None] * rank_count for _ in range(rank_count)]

    # The first NCCL group completes before the implementation writes user recv buffers.
    stage = [[[None] * island_size for _ in range(n_islands)] for _ in range(rank_count)]
    for rank in range(rank_count):
        destination_island, island_local = divmod(rank, island_size)
        for source_island in range(n_islands):
            source_rank = source_island * island_size + island_local
            for destination_local in range(island_size):
                destination_rank = destination_island * island_size + destination_local
                stage[rank][source_island][destination_local] = send_buffers[source_rank][destination_rank]

    local_send: list[dict[int, list[tuple[int, int]]]] = [dict() for _ in range(rank_count)]
    for rank in range(rank_count):
        _, island_local = divmod(rank, island_size)
        for source_island in range(n_islands):
            source_rank = source_island * island_size + island_local
            recv_buffers[rank][source_rank] = stage[rank][source_island][island_local]
        for destination_local in range(island_size):
            if destination_local == island_local:
                continue
            local_send[rank][destination_local] = [
                stage[rank][source_island][destination_local]
                for source_island in range(n_islands)
            ]

    for rank in range(rank_count):
        destination_island, island_local = divmod(rank, island_size)
        for source_local in range(island_size):
            if source_local == island_local:
                continue
            source_peer = destination_island * island_size + source_local
            packed = local_send[source_peer][island_local]
            for source_island, value in enumerate(packed):
                source_rank = source_island * island_size + source_local
                recv_buffers[rank][source_rank] = value

    for destination_rank in range(rank_count):
        for source_rank in range(rank_count):
            expected = expected_alltoall_value(source_rank, destination_rank)
            actual = recv_buffers[destination_rank][source_rank]
            require(
                actual == expected,
                f"island-bulk mismatch in_place={in_place} dst={destination_rank} src={source_rank}: {actual}",
            )
    return {
        "kind": "a2a_island_bulk",
        "islands": n_islands,
        "island_size": island_size,
        "ranks": rank_count,
        "in_place": in_place,
        "elements_checked": rank_count * rank_count,
    }


def layouts(full_scale: bool) -> Iterable[ScheduleLayout]:
    topology_nodes = [8, 16, 32, 64]
    if full_scale:
        topology_nodes.extend([320, 336])
    return (ScheduleLayout(topology_nodes=count) for count in topology_nodes)


def run(full_scale: bool) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for layout in layouts(full_scale):
        results.append(verify_lane_schedule(layout))
        if layout.topology_nodes % layout.fabric_nodes == 0:
            results.append(verify_fabric_schedule(layout))
    for n_islands in (1, 2, 5, 16):
        for in_place in (False, True):
            results.append(verify_island_bulk(n_islands, island_size=4, in_place=in_place))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-scale",
        action="store_true",
        help="also model 20- and 21-fabric-group communicator shapes",
    )
    args = parser.parse_args()
    results = run(full_scale=args.full_scale)
    print(json.dumps({"status": "ok", "checks": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
