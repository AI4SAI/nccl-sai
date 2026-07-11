#!/usr/bin/env python3
"""Verify NCCL-SAI P2P schedules and phased AlltoAll planner invariants."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable


P2P_BATCH_ROUNDS = 8
DEFAULT_PLANNER_ROUNDS = 4


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
    fabric_nodes: int = 4
    p2p_group_size: int = 4

    @property
    def ranks(self) -> int:
        return self.topology_nodes * self.local_ranks

    @property
    def fabric_groups(self) -> int:
        return self.topology_nodes // self.fabric_nodes

    def validate_fabric_schedule(self) -> None:
        require(self.topology_nodes > 0, "topology_nodes must be positive")
        require(self.local_ranks > 0, "local_ranks must be positive")
        require(self.fabric_nodes > 0, "fabric_nodes must be positive")
        require(self.p2p_group_size > 0, "p2p_group_size must be positive")
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


def planner_round_window(layout: ScheduleLayout, configured_rounds: int = -1) -> int:
    layout.validate_fabric_schedule()
    if configured_rounds > 0:
        return configured_rounds
    require(configured_rounds == -1, "configured planner rounds must be -1 or positive")
    complete_cycles = (DEFAULT_PLANNER_ROUNDS + layout.fabric_groups - 1) // layout.fabric_groups
    return layout.fabric_groups * complete_cycles


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
    required_undirected_edges = layout.fabric_groups * (layout.fabric_groups - 1) // 2
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
            required_undirected_edges > 0
            and
            first_full_undirected_batch == 0
            and len(cumulative_edges) == required_undirected_edges
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


def verify_planner_phases(layout: ScheduleLayout, rounds_per_plan: int = DEFAULT_PLANNER_ROUNDS) -> dict[str, object]:
    require(rounds_per_plan > 0, "rounds_per_plan must be positive")
    descriptors = fabric_rounds(layout)
    rank_count = layout.ranks
    covered: list[set[int]] = [set() for _ in range(rank_count)]
    phase_sizes: list[int] = []
    for phase_begin in range(0, rank_count, rounds_per_plan):
        phase = descriptors[phase_begin : phase_begin + rounds_per_plan]
        phase_sizes.append(len(phase))
        for rank in range(rank_count):
            for descriptor in phase:
                send_peer, recv_peer = p2p_peers(layout, rank, descriptor)
                require(
                    p2p_peers(layout, send_peer, descriptor)[1] == rank,
                    f"planner phase send/recv mismatch at rank {rank}",
                )
                require(
                    p2p_peers(layout, recv_peer, descriptor)[0] == rank,
                    f"planner phase recv/send mismatch at rank {rank}",
                )
                covered[rank].add(send_peer)
    for rank, peers in enumerate(covered):
        require(len(peers) == rank_count, f"planner phases missed peers for rank {rank}")
    return {
        "kind": "a2a_native_planner_phases",
        "layout": asdict(layout),
        "ranks": rank_count,
        "rounds_per_plan": rounds_per_plan,
        "plans": len(phase_sizes),
        "minimum_rounds_per_plan": min(phase_sizes),
        "maximum_rounds_per_plan": max(phase_sizes),
    }


def verify_planner_fabric_edge_phases(
    layout: ScheduleLayout, rounds_per_plan: int
) -> dict[str, object]:
    require(rounds_per_plan > 0, "rounds_per_plan must be positive")
    descriptors = fabric_rounds(layout)
    expected_edges = layout.fabric_groups * (layout.fabric_groups - 1)
    active_edge_counts: list[int] = []
    messages_per_edge: list[int] = []
    for phase_begin in range(0, layout.ranks, rounds_per_plan):
        phase_load: Counter[tuple[int, int]] = Counter()
        for rank in range(layout.ranks):
            source_group = fabric_group(layout, rank)
            for descriptor in descriptors[phase_begin : phase_begin + rounds_per_plan]:
                send_peer, _ = p2p_peers(layout, rank, descriptor)
                destination_group = fabric_group(layout, send_peer)
                if source_group != destination_group:
                    phase_load[(source_group, destination_group)] += 1
        active_edge_counts.append(len(phase_load))
        if expected_edges:
            require(len(phase_load) == expected_edges, "planner phase left fabric edges idle")
            require(len(set(phase_load.values())) == 1, "planner phase fabric load is imbalanced")
            messages_per_edge.append(next(iter(phase_load.values())))
    return {
        "kind": "a2a_planner_fabric_edge_phases",
        "layout": asdict(layout),
        "rounds_per_plan": rounds_per_plan,
        "phases": len(active_edge_counts),
        "minimum_active_directed_edges": min(active_edge_counts),
        "maximum_active_directed_edges": max(active_edge_counts),
        "minimum_messages_per_edge": min(messages_per_edge, default=0),
        "maximum_messages_per_edge": max(messages_per_edge, default=0),
    }


def verify_planner_queue_model(
    layout: ScheduleLayout,
    sequences: tuple[int, ...],
    rounds_per_phase: int,
    work_round_budget: int,
    skip_self: bool,
) -> dict[str, object]:
    require(sequences and all(sequence > 0 for sequence in sequences), "planner sequences must be nonzero")
    require(len(set(sequences)) == len(sequences), "planner sequences must be unique")
    require(rounds_per_phase > 0, "rounds_per_phase must be positive")
    require(work_round_budget > 0, "work_round_budget must be positive")
    descriptors = fabric_rounds(layout)
    rank = 0
    send_queues = [list(sequences) for _ in range(layout.ranks)]
    recv_queues = [list(sequences) for _ in range(layout.ranks)]
    plans: list[tuple[int, tuple[int, ...]]] = []
    empty_plans_released = 0
    covered: dict[int, set[int]] = {sequence: set() for sequence in sequences}

    for sequence in sequences:
        heads = [queue[0] for queue in send_queues + recv_queues if queue]
        require(heads and min(heads) == sequence, f"sequence {sequence} is not at the queue heads")
        require(all(head == sequence for head in heads), f"sequence {sequence} heads are inconsistent")
        next_round = 0
        round_end = 0
        while next_round < layout.ranks:
            if round_end == 0:
                round_end = min(layout.ranks, next_round + rounds_per_phase)
            plan_rounds: list[int] = []
            while next_round < round_end:
                send_peer, recv_peer = p2p_peers(layout, rank, descriptors[next_round])
                require(send_queues[send_peer][0] == sequence, "send queue sequence changed mid-phase")
                require(recv_queues[recv_peer][0] == sequence, "recv queue sequence changed mid-phase")
                self_noop = skip_self and send_peer == rank
                if self_noop:
                    require(recv_peer == rank, "self send did not have a matching self receive")
                elif len(plan_rounds) == work_round_budget:
                    break
                send_queues[send_peer].pop(0)
                recv_queues[recv_peer].pop(0)
                covered[sequence].add(next_round)
                if not self_noop:
                    plan_rounds.append(next_round)
                next_round += 1

            if plan_rounds:
                plans.append((sequence, tuple(plan_rounds)))
            else:
                empty_plans_released += 1
            if next_round == round_end:
                round_end = 0
            else:
                require(plan_rounds, "planner made no progress while a phase remained active")

        require(
            covered[sequence] == set(range(layout.ranks)),
            f"sequence {sequence} did not consume every P2P round exactly once",
        )

    require(all(not queue for queue in send_queues + recv_queues), "planner left tagged tasks queued")
    require(all(plan for _, plan in plans), "planner retained an empty kernel plan")
    if skip_self:
        require(empty_plans_released > 0, "self-only phase did not exercise empty-plan release")
    return {
        "kind": "a2a_native_planner_queue_model",
        "layout": asdict(layout),
        "sequences": list(sequences),
        "rounds_per_phase": rounds_per_phase,
        "work_round_budget": work_round_budget,
        "skip_self": skip_self,
        "nonempty_plans": len(plans),
        "empty_plans_released": empty_plans_released,
    }


def verify_planner_head_guard() -> dict[str, object]:
    send_heads = [1, 1, 1, 2]
    recv_heads = [1, 1, 1, 1]
    candidate = min(send_heads + recv_heads)
    ready = all(head == candidate for head in send_heads + recv_heads)
    require(not ready, "planner accepted inconsistent tagged queue heads")
    return {
        "kind": "a2a_native_planner_head_guard",
        "candidate_sequence": candidate,
        "ready": ready,
    }


def fabric_metadata_order(group_ids: tuple[int, ...], expected_nodes_per_group: int) -> tuple[list[int], bool]:
    require(group_ids, "fabric metadata must contain at least one topology node")
    require(expected_nodes_per_group > 0, "expected fabric group size must be positive")
    dense_ids: list[int] = []
    node_to_group: list[int] = []
    counts: Counter[int] = Counter()
    for group_id in group_ids:
        if group_id not in dense_ids:
            dense_ids.append(group_id)
        dense_group = dense_ids.index(group_id)
        node_to_group.append(dense_group)
        counts[dense_group] += 1
    order = [
        node
        for dense_group in range(len(dense_ids))
        for node, node_group in enumerate(node_to_group)
        if node_group == dense_group
    ]
    complete = len(group_ids) == len(dense_ids) * expected_nodes_per_group and all(
        counts[dense_group] == expected_nodes_per_group for dense_group in range(len(dense_ids))
    )
    require(sorted(order) == list(range(len(group_ids))), "fabric metadata order lost a topology node")
    return order, complete


def verify_fabric_metadata_order() -> dict[str, object]:
    interleaved = (90, 70, 90, 70, 90, 70, 90, 70)
    order, complete = fabric_metadata_order(interleaved, 4)
    require(complete, "complete interleaved fabric groups were rejected")
    require(order == [0, 2, 4, 6, 1, 3, 5, 7], "fabric metadata order is not group-contiguous")
    _, incomplete = fabric_metadata_order((90, 70, 90, 70, 90, 70, 90), 4)
    require(not incomplete, "incomplete fabric groups were accepted")
    return {
        "kind": "fabric_group_metadata_order",
        "topology_nodes": len(interleaved),
        "expected_nodes_per_group": 4,
        "ordered_nodes": order,
        "complete": complete,
    }


def layouts(full_scale: bool) -> Iterable[ScheduleLayout]:
    topology_nodes = [4, 8, 16, 32, 64]
    if full_scale:
        # Keep public stress shapes representative rather than mirroring a
        # particular deployed system. Cover both power-of-two and
        # non-power-of-two fabric-group counts at large communicator sizes.
        topology_nodes.extend([256, 272])
    result = [ScheduleLayout(topology_nodes=count) for count in topology_nodes]
    # Multi-node NCCL starts from an eight-rank P2P schedule group and reduces
    # it with gcd() when a node has fewer local ranks. Exercise the resulting
    # 4-, 8-, and 16-local-rank layouts without making the large model cubic in
    # the full communicator size.
    result.extend(
        [
            ScheduleLayout(topology_nodes=4, local_ranks=16, p2p_group_size=8),
            ScheduleLayout(topology_nodes=8, local_ranks=8, p2p_group_size=8),
            ScheduleLayout(topology_nodes=8, local_ranks=16, p2p_group_size=8),
            ScheduleLayout(topology_nodes=16, fabric_nodes=16),
        ]
    )
    if full_scale:
        # Also exercise an explicit larger expert configuration without making
        # it the model default used by the runtime knob.
        result.extend(ScheduleLayout(topology_nodes=count, fabric_nodes=16) for count in (256, 272))
    return result


def run(full_scale: bool) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for layout in layouts(full_scale):
        if layout.topology_nodes % layout.fabric_nodes == 0:
            results.append(verify_fabric_schedule(layout))
            rounds_per_plan = planner_round_window(layout)
            results.append(verify_planner_phases(layout, rounds_per_plan))
            results.append(verify_planner_fabric_edge_phases(layout, rounds_per_plan))
    queue_layout = ScheduleLayout(topology_nodes=4, fabric_nodes=4)
    results.append(verify_planner_queue_model(queue_layout, (1, 2), 8, 3, False))
    results.append(verify_planner_queue_model(queue_layout, (1,), 1, 1, True))
    results.append(
        verify_planner_queue_model(
            ScheduleLayout(topology_nodes=5, fabric_nodes=1),
            (7, 8),
            3,
            2,
            False,
        )
    )
    results.append(verify_planner_head_guard())
    results.append(verify_fabric_metadata_order())
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-scale",
        action="store_true",
        help="also model large power-of-two and non-power-of-two communicator shapes",
    )
    args = parser.parse_args()
    results = run(full_scale=args.full_scale)
    print(json.dumps({"status": "ok", "checks": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
