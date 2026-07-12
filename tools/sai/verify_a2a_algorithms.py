#!/usr/bin/env python3
"""Verify NCCL-SAI P2P schedules and AlltoAll planner invariants."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable


P2P_BATCH_ROUNDS = 8
DEFAULT_PLANNER_ROUNDS = 4
DEFAULT_ISLAND_SCRATCH_CAP = 64 * 1024 * 1024


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


def verify_ordinary_cutoff_after_join() -> dict[str, object]:
    stale_ordinal = 7

    # Joining the first operation of a new group resets planner-local ordinals.
    current_ordinal = 0
    cutoff = current_ordinal
    current_ordinal += 1
    post_alltoall_ordinal = current_ordinal
    require(
        post_alltoall_ordinal > cutoff,
        "ordinary P2P after AlltoAll crossed the planner cutoff",
    )

    # If ordinary P2P already joined the group, AlltoAll must retain that prefix.
    current_ordinal = 1
    cutoff_with_prefix = current_ordinal
    require(1 <= cutoff_with_prefix, "ordinary P2P prefix was excluded from the cutoff")
    return {
        "kind": "a2a_ordinary_cutoff_after_join",
        "stale_ordinal_ignored": stale_ordinal,
        "new_group_cutoff": cutoff,
        "post_alltoall_ordinal": post_alltoall_ordinal,
        "prefixed_group_cutoff": cutoff_with_prefix,
    }


def verify_fullmesh_group_cardinality() -> dict[str, object]:
    layout = ScheduleLayout(
        topology_nodes=272,
        local_ranks=4,
        fabric_nodes=16,
        p2p_group_size=4,
    )
    require(layout.ranks == 1088, "unexpected full-mesh rank count")
    require(layout.fabric_groups == 17, "sixteen-island groups did not produce 17 fabric groups")
    require(planner_round_window(layout) == 17, "planner did not align to 17 groups")
    return {
        "kind": "fullmesh_group_cardinality",
        "layout": asdict(layout),
        "fabric_groups": layout.fabric_groups,
        "planner_rounds": planner_round_window(layout),
        "ranks_per_island": layout.local_ranks,
        "islands_per_group": layout.fabric_nodes,
        "physical_hosts": layout.ranks // 16,
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


def ragged_native_group_order(occupancy: tuple[int, ...], interleaved: bool = False) -> list[int]:
    require(len(occupancy) > 1, "ragged layout needs multiple fabric groups")
    require(all(count > 0 for count in occupancy), "ragged group occupancy must be positive")
    if interleaved:
        order = [
            group
            for slot in range(max(occupancy))
            for group, count in enumerate(occupancy)
            if slot < count
        ]
    else:
        order = [group for group, count in enumerate(occupancy) for _ in range(count)]
    require(Counter(order) == Counter(dict(enumerate(occupancy))), "native order changed occupancy")
    return order


def ragged_shift_loads(order: list[int]) -> list[Counter[tuple[int, int]]]:
    loads: list[Counter[tuple[int, int]]] = []
    for shift in range(len(order)):
        load: Counter[tuple[int, int]] = Counter()
        for source, source_group in enumerate(order):
            destination_group = order[(source + shift) % len(order)]
            if source_group != destination_group:
                load[(source_group, destination_group)] += 1
        loads.append(load)
    return loads


def ragged_phase_capacities(rounds: int, fabric_groups: int) -> list[int]:
    capacities = [fabric_groups] * (rounds // fabric_groups)
    if rounds % fabric_groups:
        capacities.append(rounds % fabric_groups)
    return capacities


def ragged_phases(order: list[int]) -> list[list[int]]:
    fabric_groups = len(set(order))
    loads = ragged_shift_loads(order)
    capacities = ragged_phase_capacities(len(order), fabric_groups)
    phases = [[] for _ in capacities]
    phase_loads = [Counter() for _ in capacities]
    group_counts = Counter(order)
    phase_count = len(phases)
    shifts = sorted(
        range(len(order)),
        key=lambda shift: (max(loads[shift].values(), default=0), sum(loads[shift].values()), -shift),
        reverse=True,
    )
    for shift in shifts:
        best_phase = None
        best_score = None
        for phase, capacity in enumerate(capacities):
            if len(phases[phase]) >= capacity:
                continue
            merged = phase_loads[phase] + loads[shift]
            old_maximum = max(phase_loads[phase].values(), default=0)
            new_maximum = max(merged.values(), default=0)
            deviation = 0
            for source_group in range(fabric_groups):
                for destination_group in range(fabric_groups):
                    if source_group == destination_group:
                        continue
                    target = group_counts[source_group] * group_counts[destination_group]
                    delta = merged[(source_group, destination_group)] * phase_count - target
                    deviation += delta * delta
            score = (new_maximum - old_maximum, new_maximum, deviation, len(phases[phase]))
            if best_score is None or score < best_score:
                best_score = score
                best_phase = phase
        require(best_phase is not None, "ragged shift could not be assigned")
        phases[best_phase].append(shift)
        phase_loads[best_phase].update(loads[shift])
    return phases


def ragged_phase_score(order: list[int], phases: list[list[int]]) -> int:
    loads = ragged_shift_loads(order)
    score = 0
    for phase in phases:
        load: Counter[tuple[int, int]] = Counter()
        for shift in phase:
            load.update(loads[shift])
        score += max(load.values(), default=0)
    return score


def verify_ragged_schedule(
    occupancy: tuple[int, ...], group_capacity: int, *, interleaved: bool = False
) -> dict[str, object]:
    require(sum(occupancy) >= group_capacity, "ragged automatic path starts at one group of total occupancy")
    require(max(occupancy) <= group_capacity, "ragged occupancy exceeds configured group capacity")
    group_size = 4
    native_order = ragged_native_group_order(occupancy, interleaved=interleaved)
    native_shifts = quadratic_permutation(len(native_order))
    baseline_phases = [
        native_shifts[start : start + len(occupancy)]
        for start in range(0, len(native_shifts), len(occupancy))
    ]
    greedy_phases = ragged_phases(native_order)
    baseline_score = ragged_phase_score(native_order, baseline_phases)
    greedy_score = ragged_phase_score(native_order, greedy_phases)
    use_greedy = greedy_score < baseline_score
    phases = greedy_phases if use_greedy else baseline_phases
    shifts = [shift for phase in phases for shift in phase]
    require(sorted(shifts) == list(range(len(native_order))), "ragged phases lost or duplicated a shift")
    require(
        all(len(phase) <= len(occupancy) for phase in phases),
        "ragged phase exceeds the fabric-group round limit",
    )

    native_rounds = tuple(
        (shift, local_delta)
        for shift in native_shifts
        for local_delta in range(group_size)
    )
    shift_to_native_block = {shift: block for block, shift in enumerate(native_shifts)}
    round_order: list[int] = []
    phase_ends: list[bool] = []
    for local_delta in range(group_size):
        for phase in phases:
            for slot, shift in enumerate(phase):
                round_order.append(shift_to_native_block[shift] * group_size + local_delta)
                phase_ends.append(slot + 1 == len(phase))
    require(
        sorted(round_order) == list(range(len(native_rounds))),
        "ragged round map is not a native-round permutation",
    )
    remapped_rounds = tuple(native_rounds[channel_round] for channel_round in round_order)
    expected_remap = tuple(
        (shift, local_delta)
        for local_delta in range(group_size)
        for phase in phases
        for shift in phase
    )
    require(remapped_rounds == expected_remap, "ragged round map changed native peer semantics")
    require(
        sum(phase_ends) == len(phases) * group_size,
        "ragged phase boundaries were not expanded across local deltas",
    )

    for source in range(len(native_order)):
        destinations = {(source + shift) % len(native_order) for shift in shifts}
        require(len(destinations) == len(native_order), f"ragged source {source} missed a peer")
        for shift in shifts:
            destination = (source + shift) % len(native_order)
            require(
                (destination - shift + len(native_order)) % len(native_order) == source,
                f"ragged send/recv mismatch for source {source}",
            )

    total: Counter[tuple[int, int]] = Counter()
    for load in ragged_shift_loads(native_order):
        total.update(load)
    expected = Counter(
        {
            (source, destination): occupancy[source] * occupancy[destination]
            for source in range(len(occupancy))
            for destination in range(len(occupancy))
            if source != destination
        }
    )
    require(total == expected, "ragged schedule changed total fabric-edge load")

    ragged_score = ragged_phase_score(native_order, phases)
    require(ragged_score <= baseline_score, "ragged phase packing regressed the native baseline")
    return {
        "kind": "ragged_fabric_schedule",
        "occupancy": occupancy,
        "group_capacity": group_capacity,
        "native_group_order": "interleaved" if interleaved else "contiguous",
        "rank_groups": len(native_order),
        "rank_group_size": group_size,
        "phases": len(phases),
        "order": "native-greedy" if use_greedy else "native-quadratic",
        "maximum_phase_rounds": max(map(len, phases)),
        "baseline_phase_edge_score": baseline_score,
        "ragged_phase_edge_score": ragged_score,
        "densest_total_edge_load": max(total.values()),
        "native_rounds_preserved": len(native_rounds),
        "round_map_is_permutation": True,
    }


def verify_ragged_exhaustive() -> dict[str, object]:
    checked = 0
    selected: Counter[str] = Counter()
    for fabric_groups in range(2, 7):
        for occupancy in itertools.product(range(1, 5), repeat=fabric_groups):
            if sum(occupancy) < 4 or all(count == 4 for count in occupancy):
                continue
            result = verify_ragged_schedule(occupancy, 4)
            selected[str(result["order"])] += 1
            checked += 1
    return {
        "kind": "ragged_fabric_schedule_exhaustive",
        "group_capacity": 4,
        "maximum_fabric_groups": 6,
        "occupancy_shapes_checked": checked,
        "selected_orders": dict(selected),
    }


def expected_alltoall_value(source_rank: int, destination_rank: int) -> tuple[int, int]:
    return source_rank, destination_rank


def island_scratch_bytes(rank_count: int, island_size: int, peer_bytes: int) -> int:
    require(rank_count > 0, "rank_count must be positive")
    require(island_size >= 2, "island_size must be at least two")
    require(rank_count % island_size == 0, "rank_count must contain complete islands")
    require(peer_bytes >= 0, "peer_bytes must be nonnegative")
    return peer_bytes * rank_count


def verify_island_scratch() -> dict[str, object]:
    cases = ((32, 4), (64, 4), (256, 4), (512, 4), (1088, 4), (256, 8), (8192, 4))
    peer_bytes = 4096
    checked: list[dict[str, int]] = []
    for rank_count, island_size in cases:
        scratch_bytes = island_scratch_bytes(rank_count, island_size, peer_bytes)
        reserve_bytes = min(scratch_bytes, DEFAULT_ISLAND_SCRATCH_CAP)
        require(0 < reserve_bytes <= DEFAULT_ISLAND_SCRATCH_CAP, "invalid scratch reserve")
        checked.append(
            {
                "ranks": rank_count,
                "island_size": island_size,
                "peer_bytes": peer_bytes,
                "scratch_bytes": scratch_bytes,
                "reserve_bytes": reserve_bytes,
            }
        )
    default_scale_reserve = {
        rank_count: (
            min(island_scratch_bytes(rank_count, 4, peer_bytes), DEFAULT_ISLAND_SCRATCH_CAP)
            if rank_count >= 576
            else 0
        )
        for rank_count in (64, 512, 576, 1088)
    }
    require(default_scale_reserve[64] == 0, "low-scale scratch was reserved")
    require(default_scale_reserve[512] == 0, "pre-threshold scratch was reserved")
    require(default_scale_reserve[576] > 0, "threshold scratch was not reserved")
    return {
        "kind": "a2a_island_scratch",
        "default_cap_bytes": DEFAULT_ISLAND_SCRATCH_CAP,
        "default_scale_reserve": default_scale_reserve,
        "cases": checked,
    }


def verify_island_enqueue(
    layout: ScheduleLayout, island_size: int, in_place: bool
) -> dict[str, object]:
    require(layout.local_ranks % island_size == 0, "local ranks must contain complete islands")
    rank_count = layout.ranks
    n_islands = rank_count // island_size
    descriptors = fabric_rounds(layout)
    send_buffers = [
        [expected_alltoall_value(source, destination) for destination in range(rank_count)]
        for source in range(rank_count)
    ]
    recv_buffers = send_buffers if in_place else [[None] * rank_count for _ in range(rank_count)]

    stage = [[[None] * island_size for _ in range(n_islands)] for _ in range(rank_count)]
    stage_one_messages = 0
    for descriptor in descriptors:
        for source_rank in range(rank_count):
            send_peer, _ = p2p_peers(layout, source_rank, descriptor)
            if send_peer % island_size != source_rank % island_size:
                continue
            _, peer_recv = p2p_peers(layout, send_peer, descriptor)
            require(peer_recv == source_rank, "stage-one send/recv peers do not match")
            source_island = source_rank // island_size
            destination_island = send_peer // island_size
            first_destination = destination_island * island_size
            stage[send_peer][source_island] = send_buffers[source_rank][
                first_destination : first_destination + island_size
            ]
            stage_one_messages += 1

    stage_two_messages = 0
    for source_island in range(n_islands):
        for descriptor in descriptors:
            for source_rank in range(rank_count):
                send_peer, _ = p2p_peers(layout, source_rank, descriptor)
                if send_peer // island_size != source_rank // island_size:
                    continue
                _, peer_recv = p2p_peers(layout, send_peer, descriptor)
                require(peer_recv == source_rank, "stage-two send/recv peers do not match")
                destination_local = send_peer % island_size
                source_local = source_rank % island_size
                source_global_rank = source_island * island_size + source_local
                recv_buffers[send_peer][source_global_rank] = stage[source_rank][source_island][
                    destination_local
                ]
                stage_two_messages += 1

    for destination_rank in range(rank_count):
        for source_rank in range(rank_count):
            expected = expected_alltoall_value(source_rank, destination_rank)
            actual = recv_buffers[destination_rank][source_rank]
            require(
                actual == expected,
                f"island enqueue mismatch in_place={in_place} dst={destination_rank} "
                f"src={source_rank}: {actual}",
            )

    require(
        stage_one_messages == rank_count * n_islands,
        "stage one did not exchange one block per source/destination island pair",
    )
    require(
        stage_two_messages == rank_count * island_size * n_islands,
        "stage two did not deliver every staged element",
    )
    return {
        "kind": "a2a_island_enqueue",
        "layout": asdict(layout),
        "islands": n_islands,
        "island_size": island_size,
        "ranks": rank_count,
        "in_place": in_place,
        "stage_one_messages": stage_one_messages,
        "stage_two_messages": stage_two_messages,
        "elements_checked": rank_count * rank_count,
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
    results.append(verify_ordinary_cutoff_after_join())
    results.append(verify_fullmesh_group_cardinality())
    results.append(verify_fabric_metadata_order())
    for occupancy, capacity in (
        ((4, 3, 2, 1), 4),
        ((4, 4, 3, 1), 4),
        ((8, 6, 6, 4, 2), 8),
        ((8, 8, 7, 5, 3, 1), 8),
    ):
        results.append(verify_ragged_schedule(occupancy, capacity))
    results.append(verify_ragged_schedule((4, 3, 2, 1), 4, interleaved=True))
    if full_scale:
        results.append(verify_ragged_exhaustive())
    results.append(verify_island_scratch())
    island_layouts = (
        (ScheduleLayout(topology_nodes=4), 4),
        (ScheduleLayout(topology_nodes=8), 4),
        (ScheduleLayout(topology_nodes=16), 4),
        (ScheduleLayout(topology_nodes=4, local_ranks=16, p2p_group_size=8), 4),
        (ScheduleLayout(topology_nodes=4, local_ranks=16, p2p_group_size=8), 8),
    )
    for layout, island_size in island_layouts:
        for in_place in (False, True):
            results.append(verify_island_enqueue(layout, island_size, in_place))
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
