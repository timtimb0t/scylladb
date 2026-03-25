#
# Copyright (C) 2026-present ScyllaDB
#
# SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
#
"""
LWT Performance Benchmark: VNodes vs Tablets

Compares LWT/CAS performance across four cluster configurations:
  1-node VNodes, 3-node VNodes, 1-node Tablets, 3-node Tablets

Workload scenarios (each in unthrottled + throttled modes):
  write_light  — INSERT ... IF NOT EXISTS on a narrow table
  write_heavy  — conditional UPDATE ... IF v1=? AND v2=? AND v3=? AND v4=? AND v5=?
  read_light   — SELECT at LOCAL_SERIAL on a narrow table (serial/Paxos read)
  read_heavy   — SELECT at LOCAL_SERIAL on a wide table
  mixed_light  — 50/50 write_light + read_light
  mixed_heavy  — 50/50 write_heavy + read_heavy

Note on read semantics: all "read" scenarios use LOCAL_SERIAL consistency,
meaning they exercise the Paxos serial-read path. This is our chosen
interpretation of "read-only LWT" for this benchmark.

Environment variables:
  LWT_BENCH_DURATION      — measurement duration in seconds       (default 60)
  LWT_BENCH_WARMUP        — warmup duration in seconds            (default 10)
  LWT_BENCH_CONCURRENCY   — concurrent async workers              (default 32)
  LWT_BENCH_NUM_KEYS      — pre-populated partitions              (default 1000)
  LWT_BENCH_TARGET_OPS    — target ops/sec for throttled mode     (default 200)
"""

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import List, Callable, Awaitable, Any

import pytest
from cassandra import ConsistencyLevel, WriteTimeout, ReadTimeout, OperationTimedOut

from test.cluster.util import create_new_test_keyspace
from test.cluster.metrics_helpers import (
    snapshot_latency_metrics, compute_latency_from_snapshots,
)
from test.pylib.manager_client import ManagerClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


BENCH_DURATION = 60
BENCH_WARMUP = 10
BENCH_CONCURRENCY = 64
NUM_KEYS = 1000
TARGET_OPS = 200

# write_light uses keys from a separate high range so most INSERTs are applied=true
WRITE_KEY_BASE_UNTHROTTLED = 1_000_000
WRITE_KEY_BASE_THROTTLED = 101_000_000

MIXED_LIGHT_WRITE_KEY_BASE_UNTHROTTLED = 201_000_000
MIXED_LIGHT_WRITE_KEY_BASE_THROTTLED = 301_000_000

# Table payload sizes
LIGHT_PAYLOAD_SIZE = 100          # ~100 bytes for light_tbl.payload
HEAVY_COL_PAYLOAD_SIZE = 500      # ~500 bytes per d* column in heavy_read_tbl

# Population batch size for parallel inserts
POPULATION_BATCH = 200

# Known fixed values for heavy_write_tbl condition columns
HEAVY_WRITE_KNOWN_VALUE = 1

# no mixed scenarios for now
SCENARIOS = [
    "write_light", "write_heavy",
    "read_light", "read_heavy",
]


@dataclass
class BenchmarkMetrics:
    """Benchmark result for a single (scenario, mode, cluster_config) run.

    Latency percentiles are computed from server-side Prometheus histograms
    (delta between snapshots taken before/after the measurement window).
    """
    # Identity
    cluster_config: str = ""
    scenario: str = ""
    mode: str = ""

    # Cluster metadata
    nodes: int = 0
    tablets_enabled: bool = False
    rf: int = 0

    # Run parameters
    concurrency: int = 0
    warmup_sec: int = 0
    duration_sec: int = 0
    num_keys: int = 0
    target_rate_ops_sec: int = 0

    # Measured — populated in v1
    wall_clock_sec: float = 0.0
    attempted_ops: int = 0
    successful_ops: int = 0
    applied_ops: int = 0
    not_applied_ops: int = 0
    errors: int = 0
    achieved_ops_sec: float = 0.0
    applied_rate: float = 0.0          # applied_ops / successful_ops

    # Measured — latency from server-side Prometheus histograms (ms)
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    @staticmethod
    def make(*, cluster_config: str, scenario: str, mode: str,
             nodes: int, tablets_enabled: bool, rf: int) -> "BenchmarkMetrics":
        """Create a metrics object pre-filled with run identity and params."""
        return BenchmarkMetrics(
            cluster_config=cluster_config,
            scenario=scenario,
            mode=mode,
            nodes=nodes,
            tablets_enabled=tablets_enabled,
            rf=rf,
            concurrency=BENCH_CONCURRENCY,
            warmup_sec=BENCH_WARMUP,
            duration_sec=BENCH_DURATION,
            num_keys=NUM_KEYS,
            target_rate_ops_sec=TARGET_OPS,
        )


OpFn = Callable[[Any, random.Random], Awaitable[bool]]


# Each factory accepts the CQL session and table metadata, and returns an
# async callable  ``op(cql, rng)``  that performs one benchmark operation.
def _make_write_light(cql, table: str, payload: str, key_base: int) -> OpFn:
    """INSERT ... IF NOT EXISTS on the light table.

    Uses keys from WRITE_KEY_BASE + random offset, so the vast majority
    of inserts hit non-existing keys and are applied=true.
    """
    ps = cql.prepare(
        f"INSERT INTO {table} (pk, v, payload) VALUES (?, ?, ?) IF NOT EXISTS"
    )

    async def op(session, rng: random.Random):
        pk = key_base + rng.randint(0, 10_000_000)
        b = ps.bind([pk, pk, payload])
        b.consistency_level = ConsistencyLevel.LOCAL_QUORUM
        b.serial_consistency_level = ConsistencyLevel.LOCAL_SERIAL
        res = await session.run_async(b)
        return bool(res and res[0].applied)

    return op


def _make_write_heavy(cql, table: str, payload: str) -> OpFn:
    """Conditional UPDATE with a multi-predicate IF clause.

    The heavy_write_tbl is pre-populated with v1..v5 = HEAVY_WRITE_KNOWN_VALUE.
    The worker always sends IF v1=1 AND v2=1 AND v3=1 AND v4=1 AND v5=1,
    and only updates the payload column. This ensures the CAS path is
    exercised with a semantically heavy condition evaluation while keeping
    the known-value model stable (condition columns are never mutated).
    """
    ps = cql.prepare(
        f"UPDATE {table} SET payload = ? WHERE pk = ? "
        f"IF v1 = ? AND v2 = ? AND v3 = ? AND v4 = ? AND v5 = ?"
    )
    kv = HEAVY_WRITE_KNOWN_VALUE

    async def op(session, rng: random.Random):
        pk = rng.randint(0, NUM_KEYS - 1)
        b = ps.bind([payload, pk, kv, kv, kv, kv, kv])
        b.consistency_level = ConsistencyLevel.LOCAL_QUORUM
        b.serial_consistency_level = ConsistencyLevel.LOCAL_SERIAL
        res = await session.run_async(b)
        return bool(res and res[0].applied)

    return op


def _make_read_light(cql, table: str) -> OpFn:
    """Serial read on the light table (narrow row, small payload)."""
    ps = cql.prepare(f"SELECT * FROM {table} WHERE pk = ?")

    async def op(session, rng: random.Random):
        b = ps.bind([rng.randint(0, NUM_KEYS - 1)])
        b.consistency_level = ConsistencyLevel.LOCAL_SERIAL
        await session.run_async(b)
        return True  # reads have no applied/not-applied semantics

    return op


def _make_read_heavy(cql, table: str) -> OpFn:
    """Serial read on the heavy read table (wide row, ~2.5KB payload)."""
    ps = cql.prepare(f"SELECT * FROM {table} WHERE pk = ?")

    async def op(session, rng: random.Random):
        b = ps.bind([rng.randint(0, NUM_KEYS - 1)])
        b.consistency_level = ConsistencyLevel.LOCAL_SERIAL
        await session.run_async(b)
        return True  # reads have no applied/not-applied semantics

    return op


def _make_mixed_light(cql, light_tbl: str, light_payload: str, key_base: int) -> OpFn:
    """50/50 mix of write_light + read_light.

    Writes go to the separate high key range; reads go to pre-populated keys.
    """
    w_ps = cql.prepare(
        f"INSERT INTO {light_tbl} (pk, v, payload) VALUES (?, ?, ?) IF NOT EXISTS"
    )
    r_ps = cql.prepare(f"SELECT * FROM {light_tbl} WHERE pk = ?")

    async def op(session, rng: random.Random):
        if rng.random() < 0.5:
            pk = key_base + rng.randint(0, 10_000_000)
            b = w_ps.bind([pk, pk, light_payload])
            b.consistency_level = ConsistencyLevel.LOCAL_QUORUM
            b.serial_consistency_level = ConsistencyLevel.LOCAL_SERIAL
            res = await session.run_async(b)
            return bool(res and res[0].applied)
        else:
            b = r_ps.bind([rng.randint(0, NUM_KEYS - 1)])
            b.consistency_level = ConsistencyLevel.LOCAL_SERIAL
            await session.run_async(b)
            return True  # reads have no applied/not-applied semantics

    return op


def _make_mixed_heavy(cql, heavy_write_tbl: str, heavy_read_tbl: str,
                      heavy_payload: str) -> OpFn:
    """50/50 mix of write_heavy + read_heavy.

    Writes go to heavy_write_tbl (conditional UPDATE on pre-populated keys).
    Reads go to heavy_read_tbl (serial SELECT on wide rows).
    """
    w_ps = cql.prepare(
        f"UPDATE {heavy_write_tbl} SET payload = ? WHERE pk = ? "
        f"IF v1 = ? AND v2 = ? AND v3 = ? AND v4 = ? AND v5 = ?"
    )
    r_ps = cql.prepare(f"SELECT * FROM {heavy_read_tbl} WHERE pk = ?")
    kv = HEAVY_WRITE_KNOWN_VALUE

    async def op(session, rng: random.Random):
        if rng.random() < 0.5:
            pk = rng.randint(0, NUM_KEYS - 1)
            b = w_ps.bind([heavy_payload, pk, kv, kv, kv, kv, kv])
            b.consistency_level = ConsistencyLevel.LOCAL_QUORUM
            b.serial_consistency_level = ConsistencyLevel.LOCAL_SERIAL
            res = await session.run_async(b)
            return bool(res and res[0].applied)
        else:
            b = r_ps.bind([rng.randint(0, NUM_KEYS - 1)])
            b.consistency_level = ConsistencyLevel.LOCAL_SERIAL
            await session.run_async(b)
            return True  # reads have no applied/not-applied semantics

    return op


# Data population
async def _populate_light(cql, table: str, n: int, payload: str):
    """Populate light_tbl with n rows: pk=0..n-1, v=pk, payload=fixed."""
    ps = cql.prepare(f"INSERT INTO {table} (pk, v, payload) VALUES (?, ?, ?)")
    for start in range(0, n, POPULATION_BATCH):
        end = min(start + POPULATION_BATCH, n)
        await asyncio.gather(
            *(cql.run_async(ps.bind([pk, pk, payload]))
              for pk in range(start, end))
        )
    logger.info("Populated %s with %d rows", table, n)


async def _populate_heavy_read(cql, table: str, n: int, col_payload: str):
    """Populate heavy_read_tbl with n rows: pk=0..n-1, d1..d5=col_payload."""
    ps = cql.prepare(
        f"INSERT INTO {table} (pk, v, d1, d2, d3, d4, d5) VALUES (?, ?, ?, ?, ?, ?, ?)"
    )
    for start in range(0, n, POPULATION_BATCH):
        end = min(start + POPULATION_BATCH, n)
        await asyncio.gather(
            *(cql.run_async(ps.bind(
                [pk, pk, col_payload, col_payload, col_payload,
                 col_payload, col_payload]))
              for pk in range(start, end))
        )
    logger.info("Populated %s with %d rows", table, n)


async def _populate_heavy_write(cql, table: str, n: int, payload: str):
    """Populate heavy_write_tbl with n rows: v1..v5=KNOWN_VALUE, payload=fixed.

    All rows must exist with known condition-column values so that
    conditional UPDATE ... IF v1=1 AND ... succeeds.
    """
    kv = HEAVY_WRITE_KNOWN_VALUE
    ps = cql.prepare(
        f"INSERT INTO {table} (pk, v1, v2, v3, v4, v5, payload) "
        f"VALUES (?, ?, ?, ?, ?, ?, ?)"
    )
    for start in range(0, n, POPULATION_BATCH):
        end = min(start + POPULATION_BATCH, n)
        await asyncio.gather(
            *(cql.run_async(ps.bind([pk, kv, kv, kv, kv, kv, payload]))
              for pk in range(start, end))
        )
    logger.info("Populated %s with %d rows (v1..v5=%d)", table, n, kv)


async def _run_benchmark(
    cql,
    op_fn: OpFn,
    concurrency: int,
    duration_s: int,
    warmup_s: int,
    on_measurement_start=None,
    on_measurement_end=None,
) -> dict:
    """Run *concurrency* async workers calling *op_fn* for *duration_s* seconds
    (after *warmup_s* seconds of warmup). Returns raw counters dict.

    Optional async callbacks:
        on_measurement_start — called after warmup, right before counters start.
        on_measurement_end   — called after measurement, after workers stop.
    These are used for snapping server-side Prometheus metrics.
    """
    stop = asyncio.Event()
    measuring = asyncio.Event()

    # Per-worker counters (no lock needed — each slot is single-writer)
    w_attempted = [0] * concurrency
    w_success = [0] * concurrency
    w_applied = [0] * concurrency
    w_not_applied = [0] * concurrency
    w_errors = [0] * concurrency

    async def worker(wid: int):
        rng = random.Random(wid * 31337)
        while not stop.is_set():
            try:
                applied = await op_fn(cql, rng)
                if measuring.is_set():
                    w_attempted[wid] += 1
                    w_success[wid] += 1
                    if applied:
                        w_applied[wid] += 1
                    else:
                        w_not_applied[wid] += 1
            except (WriteTimeout, ReadTimeout, OperationTimedOut):
                if measuring.is_set():
                    w_attempted[wid] += 1
                    w_errors[wid] += 1
            except Exception:
                if measuring.is_set():
                    w_attempted[wid] += 1
                    w_errors[wid] += 1

    tasks = [asyncio.create_task(worker(i)) for i in range(concurrency)]

    # Warmup phase — operations run but are not counted
    await asyncio.sleep(warmup_s)

    # Snap server-side metrics BEFORE measurement
    if on_measurement_start:
        await on_measurement_start()

    # Measurement phase
    measuring.set()
    t_start = time.monotonic()
    await asyncio.sleep(duration_s)
    wall = time.monotonic() - t_start
    stop.set()

    await asyncio.gather(*tasks, return_exceptions=True)

    # Snap server-side metrics AFTER measurement (all in-flight ops landed)
    if on_measurement_end:
        await on_measurement_end()

    total_attempted = sum(w_attempted)
    total_success = sum(w_success)
    total_applied = sum(w_applied)
    total_not_applied = sum(w_not_applied)
    total_errors = sum(w_errors)

    return {
        "wall_clock_sec": round(wall, 3),
        "attempted_ops": total_attempted,
        "successful_ops": total_success,
        "applied_ops": total_applied,
        "not_applied_ops": total_not_applied,
        "errors": total_errors,
        "achieved_ops_sec": round(total_success / wall, 2) if wall > 0 else 0.0,
    }


def _fmt_summary(results: List[BenchmarkMetrics]) -> str:
    """Format a human-readable table of benchmark results."""
    hdr = (
        f"{'Scenario':<14} {'Mode':<12} {'Ops/s':>10} "
        f"{'OK':>8} {'Applied':>8} {'NotApp':>7} {'Err':>5} {'Wall':>7} {'ApplRate':>9} "
        f"{'Avg ms':>8} {'p50 ms':>8} {'p95 ms':>8} {'p99 ms':>8}"
    )
    sep = "-" * len(hdr)
    lines = [sep, hdr, sep]
    for r in results:
        lines.append(
            f"{r.scenario:<14} {r.mode:<12} {r.achieved_ops_sec:>10.1f} "
            f"{r.successful_ops:>8} {r.applied_ops:>8} {r.not_applied_ops:>7} "
            f"{r.errors:>5} {r.wall_clock_sec:>7.1f} {r.applied_rate:>8.1%} "
            f"{r.avg_latency_ms:>8.3f} {r.p50_latency_ms:>8.3f} "
            f"{r.p95_latency_ms:>8.3f} {r.p99_latency_ms:>8.3f}"
        )
    lines.append(sep)
    return "\n".join(lines)


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.skip_mode(mode='debug',
                       reason='performance benchmarks are not meaningful in debug mode')
# @pytest.mark.parametrize("num_nodes,tablets_enabled", [
#     (1, False),
#     (3, False),
#     (1, True),
#     (3, True),
# ], ids=["1n_vnodes", "3n_vnodes", "1n_tablets", "3n_tablets"])
@pytest.mark.parametrize("num_nodes,tablets_enabled", [
    (3, False),
    (3, True),
], ids=["3n_vnodes", "3n_tablets"])
async def test_lwt_benchmark(
    manager: ManagerClient, num_nodes: int, tablets_enabled: bool,
):
    """Benchmark LWT throughput: VNodes vs Tablets.

    Each parametrized instance starts its own cluster, creates three tables
    (light, heavy_read, heavy_write), pre-populates data, then runs every
    workload scenario in both unthrottled and throttled modes.

    Collects ops/errors/wall_clock and latency percentiles (avg/p50/p95/p99)
    from server-side Prometheus histograms (delta between before/after snapshots).
    Results are logged and saved to /tmp/lwt_benchmark_<config>.json.
    """
    cluster_label = f"{num_nodes}n_{'tablets' if tablets_enabled else 'vnodes'}"
    logger.info("=== LWT Benchmark start: %s ===", cluster_label)

    if tablets_enabled:
        cfg = {"enable_tablets": True}
    else:
        cfg = {"tablets_mode_for_new_keyspaces": "disabled"}

    # Pin each Scylla node to separate P-core CPUs for stable benchmarking.
    # Each pair uses different physical cores (no HT sharing within a node).
    # Layout (from lscpu): CPUs 1,3 = 4800 MHz P-cores; 0,6,8,10 = 4500 MHz P-cores.
    # NOTE: this needs to be tuned for EACH machine
    per_node_cpusets = ["1,3", "0,6", "8,10"]

    servers = []
    for i in range(num_nodes):
        pf = None
        if tablets_enabled and num_nodes > 1:
            pf = {"dc": "dc1", "rack": f"rack{i + 1}"}
        cpuset_args = ['--cpuset', per_node_cpusets[i]] if i < len(per_node_cpusets) else []
        srv = await manager.server_add(
            config=cfg,
            property_file=pf,
            cmdline=cpuset_args,
        )
        servers.append(srv)

    cql = manager.get_cql()
    rf = num_nodes
    ks_opts = (
        f"WITH replication = {{'class': 'NetworkTopologyStrategy', "
        f"'replication_factor': {rf}}}"
    )
    if tablets_enabled:
        ks_opts += " AND tablets = {'enabled': true}"
    else:
        ks_opts += " AND tablets = {'enabled': false}"

    ks = await create_new_test_keyspace(cql, ks_opts)

    light_tbl = f"{ks}.lwt_light"
    heavy_read_tbl = f"{ks}.lwt_heavy_read"
    heavy_write_tbl = f"{ks}.lwt_heavy_write"

    await cql.run_async(
        f"CREATE TABLE {light_tbl} ("
        f"  pk int PRIMARY KEY,"
        f"  v int,"
        f"  payload text"
        f")"
    )
    await cql.run_async(
        f"CREATE TABLE {heavy_read_tbl} ("
        f"  pk int PRIMARY KEY,"
        f"  v int,"
        f"  d1 text, d2 text, d3 text, d4 text, d5 text"
        f")"
    )
    await cql.run_async(
        f"CREATE TABLE {heavy_write_tbl} ("
        f"  pk int PRIMARY KEY,"
        f"  v1 int, v2 int, v3 int, v4 int, v5 int,"
        f"  payload text"
        f")"
    )
    logger.info(
        "Created keyspace=%s tables=[%s, %s, %s] rf=%d tablets=%s",
        ks, light_tbl, heavy_read_tbl, heavy_write_tbl, rf, tablets_enabled,
    )

    # populate data
    light_payload = "x" * LIGHT_PAYLOAD_SIZE
    heavy_col_payload = "x" * HEAVY_COL_PAYLOAD_SIZE
    heavy_write_payload = "x" * LIGHT_PAYLOAD_SIZE  # payload column for write_heavy

    await _populate_light(cql, light_tbl, NUM_KEYS, light_payload)
    await _populate_heavy_read(cql, heavy_read_tbl, NUM_KEYS, heavy_col_payload)
    await _populate_heavy_write(cql, heavy_write_tbl, NUM_KEYS, heavy_write_payload)

    # prepare operations
    ops = {
        "write_light": {
            "unthrottled": _make_write_light(cql, light_tbl, light_payload, WRITE_KEY_BASE_UNTHROTTLED),
            "throttled": _make_write_light(cql, light_tbl, light_payload, WRITE_KEY_BASE_THROTTLED),
        },
        "write_heavy": {
            "unthrottled": _make_write_heavy(cql, heavy_write_tbl, heavy_write_payload),
            "throttled": _make_write_heavy(cql, heavy_write_tbl, heavy_write_payload),
        },
        "read_light": {
            "unthrottled": _make_read_light(cql, light_tbl),
            "throttled": _make_read_light(cql, light_tbl),
        },
        "read_heavy": {
            "unthrottled": _make_read_heavy(cql, heavy_read_tbl),
            "throttled": _make_read_heavy(cql, heavy_read_tbl),
        },
        "mixed_light": {
            "unthrottled": _make_mixed_light(cql, light_tbl, light_payload, MIXED_LIGHT_WRITE_KEY_BASE_UNTHROTTLED),
            "throttled": _make_mixed_light(cql, light_tbl, light_payload, MIXED_LIGHT_WRITE_KEY_BASE_THROTTLED),
        },
        "mixed_heavy": {
            "unthrottled": _make_mixed_heavy(cql, heavy_write_tbl, heavy_read_tbl, heavy_write_payload),
            "throttled": _make_mixed_heavy(cql, heavy_write_tbl, heavy_read_tbl, heavy_write_payload),
        },
    }

    # run benchmarks
    all_results: List[BenchmarkMetrics] = []

    # Map scenario prefix to which latency direction(s) to use
    _LATENCY_DIRECTIONS = {
        "write": ["write"],
        "read":  ["read"],
        "mixed": ["read", "write"],
    }

    for scenario in SCENARIOS:
        # for mode in ("unthrottled", "throttled"):
        for mode in ("unthrottled",):
            logger.info(
                "[%s] %s / %s — concurrency=%d warmup=%ds duration=%ds",
                cluster_label, scenario, mode,
                BENCH_CONCURRENCY, BENCH_WARMUP, BENCH_DURATION,
            )

            # Mutable container for metrics snapshots captured by callbacks
            snaps = {}

            async def _snap_before():
                snaps["before"] = await snapshot_latency_metrics(
                    manager, servers)

            async def _snap_after():
                snaps["after"] = await snapshot_latency_metrics(
                    manager, servers)

            raw = await _run_benchmark(
                cql=cql,
                op_fn=ops[scenario][mode],
                concurrency=BENCH_CONCURRENCY,
                duration_s=BENCH_DURATION,
                warmup_s=BENCH_WARMUP,
                on_measurement_start=_snap_before,
                on_measurement_end=_snap_after,
            )

            # compute latency from Prometheus histogram deltas
            scenario_prefix = scenario.split("_")[0]   # "write" / "read" / "mixed"
            directions = _LATENCY_DIRECTIONS[scenario_prefix]

            latency_stats_list = []
            for d in directions:
                ls = compute_latency_from_snapshots(
                    snaps["before"], snaps["after"], d)
                latency_stats_list.append(ls)

            # For mixed scenarios, pick the direction with more observations
            # to get the most representative latency (both are also logged).
            best = max(latency_stats_list, key=lambda s: s.total_count)

            m = BenchmarkMetrics.make(
                cluster_config=cluster_label,
                scenario=scenario,
                mode=mode,
                nodes=num_nodes,
                tablets_enabled=tablets_enabled,
                rf=rf,
            )
            m.wall_clock_sec = raw["wall_clock_sec"]
            m.attempted_ops = raw["attempted_ops"]
            m.successful_ops = raw["successful_ops"]
            m.applied_ops = raw["applied_ops"]
            m.not_applied_ops = raw["not_applied_ops"]
            m.errors = raw["errors"]
            m.achieved_ops_sec = raw["achieved_ops_sec"]
            m.applied_rate = (
                m.applied_ops / m.successful_ops if m.successful_ops > 0 else 0.0
            )

            # Fill latency percentiles from server-side histograms
            m.avg_latency_ms = best.avg_latency_ms
            m.p50_latency_ms = best.p50_latency_ms
            m.p95_latency_ms = best.p95_latency_ms
            m.p99_latency_ms = best.p99_latency_ms

            all_results.append(m)

            logger.info(
                "  => ok=%d applied=%d not_applied=%d err=%d ops/s=%.1f "
                "wall=%.1fs applied_rate=%.1f%% "
                "avg=%.3fms p50=%.3fms p95=%.3fms p99=%.3fms",
                m.successful_ops, m.applied_ops, m.not_applied_ops, m.errors,
                m.achieved_ops_sec, m.wall_clock_sec, m.applied_rate * 100,
                m.avg_latency_ms, m.p50_latency_ms,
                m.p95_latency_ms, m.p99_latency_ms,
            )

    logger.info(
        "\n=== Results: %s ===\n%s", cluster_label, _fmt_summary(all_results),
    )

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmark_results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"lwt_benchmark_{cluster_label}_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(
            {
                "cluster": cluster_label,
                "config": {
                    "bench_duration_s": BENCH_DURATION,
                    "warmup_s": BENCH_WARMUP,
                    "concurrency": BENCH_CONCURRENCY,
                    "num_keys": NUM_KEYS,
                    "target_rate_ops_sec": TARGET_OPS,
                    "light_payload_size": LIGHT_PAYLOAD_SIZE,
                    "heavy_col_payload_size": HEAVY_COL_PAYLOAD_SIZE,
                },
                "results": [asdict(m) for m in all_results],
            },
            f,
            indent=2,
        )
    logger.info("Results saved to %s", results_file)
