#
# Copyright (C) 2026-present ScyllaDB
#
# SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
#

"""SCRATCH -- not part of the series, not committed.

Three short runs to find out whether the harness actually works end to end,
and in particular whether the parts no committed test exercises do:
``tolerate_timeouts``, the indeterminate (unpaired) write, and a disruptor
other than the schema changer.

Run (inside dbuild):

    ./test.py --mode dev test/cluster/strong_consistency/test_harness_smoke.py
    ./test.py --mode dev test/cluster/strong_consistency/test_harness_smoke.py::test_survives_a_node_restart

What each one is looking for:

  * baseline -- the pipeline works at all: cluster, workload, history, checker.
    Uses the default policy, so *any* driver error fails it.
  * node restart -- the first real use of tolerate_timeouts.  Expect this one
    to be the informative failure: a restart can surface NoHostAvailable, which
    no policy currently recognises (tolerate_reset_windows only accepts it as a
    stale table UUID inside a reset window).  If that happens, the answer is a
    decision, not a blind widening: NoHostAvailable means the driver found no
    host to send to, which is usually "not applied" (Outcome.FAIL) -- but not
    guaranteed, because it can also wrap per-host failures of a request that
    did leave.  Pick FAIL only if we are sure; otherwise UNKNOWN.
  * tablet migration -- keeps the default policy on purpose: migration is meant
    to be invisible to clients, so any error here is a finding, not noise.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from test.cluster.strong_consistency.config import boot_sc_cluster, sc_keyspace_opts
from test.cluster.strong_consistency.outcomes import (
    first_match,
    tolerate_reset_windows,
    tolerate_timeouts,
)
from test.cluster.strong_consistency.workload import (
    RegisterWorkload,
    check_linearizable,
    run_workload,
)
from test.cluster.util import new_test_keyspace
from test.pylib.scylla_cluster_manager import ScyllaClusterManager
from test.pylib.tablets import get_tablet_replicas

logger = logging.getLogger(__name__)

NUM_NODES = 4                      # RF=3 plus one free host, so a tablet can move
REPLICATION_FACTOR = 3
NUM_TABLETS = 8
KEYSPACE_OPTS = sc_keyspace_opts(
    replication_factor=REPLICATION_FACTOR, initial_tablets=NUM_TABLETS)

TABLE = "main"
SCHEMA = "(pk int PRIMARY KEY, c int)"

NUM_KEYS = 50
NUM_WRITERS = 2
NUM_READERS = 2


async def _prepared_workload(ks: str, cql, **kwargs):
    workload = RegisterWorkload(
        ks=ks, table_name=TABLE, num_keys=NUM_KEYS,
        num_writers=NUM_WRITERS, num_readers=NUM_READERS, **kwargs)
    await cql.run_async(f"CREATE TABLE {workload.fqtn} {SCHEMA}")
    workload.prepare(cql)
    return workload


async def _finish(workload, task_errors, tmp_path, label, *, min_writes=50, min_reads=50):
    logger.info("Stats: %s", workload.stats_line())
    assert not task_errors, (
        f"task(s) failed (seed={workload.seed}): {task_errors}")
    workload.assert_progress(min_writes=min_writes, min_reads=min_reads)
    await check_linearizable(workload, output_dir=tmp_path / label)


@pytest.mark.asyncio
@pytest.mark.no_parallel
async def test_workload_alone_is_linearizable(manager: ScyllaClusterManager, tmp_path):
    """No disruptor at all: does the plain pipeline come out linearizable?"""
    _servers, cql = await boot_sc_cluster(manager, NUM_NODES)

    async with new_test_keyspace(manager, KEYSPACE_OPTS) as ks:
        workload = await _prepared_workload(ks, cql)
        errors = await run_workload(workload, cql, 15)
        await _finish(workload, errors, tmp_path, "baseline")


class _Killer:
    """Stops and starts one node at a time; counts what it managed to do."""

    def __init__(self, manager, servers, workload):
        self.manager, self.servers, self.workload = manager, servers, workload
        self.restarts = 0

    async def run(self) -> None:
        rng = self.workload.rng_for("node-killer")
        while not self.workload.stop_event.is_set():
            await asyncio.sleep(rng.uniform(4.0, 7.0))
            if self.workload.stop_event.is_set():
                break
            victim = rng.choice(self.servers)
            logger.info("Stopping %s", victim.server_id)
            await self.manager.server_stop_gracefully(victim.server_id)
            try:
                await asyncio.sleep(rng.uniform(2.0, 4.0))
            finally:
                # The node must come back even if the run is being torn down,
                # or everything after this point runs against a smaller cluster.
                logger.info("Starting %s", victim.server_id)
                await self.manager.server_start(victim.server_id)
            self.restarts += 1
        logger.info("Node killer finished: %d restarts", self.restarts)


@pytest.mark.asyncio
@pytest.mark.no_parallel
async def test_survives_a_node_restart(manager: ScyllaClusterManager, tmp_path):
    """One node goes down and comes back while the workload runs.

    The point is tolerate_timeouts: a write that times out must be recorded as
    indeterminate (no return event), and the checker must still accept the
    history.  A non-zero 'pending' in the stats line is the sign that the path
    was actually taken -- if it stays 0, this run proved nothing about it.
    """
    servers, cql = await boot_sc_cluster(manager, NUM_NODES)

    async with new_test_keyspace(manager, KEYSPACE_OPTS) as ks:
        workload = await _prepared_workload(
            ks, cql,
            exception_policy=first_match(tolerate_reset_windows, tolerate_timeouts))
        killer = _Killer(manager, servers, workload)

        errors = await run_workload(workload, cql, 40, disruptors=[
            ("node-killer", killer.run),
        ])

        assert killer.restarts >= 1, "the disruptor never managed to restart a node"
        logger.info("pending (indeterminate) operations: %d", workload.stats().pending)
        await _finish(workload, errors, tmp_path, "node-restart", min_writes=20, min_reads=20)


class _TabletMover:
    """Moves one tablet replica at a time to a host that does not hold it."""

    def __init__(self, manager, servers, workload):
        self.manager, self.servers, self.workload = manager, servers, workload
        self.moves = 0

    async def run(self) -> None:
        rng = self.workload.rng_for("tablet-mover")
        hosts = [await self.manager.get_host_id(s.server_id) for s in self.servers]
        while not self.workload.stop_event.is_set():
            await asyncio.sleep(rng.uniform(1.0, 3.0))
            if self.workload.stop_event.is_set():
                break
            token = rng.randrange(-2**63, 2**63)
            replicas = await get_tablet_replicas(
                self.manager, self.servers[0], self.workload.ks, TABLE, token)
            free = [h for h in hosts if h not in {r[0] for r in replicas}]
            if not replicas or not free:
                continue
            src_host, src_shard = rng.choice(replicas)
            dst_host = rng.choice(free)
            logger.info("Moving tablet of token %d: %s -> %s", token, src_host, dst_host)
            await self.manager.api.move_tablet(
                self.servers[0].ip_addr, self.workload.ks, TABLE,
                src_host, src_shard, dst_host, 0, token)
            self.moves += 1
        logger.info("Tablet mover finished: %d moves", self.moves)


@pytest.mark.asyncio
@pytest.mark.no_parallel
async def test_tablet_migration_is_invisible(manager: ScyllaClusterManager, tmp_path):
    """Tablets move under the workload; the default policy tolerates nothing.

    Migration is supposed to be transparent to clients, so this run keeps
    tolerate_reset_windows: any driver error becomes a failed test rather than
    a tolerated blip.

    sc_coordinator=trace so that a failing read leaves its path in the node
    log: the step it was waiting on and the exception filter_error turned into
    a timeout.
    """
    servers, cql = await boot_sc_cluster(
        manager, NUM_NODES, cmdline=["--logger-log-level", "sc_coordinator=trace"])

    async with new_test_keyspace(manager, KEYSPACE_OPTS) as ks:
        workload = await _prepared_workload(ks, cql)
        mover = _TabletMover(manager, servers, workload)

        errors = await run_workload(workload, cql, 30, disruptors=[
            ("tablet-mover", mover.run),
        ])

        assert mover.moves >= 1, "the disruptor never managed to move a tablet"
        await _finish(workload, errors, tmp_path, "tablet-migration")
