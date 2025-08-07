import asyncio
import contextlib
import json
import logging
import random
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict

import pytest
from cassandra import ConsistencyLevel
from cassandra import WriteTimeout, OperationTimedOut
from cassandra.protocol import WriteFailure
from cassandra.query import SimpleStatement, PreparedStatement
from test.cluster.conftest import skip_mode
from test.cluster.util import new_test_keyspace
from test.pylib.manager_client import ManagerClient
from test.pylib.tablets import get_tablet_replicas
from test.pylib.tablets import get_all_tablet_replicas

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Arbitrary constants for the test
WORKERS = 20
BACKOFF_BASE = 0.02
WORKLOAD_SEC = 30
POST_MIGRATION_SEC = 5
NUM_KEYS = 50


@dataclass
class LogEntry:
    ts: int
    pid: int
    phase: str
    key: int
    col: int
    applied: Optional[bool]
    prev: Optional[int]
    new: Optional[int]
    err: Optional[str]
    operation_id: int



class SchemaManager:
    def __init__(self, manager: ManagerClient, ks: str, tbl: str, num_workers: int, num_keys: int = NUM_KEYS):
        self.manager = manager
        self.ks = ks
        self.tbl = tbl
        self.num_workers = num_workers
        self.num_keys = num_keys
        self.columns = [f"s{i}" for i in range(self.num_workers)]
        self.select_cols = ", ".join(self.columns)
        self.cql = manager.get_cql()
        self.pks: List[int] = list(range(1, num_keys + 1))

    async def create_schema(self):
        """Create table with multiple counter columns"""
        cols_def = ", ".join(f"s{i} int" for i in range(self.num_workers))
        await self.cql.run_async(
            f"CREATE TABLE {self.ks}.{self.tbl} (pk int PRIMARY KEY, {cols_def})"
        )
        logger.info(f"Created table {self.ks}.{self.tbl} with {self.num_workers} columns")

    async def initialize_rows(self):
        """Initialize the test row with all columns set to 0"""
        zeros = ", ".join("0" for _ in range(self.num_workers))
        for pk in self.pks:
            await self.cql.run_async(
                f"INSERT INTO {self.ks}.{self.tbl} "
                f"(pk, {self.select_cols}) VALUES ({pk}, {zeros})"
            )


class ResultTracker:
    """Tracks operation results and maintains consistency counters"""

    def __init__(self, num_workers: int, num_keys: int):
        self.num_workers = num_workers
        self.num_keys = num_keys
        self.pks: List[int] = list(range(1, num_keys + 1))
        self.success_counts: Dict[int, List[int]] = {
            pk: [0] * self.num_workers for pk in self.pks
        }
        self.history: List[dict] = []

    def increment_success(self, pk: int, worker_id: int):
        """Increment success count for a specific worker and primary key"""
        self.success_counts[pk][worker_id] += 1

    def get_success_count(self, pk: int, worker_id: int) -> int:
        """Get current success count for a worker and primary key"""
        return self.success_counts[pk][worker_id]

    def log_entry(self, worker_id: int, phase: str, prev_val: int = None,
                  new_val: int = None, applied: bool = None, error: str = None,
                  pk: int = None, operation_id: int = None):
        """Log an operation entry"""
        entry = LogEntry(
            operation_id=operation_id,
            ts=time.time_ns(),
            pid=worker_id,
            phase=phase,
            key=pk,
            col=worker_id,
            applied=applied,
            prev=prev_val,
            new=new_val,
            err=error
        )
        self.history.append(asdict(entry))

    def save_history(self, filename: str = "lwt_history.json"):
        """Save operation history to file"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Saved {len(self.history)} history entries to {filename}")


class WorkloadManager:
    def __init__(self, manager: ManagerClient, schema: SchemaManager, tracker: ResultTracker):
        self.manager = manager
        self.schema = schema
        self.tracker = tracker
        self.cql = manager.get_cql()
        self.workers: List[asyncio.Task] = []
        self.stop_event = asyncio.Event()

        # Prepared statements
        self.ps_select_row: Optional[PreparedStatement] = None
        self.ps_update: List[Optional[PreparedStatement]] = [None] * schema.num_workers

        # Worker configuration
        self.others_per_worker: List[List[int]] = [
            [col_index for col_index in range(schema.num_workers) if col_index != worker_index]
            for worker_index in range(schema.num_workers)
        ]
        self.rngs = [random.Random(i) for i in range(schema.num_workers)]

    def prepare_statements(self):
        """Prepare CQL statements for row selection and updates"""
        self.ps_select_row = self.cql.prepare(
            f"SELECT {self.schema.select_cols} FROM {self.schema.ks}.{self.schema.tbl} WHERE pk = ?"
        )

        for i in range(self.schema.num_workers):
            others = self.others_per_worker[i]
            cond = " AND ".join([f"s{j} >= ?" for j in others] + [f"s{i} = ?"])
            query = (
                f"UPDATE {self.schema.ks}.{self.schema.tbl} SET s{i} = ? "
                f"WHERE pk = ? IF {cond}"
            )
            self.ps_update[i] = self.cql.prepare(query)

    async def _worker(self, worker_id: int):
        """Worker coroutine that performs LWT operations"""
        rng = self.rngs[worker_id]
        operation_id = 0

        while not self.stop_event.is_set():
            pk = rng.choice(self.schema.pks)

            # Read current values
            verify_query = self.ps_select_row.bind([pk])
            verify_query.consistency_level = ConsistencyLevel.LOCAL_QUORUM
            rows = await self.cql.run_async(verify_query)

            row = rows[0]
            prev_val = getattr(row, f"s{worker_id}")
            new_val = self.tracker.get_success_count(pk, worker_id) + 1

            # Verify consistency before update
            assert prev_val == self.tracker.get_success_count(pk, worker_id)

            # Prepare conditional update
            others = self.others_per_worker[worker_id]
            if_vals = [getattr(row, f"s{col_idx}") for col_idx in others] + [prev_val]
            params = [new_val, pk] + if_vals

            update = self.ps_update[worker_id].bind(params)
            update.consistency_level = ConsistencyLevel.LOCAL_QUORUM
            update.serial_consistency_level = ConsistencyLevel.LOCAL_SERIAL

            try:
                res = await self.cql.run_async(update)
                applied = bool(res and res[0].applied)
            except (WriteTimeout, OperationTimedOut) as e:
                # Handle timeout by checking if operation actually succeeded
                verify_after_cas_error = self.ps_select_row.bind([pk])
                verify_after_cas_error.consistency_level = ConsistencyLevel.LOCAL_SERIAL
                vrow = (await self.cql.run_async(verify_after_cas_error))[0]
                applied = (getattr(vrow, f"s{worker_id}") == new_val)

            if applied:
                self.tracker.increment_success(pk, worker_id)
            else:
                raise AssertionError(
                    f"Unexpected CAS non-apply without timeout: pk={pk} worker={worker_id} prev={prev_val}"
                )

            operation_id += 1
            if self.stop_event.is_set():
                break

    async def start_workers(self):
        """Start all worker tasks"""
        self.workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self.schema.num_workers)
        ]
        logger.info(f"Started {self.schema.num_workers} LWT workers")

    async def stop_workers(self):
        """Stop all worker tasks"""
        # self.stop_event.set()
        # await asyncio.gather(*self.workers, return_exceptions=True)
        # logger.info("All workers stopped")
        self.stop_event.set()
        results = await asyncio.gather(*self.workers, return_exceptions=True)
        errs = [e for e in results if isinstance(e, Exception)]
        assert not errs, f"worker errors: {errs}"
        logger.info("All workers stopped")

    async def verify_consistency(self):
        """Ensure every (pk, column) reflects the number of successful CAS writes."""
        mismatches = []
        for pk in self.schema.pks:
            stmt = SimpleStatement(
                f"SELECT {self.schema.select_cols} FROM {self.schema.ks}.{self.schema.tbl} WHERE pk = %s",
                consistency_level=ConsistencyLevel.LOCAL_QUORUM,
            )
            row = (await self.cql.run_async(stmt, [pk]))[0]
            for col_idx in range(self.schema.num_workers):
                actual = getattr(row, f"s{col_idx}")
                expected = self.tracker.get_success_count(pk, col_idx)
                if actual != expected:
                    mismatches.append(
                        f"pk={pk} s{col_idx}={actual}, expected {expected}"
                    )

        assert not mismatches, "Consistency violations: " + "; ".join(mismatches)
        total_ops = sum(sum(v) for v in self.tracker.success_counts.values())
        logger.info("Consistency verified – %d total successful CAS operations", total_ops)


class MultiColumnLWTTester_NEW:
    """
    Coordinates multi-column LWT testing using dedicated components.
    """
    def __init__(self, manager: ManagerClient, ks: str, tbl: str, num_workers: int = WORKERS, num_keys: int = NUM_KEYS):
        self.schema = SchemaManager(manager, ks, tbl, num_workers, num_keys)
        self.tracker = ResultTracker(num_workers, num_keys)
        self.workload = WorkloadManager(manager, self.schema, self.tracker)
        self.ks = ks
        self.tbl = tbl
        self.cql = manager.get_cql()
        self.success_counts = self.tracker.success_counts

    async def create_schema(self):
        """Create database schema"""
        await self.schema.create_schema()

    async def initialize_rows(self):
        """Initialize table rows"""
        await self.schema.initialize_rows()

    def prepare_statements(self):
        """Prepare CQL statements"""
        self.workload.prepare_statements()

    async def start_workers(self):
        """Start workload workers"""
        await self.workload.start_workers()

    async def stop_workers(self):
        """Stop workload workers"""
        await self.workload.stop_workers()

    async def verify_consistency(self):
        """Verify data consistency"""
        await self.workload.verify_consistency()

    def save_history(self, filename: str = "lwt_history.json"):
        """Save operation history"""
        self.tracker.save_history(filename)


async def get_non_replica_server(manager: ManagerClient, servers, replicas):
    replica_host_ids = {replica[0] for replica in replicas}
    for server in servers:
        host_id = await manager.get_host_id(server.server_id)
        if host_id not in replica_host_ids:
            return server
    raise ValueError("No non-replica servers found")


async def migrate_tablet(manager: ManagerClient, src_server, dst_server, ks: str, tbl: str, token: int, replicas):
    logger.info(f"Starting tablet migration to {dst_server.ip_addr}")

    dst_host_id = await manager.get_host_id(dst_server.server_id)
    src_replica = replicas[0]

    await manager.api.move_tablet(
        src_server.ip_addr, ks, tbl,
        src_replica[0], src_replica[1],
        dst_host_id, 0, token
    )

    current_replicas = await get_tablet_replicas(manager, src_server, ks, tbl, token)
    assert any(repl[0] == dst_host_id for repl in current_replicas), "Tablet migration failed"
    logger.info(f"Tablet migration completed to {dst_server.ip_addr}")


async def _host_map(manager, servers):
    ids = await asyncio.gather(*[manager.get_host_id(s.server_id) for s in servers])
    return {hid: srv for hid, srv in zip(ids, servers)}


async def _pick_non_replica_server(manager, servers, replica_host_ids):
    for s in servers:
        hid = await manager.get_host_id(s.server_id)
        if hid not in replica_host_ids:
            return s
    return None


async def continuous_tablet_migrations(manager: ManagerClient, servers, ks: str, tbl: str,
                                       duration_sec: int, pause_range=(0.5, 2.0)):

    logger.info("Starting continuous tablet migrations for %s seconds", duration_sec)
    start = time.time()
    migration_count = 0

    host_map = await _host_map(manager, servers)

    while time.time() - start < duration_sec:
        try:
            sample_pk = random.randint(1, NUM_KEYS)
            token = await get_token_for_pk(manager.get_cql(), ks, tbl, sample_pk)

            replicas = await get_tablet_replicas(manager, servers[0], ks, tbl, token)
            if not replicas:
                logger.warning("No replicas for token=%s, skipping", token)
                await asyncio.sleep(1.0); continue

            src_host_id, src_shard = random.choice(replicas)
            src_server = host_map.get(src_host_id)
            if not src_server:
                host_map = await _host_map(manager, servers)
                src_server = host_map.get(src_host_id)
                if not src_server:
                    # is it possible that src_server was not found?
                    await asyncio.sleep(1.0); continue

            replica_hids = {h for (h, _shard) in replicas}
            dst_server = await _pick_non_replica_server(manager, servers, replica_hids)
            if not dst_server:
                logger.info("No non-replica destination for token=%s, skipping", token)
                await asyncio.sleep(1.0); continue

            dst_hid = await manager.get_host_id(dst_server.server_id)
            await manager.api.move_tablet(src_server.ip_addr, ks, tbl,
                                          src_host_id, src_shard,
                                          dst_hid, 0, token)
            migration_count += 1
            logger.info("Completed migration #%d (token=%s -> %s)", migration_count,
                        token, dst_server.ip_addr)

            await asyncio.sleep(random.uniform(*pause_range))

        except Exception as e:
            logger.warning("Migration attempt failed: %s, continuing...", e)
            await asyncio.sleep(1.0)

    logger.info("Completed %d tablet migrations in %s seconds", migration_count, duration_sec)


async def get_token_for_pk(cql, ks: str, tbl: str, pk: int) -> int:
    stmt = SimpleStatement(
        f"SELECT token(pk) AS tk FROM {ks}.{tbl} WHERE pk = %s",
        consistency_level=ConsistencyLevel.ONE,
    )
    row = (await cql.run_async(stmt, [pk]))[0]
    return row.tk


@pytest.mark.asyncio
@skip_mode('release', 'error injections are not supported in release mode')
@skip_mode('debug', 'debug mode is too slow for this test')
async def test_multi_column_lwt_during_migration(manager: ManagerClient):
    """Test multi-column LWT pattern during continuous tablet migrations"""

    # Setup cluster
    cfg = {
        "tablets_mode_for_new_keyspaces": "enabled",
        "rf_rack_valid_keyspaces": False
    }

    property_files = [{"dc": "dc1", "rack": f"rack{(i % 2) + 1}"} for i in range(6)]
    servers = await manager.servers_add(6, config=cfg, property_file=property_files)

    for server in servers:
        await manager.api.disable_tablet_balancing(server.ip_addr)

    async with new_test_keyspace(
        manager,
        "WITH replication = {'class': 'NetworkTopologyStrategy', 'replication_factor': 2} AND tablets = {'initial': 5}"
    ) as ks:

        tester = MultiColumnLWTTester_NEW(manager, ks, "lwt_table")
        await tester.create_schema()
        await tester.initialize_rows()
        tester.prepare_statements()
        await tester.start_workers()

        try:
            # Run continuous tablet migrations concurrently with the LWT workload
            logger.info(f"Starting concurrent LWT workload and tablet migrations for {WORKLOAD_SEC} seconds")
            migration_task = asyncio.create_task(
                continuous_tablet_migrations(manager, servers, ks, tester.tbl, WORKLOAD_SEC)
            )
            await asyncio.wait_for(migration_task, timeout=WORKLOAD_SEC + 5)

        finally:
            await tester.stop_workers()

        await tester.verify_consistency()
        #tester.save_history()

        logger.info("Multi-column LWT during continuous migrations test completed successfully")
