#
# Copyright (C) 2026-present ScyllaDB
#
# SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
#
import os
import glob
import json
import pytest
import asyncio
import logging
import subprocess
from collections import defaultdict
from cassandra.query import SimpleStatement, ConsistencyLevel

from test.pylib.tablets import get_tablet_count, get_all_tablet_replicas
from test.pylib.internal_types import ServerInfo
from test.pylib.manager_client import ManagerClient
from test.cluster.util import new_test_keyspace, reconnect_driver

logger = logging.getLogger(__name__)

def calculate_powof2_tokens(num_nodes: int, tokens_per_node: int) -> dict[int, list[int]]:
    exp = 0
    while 2**exp < num_nodes * tokens_per_node:
        exp += 1
    n = 2**exp
    new_tokens_combined = [int(i * 2**64 // n - 2**63 - 1) for i in range(1, n+1)]
    calculated_new_tokens = defaultdict(list)
    new_server_id = 1
    for i, t in enumerate(new_tokens_combined):
        calculated_new_tokens[new_server_id  + (i % num_nodes)].append(t)
    for s, tokens in calculated_new_tokens.items():
        calculated_new_tokens[s] = sorted(tokens)
    logger.debug(f"{calculated_new_tokens=}")
    return calculated_new_tokens


def get_sstable_token_ranges(scylla_path: str, scylla_yaml: str, sstable_data_files: list[str]) -> list[tuple[int, int]]:
    """Run 'scylla sstable dump-scylla-metadata' and return a list of (first_token, last_token) per SSTable.

    Extracts the first and last token from the sharding metadata in the Scylla.db component.

    Args:
        scylla_path: Path to the scylla executable.
        scylla_yaml: Path to the scylla.yaml config file.
        sstable_data_files: List of SSTable Data.db file paths.

    Returns:
        A list of (first_token, last_token) tuples, one per SSTable.
    """
    try:
        result = subprocess.check_output(
            [scylla_path, "sstable", "dump-scylla-metadata",
             "--scylla-yaml-file", scylla_yaml,
             "--sstables"] + sstable_data_files,
            stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logger.error(f"scylla sstable dump-scylla-metadata failed with exit code {e.returncode}")
        logger.error(f"stdout: {e.output.decode('utf-8', 'ignore')}")
        logger.error(f"stderr: {e.stderr.decode('utf-8', 'ignore')}")
        raise
    metadata = json.loads(result.decode('utf-8', 'ignore'))
    ranges = []
    for sstable_name, info in metadata["sstables"].items():
        sharding = info["sharding"]
        assert not sharding[0]["left"]["exclusive"], f"SSTable {sstable_name}: Expected the left bound of the first sharding range to be inclusive"
        assert not sharding[-1]["right"]["exclusive"], f"SSTable {sstable_name}: Expected the right bound of the last sharding range to be inclusive"
        first_token = int(sharding[0]["left"]["token"])
        last_token = int(sharding[-1]["right"]["token"])
        ranges.append((first_token, last_token))
    return ranges


def sstable_range_within_vnode(first_token: int, last_token: int, vnode_boundaries: list[int]) -> bool:
    """Check whether an SSTable's token range falls entirely within a single vnode range.

    Args:
        first_token: The first token in the SSTable.
        last_token: The last token in the SSTable.
        vnode_boundaries: Sorted list of vnode token boundaries.

    Returns:
        True if both first_token and last_token fall within the same vnode range.
    """
    def find_owning_vnode(token: int) -> int:
        """Return the index of the vnode that owns this token."""
        for i, boundary in enumerate(vnode_boundaries):
            if token <= boundary:
                return i
        # Token is above the last boundary, wraps to vnode 0
        return 0

    return find_owning_vnode(first_token) == find_owning_vnode(last_token)


async def verify_data_integrity(cql, ks, table, num_keys, cl=ConsistencyLevel.QUORUM):
    stmt = SimpleStatement(f"SELECT * FROM {ks}.{table}")
    stmt.consistency_level = cl
    rows = await cql.run_async(stmt)
    data = {r.pk: r.c for r in rows}
    expected = {k: k for k in range(num_keys)}
    if data != expected:
        missing = expected.keys() - data.keys()
        extra = data.keys() - expected.keys()
        wrong = {k: (data[k], expected[k]) for k in data.keys() & expected.keys() if data[k] != expected[k]}
        assert False, f"Data mismatch: missing keys {missing}, extra keys {extra}, wrong values {wrong}"


async def verify_migration_status(manager: ManagerClient, server: ServerInfo,
                                  ks: str, expected_status: str,
                                  expected_node_statuses: dict[str, tuple[str, str]],
                                  retries: int = 0, retry_interval: float = 0):
    async def _check():
        status = await manager.api.get_vnode_tablet_migration_status(server.ip_addr, ks)
        actual_node_statuses = {n['host_id']: (n['current_mode'], n['intended_mode']) for n in status['nodes']}
        assert status['status'] == expected_status, f"Expected migration status '{expected_status}', got '{status['status']}'"
        assert actual_node_statuses == expected_node_statuses, f"Expected node statuses {expected_node_statuses}, got {actual_node_statuses}"

    for attempt in range(retries + 1):
        try:
            await _check()
            return
        except AssertionError:
            if attempt < retries and retry_interval > 0:
                await asyncio.sleep(retry_interval)
            else:
                raise


@pytest.mark.asyncio
async def test_migration(manager: ManagerClient):
    """Verify vnodes-to-tablets migration for a single table on a single-node cluster.

    Steps:
    1. Start a single node with multiple shards and power-of-2 aligned vnodes.
    2. Create a vnode table and inject data.
    3. Start the migration by creating a tablet map for the table.
       - Verify that the tablet map was created and tablet tokens match vnode tokens.
    4. Mark the node for upgrade on next restart.
    5. Restart the node (triggers resharding on vnode boundaries).
       - Verify that the new SSTables are properly segregated within vnode boundaries.
       - Verify data integrity.
    6. Finalize the migration.
       - Verify that the keyspace schema has tablets enabled.
    """
    num_shards = 3
    tokens_per_node = 16
    num_keys = 5000

    logger.info(f"Starting a node with {num_shards} shards and {tokens_per_node} power-of-2 aligned tokens")
    tokens = calculate_powof2_tokens(num_nodes=1, tokens_per_node=tokens_per_node)
    token_list = tokens[1]  # server_id 1
    initial_token = ','.join([str(t) for t in token_list])
    servers = (await manager.servers_add(1, cmdline=['--smp', str(num_shards), '--initial-token', initial_token, '--logger-log-level', 'compaction=debug']))
    server = servers[0]
    host_id = await manager.get_host_id(server.server_id)

    cql, _ = await manager.get_ready_cql(servers)

    logger.info("Creating keyspace and table with vnodes")
    async with new_test_keyspace(manager, f"WITH replication = {{'class': 'NetworkTopologyStrategy', 'replication_factor': 1}} AND tablets = {{'enabled': false}}") as ks:
        # Create the table with minor compaction disabled to ensure that Scylla
        # won't delete SSTables while we're checking them.
        await cql.run_async(f"CREATE TABLE {ks}.test (pk int PRIMARY KEY, c int) WITH compaction = {{'class': 'IncrementalCompactionStrategy', 'enabled': false}}")

        logger.info("Populating table in batches, flushing after each batch to produce multiple SSTables")
        stmt = cql.prepare(f"INSERT INTO {ks}.test (pk, c) VALUES (?, ?)")
        num_flushes = 5
        keys_per_sstable = num_keys // num_flushes
        for batch_start in range(0, num_keys, keys_per_sstable):
            batch_end = min(batch_start + keys_per_sstable, num_keys)
            await asyncio.gather(*(cql.run_async(stmt, [k, k]) for k in range(batch_start, batch_end)))
            await manager.api.keyspace_flush(server.ip_addr, ks, "test")

        logger.info("Collecting pre-migration SSTable info")
        node_workdir = await manager.server_get_workdir(server.server_id)
        scylla_path = await manager.server_get_exe(server.server_id)
        scylla_yaml = os.path.join(node_workdir, "conf", "scylla.yaml")
        table_data_dir = glob.glob(os.path.join(node_workdir, "data", ks, "test-*"))[0]
        pre_migration_sstables = glob.glob(os.path.join(table_data_dir, "*-Data.db"))
        logger.info(f"Pre-migration SSTable count: {len(pre_migration_sstables)}")
        assert len(pre_migration_sstables) == num_shards * num_flushes

        logger.info("Verifying that at least one pre-migration SSTable spans multiple vnode ranges (ensures that resharding will have work to do)")
        vnode_boundaries = sorted(token_list)
        pre_migration_ranges = get_sstable_token_ranges(scylla_path, scylla_yaml, pre_migration_sstables)
        cross_vnode_count = sum(1 for first, last in pre_migration_ranges
                                if not sstable_range_within_vnode(first, last, vnode_boundaries))
        logger.info(f"Pre-migration: {cross_vnode_count}/{len(pre_migration_ranges)} SSTables span multiple vnodes")
        assert cross_vnode_count >= 1, \
            "Expected at least one pre-migration SSTable to span multiple vnode ranges, but none was found. The test is malformed."

        logger.info("Verifying migration status before starting migration")
        await verify_migration_status(manager, server, ks, expected_status='vnodes', expected_node_statuses={})

        logger.info("Starting vnodes-to-tablets migration (creating a tablet map)")
        await manager.api.create_vnode_tablet_migration(server.ip_addr, ks)

        logger.info("Verifying migration status after creating tablet map")
        await verify_migration_status(manager, server, ks,
            expected_status='migrating_to_tablets',
            expected_node_statuses={host_id: ('vnodes', 'vnodes')})

        logger.info("Verifying that the tablet map was created")
        tablet_count = await get_tablet_count(manager, server, ks, 'test')
        assert tablet_count == tokens_per_node, \
            f"Expected {tokens_per_node} tablet(s), got {tablet_count}"

        tablet_replicas = await get_all_tablet_replicas(manager, server, ks, 'test')
        assert len(tablet_replicas) == tokens_per_node, \
            f"Expected {tokens_per_node} tablet replica entries, got {len(tablet_replicas)}"

        logger.info("Verifying that tablet tokens match vnode tokens")
        tablet_tokens = sorted([tr.last_token for tr in tablet_replicas])
        assert tablet_tokens == vnode_boundaries, \
            f"Tablet tokens {tablet_tokens} do not match vnode tokens {vnode_boundaries}"

        logger.info("Verifying data integrity after building tablet map and before resharding")
        await verify_data_integrity(cql, ks, "test", num_keys)

        logger.info("Marking node for tablets migration")
        await manager.api.upgrade_node_to_tablets(server.ip_addr)

        logger.info("Verifying migration status after marking node for upgrade")
        await verify_migration_status(manager, server, ks,
            expected_status='migrating_to_tablets',
            expected_node_statuses={host_id: ('vnodes', 'tablets')})

        logger.info("Verifying data integrity after marking the node for upgrade and before resharding (ensures that the node is still using the vnode-based ERM)")
        await verify_data_integrity(cql, ks, "test", num_keys)

        logger.info("Restarting the node to trigger resharding")
        await manager.server_restart(server.server_id)
        await reconnect_driver(manager)
        cql, _ = await manager.get_ready_cql(servers)

        # After restart, resharding should have produced new SSTables
        post_restart_sstables = glob.glob(os.path.join(table_data_dir, "*-Data.db"))
        logger.info(f"Post-restart SSTable count: {len(post_restart_sstables)}")

        post_restart_ranges = get_sstable_token_ranges(scylla_path, scylla_yaml, post_restart_sstables)
        logger.info(f"Post-restart SSTable token ranges:")
        for i, (first, last) in enumerate(post_restart_ranges):
            within = sstable_range_within_vnode(first, last, vnode_boundaries)
            logger.info(f"  SSTable {i}: tokens [{first}, {last}], within single vnode: {within}")

        logger.info("Verifying that every post-restart SSTable falls within a single vnode range")
        for i, (first, last) in enumerate(post_restart_ranges):
            assert sstable_range_within_vnode(first, last, vnode_boundaries), \
                f"Post-restart SSTable {i} with token range [{first}, {last}] " \
                f"spans multiple vnode ranges (boundaries: {vnode_boundaries})"

        logger.info("Verifying data integrity after restart")
        await verify_data_integrity(cql, ks, "test", num_keys)

        logger.info("Verifying migration status after node restart (node should now use tablets)")
        await verify_migration_status(manager, server, ks,
            expected_status='migrating_to_tablets',
            expected_node_statuses={host_id: ('tablets', 'tablets')})

        logger.info("Finalizing tablets migration")
        await manager.api.finalize_vnode_tablet_migration(server.ip_addr, ks)

        logger.info("Verifying that the keyspace schema has tablets enabled")
        res = await cql.run_async(f"SELECT * FROM system_schema.scylla_keyspaces WHERE keyspace_name = '{ks}'")
        assert len(res) == 1 and res[0].initial_tablets is not None, "keyspace is still using vnodes after migration finalization"

        logger.info("Verifying migration status after finalization")
        await verify_migration_status(manager, server, ks, expected_status='tablets', expected_node_statuses={})