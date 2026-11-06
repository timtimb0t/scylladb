#
# Copyright (C) 2026-present ScyllaDB
#
# SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
#

"""Network partitions between live nodes, cut with nft rules in test.py's private network namespace.

Until now a test could take a node away from its peers only by stopping it or freezing it with
SIGSTOP, and in both cases the node stops serving the test as well. Here the node keeps running
and keeps answering the test over CQL and the REST API: only its internode traffic is cut, as it
would be by a broken link between machines. See test/pylib/network.py for how the cut is made
without root.
"""

import asyncio
import logging
import time

import pytest
from cassandra import ConsistencyLevel, OperationTimedOut, Unavailable
from cassandra.cluster import NoHostAvailable
from cassandra.connection import ConnectionException
from cassandra.query import SimpleStatement

from test.cluster.util import new_test_keyspace, new_test_table
from test.pylib.network import NetworkPartition
from test.pylib.scylla_cluster_manager import ScyllaClusterManager
from test.pylib.util import wait_for, wait_for_cql_and_get_hosts

logger = logging.getLogger(__name__)


async def test_isolated_node_is_down_for_its_peers_only(manager: ScyllaClusterManager, network: NetworkPartition,
                                                        failure_detector_timeout: int) -> None:
    """A node cut off from its peers on the internode port is DOWN for them and they are DOWN for it,
    while the test still talks to all three. Writes at QUORUM go on, writes at ALL are unavailable, and
    everything recovers once the cut is removed.
    """
    config = {'failure_detector_timeout_in_ms': failure_detector_timeout}
    servers = await manager.servers_add(3, config=config, auto_rack_dc='dc1')
    cql, hosts = await manager.get_ready_cql(servers)
    node_a, node_b, node_c = servers
    host_a = hosts[0]

    async def down_endpoints(server) -> set[str]:
        return set(await manager.api.get_down_endpoints(server.ip_addr))

    async def wait_for_down(server, expected: set[str]) -> None:
        async def done():
            return True if await down_endpoints(server) == expected else None
        await wait_for(done, time.time() + 60, period=0.5)

    async with new_test_keyspace(manager, "WITH replication = {'class': 'NetworkTopologyStrategy', 'replication_factor': 3}") as ks:
        async with new_test_table(manager, ks, "pk int PRIMARY KEY, v int") as table:
            def insert(v: int, cl: ConsistencyLevel):
                return cql.run_async(SimpleStatement(f"INSERT INTO {table} (pk, v) VALUES (1, {v})", consistency_level=cl),
                                     host=host_a)

            await insert(1, ConsistencyLevel.ALL)

            logger.info(f"Cutting {node_c} off from {node_a} and {node_b} on the internode port")
            await network.isolate(node_c.ip_addr, [node_a.ip_addr, node_b.ip_addr])

            await wait_for_down(node_a, {node_c.ip_addr})
            await wait_for_down(node_b, {node_c.ip_addr})
            # The cut-off node is still there for the test: it answers over REST and sees its peers gone.
            await wait_for_down(node_c, {node_a.ip_addr, node_b.ip_addr})

            await insert(2, ConsistencyLevel.QUORUM)
            with pytest.raises((Unavailable, NoHostAvailable)):
                await insert(3, ConsistencyLevel.ALL)

            logger.info("Removing the cut")
            await network.heal()

            await wait_for_down(node_a, set())
            await wait_for_down(node_b, set())
            await wait_for_down(node_c, set())

            await insert(4, ConsistencyLevel.ALL)
            rows = await cql.run_async(SimpleStatement(f"SELECT v FROM {table} WHERE pk = 1", consistency_level=ConsistencyLevel.ALL),
                                       host=host_a)
            assert [r.v for r in rows] == [4]


async def test_node_cut_off_from_clients_only(manager: ScyllaClusterManager, network: NetworkPartition) -> None:
    """A node cut off from clients on the CQL ports stops taking writes from a driver pinned to it, with the
    writer failing at once rather than waiting for its request timeout. Its peers keep seeing it UP and the
    test keeps reaching it over the REST API: the node is healthy, only its clients lost it. The driver marks
    it down, brings it back once the cut is removed, and every write acknowledged before the cut is there.
    """
    servers = await manager.servers_add(3, auto_rack_dc='dc1')
    cql, hosts = await manager.get_ready_cql(servers)
    node_a, _, node_c = servers
    host_a, _, host_c = hosts

    async with new_test_keyspace(manager, "WITH replication = {'class': 'NetworkTopologyStrategy', 'replication_factor': 3}") as ks:
        async with new_test_table(manager, ks, "pk int PRIMARY KEY, v int") as table:
            acked: list[int] = []
            failure: list[Exception] = []

            async def write_until_failure() -> None:
                pk = 0
                while True:
                    stmt = SimpleStatement(f"INSERT INTO {table} (pk, v) VALUES ({pk}, {pk})",
                                           consistency_level=ConsistencyLevel.QUORUM)
                    try:
                        await cql.run_async(stmt, host=host_c)
                    except Exception as exc:  # noqa: BLE001 -- whatever the driver raises is the result under test
                        failure.append(exc)
                        return
                    acked.append(pk)
                    pk += 1

            writer = asyncio.create_task(write_until_failure())

            async def some_writes_acknowledged():
                return True if len(acked) >= 5 else None
            await wait_for(some_writes_acknowledged, time.time() + 60)

            logger.info(f"Cutting {node_c} off from clients")
            await network.block_clients(node_c.ip_addr, mode="reset")

            # A reset connection fails the in-flight write right away, well within the 200s request timeout.
            await asyncio.wait_for(writer, timeout=60)
            assert failure and isinstance(failure[0], (NoHostAvailable, OperationTimedOut, ConnectionException)), failure
            logger.info(f"Writer stopped after {len(acked)} acknowledged writes with {failure[0]!r}")

            # The peers still see the node, and so does the test over REST.
            assert await manager.api.get_down_endpoints(node_a.ip_addr) == []
            assert await manager.api.get_down_endpoints(node_c.ip_addr) == []

            async def driver_sees(up: bool):
                host = cql.cluster.metadata.get_host(node_c.ip_addr)
                return True if host is not None and host.is_up == up else None
            await wait_for(lambda: driver_sees(False), time.time() + 60)

            logger.info("Removing the cut")
            await network.heal()
            await wait_for_cql_and_get_hosts(cql, [node_c], time.time() + 60)
            await wait_for(lambda: driver_sees(True), time.time() + 120)

            rows = await cql.run_async(SimpleStatement(f"SELECT count(*) FROM {table}", consistency_level=ConsistencyLevel.ALL),
                                       host=host_a)
            # The write that failed may or may not have reached the coordinator before the reset.
            assert len(acked) <= rows[0].count <= len(acked) + 1, (len(acked), rows[0].count)

            # And the node takes writes from clients again.
            await cql.run_async(SimpleStatement(f"INSERT INTO {table} (pk, v) VALUES (-1, -1)", consistency_level=ConsistencyLevel.ALL),
                                host=host_c)
