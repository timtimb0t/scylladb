#
# Copyright (C) 2026-present ScyllaDB
#
# SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
#

import logging

import pytest

from test.pylib.manager_client import ManagerClient
from test.pylib.scylla_cluster import ScyllaVersionDescription
from test.pylib.version_fetch_utils import fetch_and_install_scylla_version

logger = logging.getLogger(__name__)

SCYLLA_2026_1_5_URL = (
    "https://s3.amazonaws.com/downloads.scylladb.com/downloads/scylla/relocatable/scylladb-2026.1/scylla-2026.1.5-0.20260602.678ab80fc38b.aarch64.tar.gz"
)


@pytest.fixture(scope="module")
def scylla_2026_1_5() -> ScyllaVersionDescription:
    exe = fetch_and_install_scylla_version(url=SCYLLA_2026_1_5_URL)
    return ScyllaVersionDescription(path=str(exe), config={}, argv=[])


@pytest.mark.check_nodes_for_errors
async def test_concurrent_bootstrap_with_tablets_and_auto_repair(
    manager: ManagerClient, scylla_2026_1_5: ScyllaVersionDescription
):
    """Reproducer: segfault on joining nodes when tablets and automatic
    incremental repair are both enabled on a fresh cluster.
    """
    cmdline = [
        '--auto-repair-enabled-default', '1',
        '--auto-repair-threshold-default-in-seconds', '1',
    ]


    seed = await manager.server_add(cmdline=cmdline, version=scylla_2026_1_5)
    logger.info(f"Seed node started: {seed}")

    total_nodes = 200
    batch_size = 20
    all_new_servers = []

    for batch_number in range(1, total_nodes // batch_size + 1):
        logger.info(
            f"Starting concurrent bootstrap batch {batch_number}: "
            f"{batch_size} nodes"
        )

        new_servers = await manager.servers_add(
            batch_size,
            cmdline=cmdline,
            version=scylla_2026_1_5,
        )
        all_new_servers.extend(new_servers)

        logger.info(
            f"Batch {batch_number} joined: "
            f"{[server.server_id for server in new_servers]}"
        )

    logger.info(
        f"All {len(all_new_servers)} new nodes joined: "
        f"{[server.server_id for server in all_new_servers]}"
    )
