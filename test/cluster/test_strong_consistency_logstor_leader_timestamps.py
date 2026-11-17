#
# Copyright (C) 2026-present ScyllaDB
#
# SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
#
#############################################################################
# The timestamps of a new leader of a strongly consistent raft group on a
# logstor table. See test_strong_consistency_leader_timestamps.py for the
# rule: a leader whose clock is behind must still issue timestamps above the
# writes in the group's log, or its writes are acked and then lose to older
# ones. The new leader learns how high to start from
# table::get_max_timestamp_for_token_range(), which looks at the memtables and
# at the sstables. A logstor table keeps its data in neither: table::apply()
# hands its writes to logstor, which keeps the entry with the higher timestamp.
#############################################################################

import pytest

from test.cluster.test_strong_consistency import DEFAULT_CONFIG
from test.cluster.test_strong_consistency_leader_timestamps import (
    WAIT_TIMEOUT_S, check_new_leader_write_wins, cluster_with_skewed_leader, hand_over_to_new_leader,
    write_on_old_leader)
from test.pylib.rest_client import read_barrier
from test.pylib.scylla_cluster_manager import ScyllaClusterManager

pytestmark = pytest.mark.skip_mode(mode='release', reason='error injections are not supported in release mode')

LOGSTOR_CONFIG = DEFAULT_CONFIG | {'experimental_features': ['strongly-consistent-tables', 'logstor']}


@pytest.mark.asyncio
@pytest.mark.xfail(reason="a new leader seeds its timestamps without the logstor data and writes below them",
                   strict=True)
async def test_new_leader_timestamps_above_write_in_logstor(manager: ScyllaClusterManager):
    """
    Verify that a new leader whose clock is behind issues write timestamps above the
    timestamp of an acked write of the old leader on a logstor table, where the write is
    in no memtable and no sstable of any replica.

    Fails while the new leader seeds its term from the memtables and sstables only: its
    write gets its own clock's timestamp, an hour below the old leader's, and a read
    through the new leader right after the ack returns the old value.
    """
    async with cluster_with_skewed_leader(manager, config=LOGSTOR_CONFIG,
                                          table_options=" WITH storage_engine = 'logstor'") as cluster:
        old_write_ts = await write_on_old_leader(cluster)
        for s in cluster.servers:
            await read_barrier(manager.api, s.ip_addr, cluster.group_id, timeout=WAIT_TIMEOUT_S)
        await hand_over_to_new_leader(cluster)
        await check_new_leader_write_wins(cluster, old_write_ts)
