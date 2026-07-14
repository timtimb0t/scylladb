#
# Copyright (C) 2026-present ScyllaDB
#
# SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
#
# Dataclasses for parameterized test fixtures.
# Import these in tests for use with @pytest.mark.parametrize(..., indirect=True)
#
# Example:
#   from test.cluster.fixture_params import ClusterConfig, KeyspaceConfig
#
#   @pytest.mark.parametrize("cluster_config", [
#       ClusterConfig(num_nodes=1),
#       ClusterConfig(num_nodes=3),
#   ], indirect=True)
#   async def test_something(manager, servers):
#       ...

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple


@dataclass(frozen=True)
class ClusterConfig:
    """Configuration for cluster setup (nodes + cmdline + config)."""
    num_nodes: int = 3
    cmdline: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    property_file: dict | list[dict] = field(default_factory=dict)
    auto_rack_dc: str | None = None

    def __str__(self) -> str:
        parts = [f"{self.num_nodes}n"]
        if self.auto_rack_dc:
            parts.append(self.auto_rack_dc)
        if self.cmdline:
            parts.append(f"cmdline={len(self.cmdline)}opts")
        return "-".join(parts)


@dataclass(frozen=True)
class KeyspaceConfig:
    """Configuration for keyspace creation (replication + tablets + consistency)."""
    replication_factor: int = 3
    tablets: bool | None = None         # None = server default, True = force on, False = force off
    initial_tablets: int | None = None  # WITH tablets = {'initial': N}
    consistency: str | None = None      # None = default, 'global' = strong consistency
    extra_opts: str = ""

    def build_opts(self) -> str:
        """Build the CQL options string for CREATE KEYSPACE."""
        opts = (
            f"WITH replication = "
            f"{{'class': 'NetworkTopologyStrategy', 'replication_factor': {self.replication_factor}}}"
        )

        if self.tablets is not None or self.initial_tablets is not None:
            tablets_parts = []
            if self.tablets is not None:
                tablets_parts.append(f"'enabled': {'true' if self.tablets else 'false'}")
            if self.initial_tablets is not None:
                tablets_parts.append(f"'initial': {self.initial_tablets}")
            opts += f" AND tablets = {{{', '.join(tablets_parts)}}}"

        if self.consistency:
            opts += f" AND consistency = '{self.consistency}'"

        if self.extra_opts:
            opts += f" AND {self.extra_opts}"

        return opts

    def __str__(self) -> str:
        parts = [f"RF{self.replication_factor}"]
        if self.tablets is True:
            parts.append("tablets")
        elif self.tablets is False:
            parts.append("vnodes")
        if self.initial_tablets is not None:
            parts.append(f"{self.initial_tablets}t")
        if self.consistency:
            parts.append(self.consistency)
        return "-".join(parts)


@dataclass(frozen=True)
class TableConfig:
    """Configuration for table creation."""
    schema: str = "pk int PRIMARY KEY, v int"
    extra: str = ""  # e.g. "WITH cdc = {'enabled': true}"

    def __str__(self) -> str:
        # Show a short summary of the schema for test IDs
        return self.schema[:40].replace(" ", "")


class FullSetupResult(NamedTuple):
    """Result of the full_setup fixture: cluster + keyspace + table."""
    servers: list
    keyspace: str
    table: str


@dataclass(frozen=True)
class TestSetup:
    """Composite config for full_setup fixture: cluster + keyspace + table in one shot."""
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    keyspace: KeyspaceConfig = field(default_factory=KeyspaceConfig)
    table: TableConfig = field(default_factory=TableConfig)

    def __str__(self) -> str:
        return f"{self.cluster}|{self.keyspace}|{self.table}"
