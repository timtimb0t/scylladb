#
# Copyright (C) 2025-present ScyllaDB
#
# SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
#

"""
Stress test: randomized schema changes concurrent with SC (strongly-consistent)
table workload.

Spawns multiple writers and readers hitting a fixed key pool while a separate
task performs DDL operations (column add/drop, ALTER TYPE, ALTER TABLE
properties, DROP+RECREATE). Validates data integrity via a conservative
valid-set verifier and ensures no unexpected errors occur.

Context: scylladb/scylladb#28546
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Awaitable, Callable

import pytest
from cassandra import (
    OperationTimedOut,
    ReadTimeout,
    WriteTimeout,
)
from cassandra.protocol import InvalidRequest
from test.cluster.util import new_test_keyspace
from test.pylib.manager_client import ManagerClient
from test.pylib.util import wait_for_cql_and_get_hosts

logger = logging.getLogger(__name__)


NUM_KEYS = 100
NUM_WRITERS = 4
NUM_READERS = 4
STRESS_DURATION_S = 30
NUM_TABLETS = 10

COLUMN_TYPES = ["int", "text", "blob", "boolean", "uuid", "timestamp", "bigint"]

# Keep ALTER TYPE, but only in the one direction observed to work:
# int -> varint. DROP+RECREATE resets the table back to int.
ALTER_TYPE_TARGET = "varint"

# Base schema. On DROP+RECREATE always recreate with c=int.
TABLE_SCHEMA = "(pk int PRIMARY KEY, c int, gen int)"


WRITER_CLEAR_FAILURE_EXCEPTIONS = (
    InvalidRequest,
)

WRITER_AMBIGUOUS_EXCEPTIONS = (
    WriteTimeout,
    OperationTimedOut,
)

READER_TRANSIENT_EXCEPTIONS = (
    InvalidRequest,
    ReadTimeout,
    OperationTimedOut,
)

SCHEMA_EXPECTED_EXCEPTIONS = (
    InvalidRequest,
    OperationTimedOut,
)


@dataclass
class ValueState:
    """State of a candidate value in the conservative valid set.

    status:
      - pending: write has started, result not known yet
      - success: write definitely succeeded at `time`
      - timeout: write outcome unknown, keep conservatively
    """
    status: str
    time: Optional[float] = None


class KeyState:
    """Conservative valid-set tracker for one primary key.

    This is not a formal linearizability checker. It maintains a set of
    values that a reader may legitimately observe given:
      - overlapping writes to the same key
      - ambiguous timeouts
      - client-visible real-time ordering

    Pruning rule:
      if write W2 *successfully* completes and W2 started after W1 had already
      successfully completed, then W1 cannot remain the latest value and can be
      removed from the valid set.
    """

    def __init__(self, initial_value: int = 0) -> None:
        now = time.monotonic()
        self.valid_values: dict[int, ValueState] = {
            initial_value: ValueState(status="success", time=now)
        }
        self.next_value: int = initial_value + 1
        self.success_history = deque([(initial_value, now)], maxlen=32)  # circular buffer with length = 32

    def begin_write(self) -> tuple[int, float]:
        """Call BEFORE sending the write. Returns (value, start_time)."""
        value = self.next_value
        self.next_value += 1
        start_time = time.monotonic()
        self.valid_values[value] = ValueState(status="pending", time=None)
        return value, start_time

    def complete_write_success(self, value: int, start_time: float) -> None:
        """Mark write as definitely successful and prune older superseded
        successful values.
        """
        end_time = time.monotonic()
        self.valid_values[value] = ValueState(status="success", time=end_time)
        self.success_history.append((value, end_time))

        to_remove = [
            v
            for v, st in self.valid_values.items()
            if v != value
            and st.status == "success"
            and st.time is not None
            and st.time < start_time
        ]
        for v in to_remove:
            del self.valid_values[v]

    def discard_write(self, value: int) -> None:
        self.valid_values.pop(value, None)

    def complete_write_failure(self, value: int) -> None:
        """Write definitely did not apply."""
        self.valid_values.pop(value, None)

    def complete_write_timeout(self, value: int) -> None:
        """Write outcome unknown. Keep it conservatively but do not let it
        prune other values like a confirmed success would.
        """
        self.valid_values[value] = ValueState(
            status="timeout",
            time=time.monotonic(),
        )

    def reset(self, initial_value: int = 0) -> None:
        """Called after DROP+RECREATE and reinitialization of rows.

        reset the valid set to the recreated-table initial value but keep
        next_value monotonic to avoid reusing previously observed values.
        """
        now = time.monotonic()
        self.valid_values = {
            initial_value: ValueState(status="success", time=now)
        }
        self.success_history = deque([(initial_value, now)], maxlen=32)


@dataclass
class SCTestState:
    ks: str
    table_name: str = "main"
    key_states: dict[int, KeyState] = field(default_factory=dict)
    added_columns: list[str] = field(default_factory=list)
    current_c_type: str = "int"
    generation: int = 0
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)

    # Monotonic column counter; avoids collisions after DROP+RECREATE.
    col_counter: int = 0

    # Stats
    write_success: int = 0
    write_errors: int = 0
    write_timeouts: int = 0
    read_success: int = 0
    read_errors: int = 0
    consistency_violations: int = 0
    schema_ops: int = 0

    # Skip / stale-generation observability
    writer_gen_skips: int = 0
    reader_gen_skips: int = 0
    reader_stale_row_skips: int = 0

    @property
    def fqtn(self) -> str:
        return f"{self.ks}.{self.table_name}"

    def next_col_name(self) -> str:
        self.col_counter += 1
        return f"col_{self.col_counter}"


async def writer_task(state: SCTestState, cql, writer_id: int) -> None:
    rng = random.Random(writer_id)
    logger.info("Writer %d started", writer_id)

    local_writes = 0
    local_errors = 0
    local_timeouts = 0
    local_gen_skips = 0

    while not state.stop_event.is_set():
        pk = rng.randint(0, NUM_KEYS - 1)
        gen = state.generation
        ks = state.key_states[pk]
        value, start_time = ks.begin_write()

        try:
            await cql.run_async(
                f"UPDATE {state.fqtn} "
                f"SET c = {value}, gen = {gen} "
                f"WHERE pk = {pk}"
            )

            if state.generation != gen:
                ks.discard_write(value)
                state.writer_gen_skips += 1
                local_gen_skips += 1
                continue

            ks.complete_write_success(value, start_time)
            state.write_success += 1
            local_writes += 1

        except WRITER_CLEAR_FAILURE_EXCEPTIONS:
            if state.generation != gen:
                ks.discard_write(value)
                state.writer_gen_skips += 1
                local_gen_skips += 1
                continue

            ks.complete_write_failure(value)
            state.write_errors += 1
            local_errors += 1

        except WRITER_AMBIGUOUS_EXCEPTIONS:
            if state.generation != gen:
                ks.discard_write(value)
                state.writer_gen_skips += 1
                local_gen_skips += 1
                continue

            ks.complete_write_timeout(value)
            state.write_timeouts += 1
            local_timeouts += 1

        except Exception as exc:
            raise AssertionError(
                f"Writer {writer_id}: unexpected exception: {exc!r}"
            ) from exc

        await asyncio.sleep(rng.uniform(0.005, 0.02))

    logger.info(
        "Writer %d finished: writes=%d errors=%d timeouts=%d gen_skips=%d",
        writer_id, local_writes, local_errors, local_timeouts, local_gen_skips
    )


async def reader_task(state: SCTestState, cql, reader_id: int) -> None:
    rng = random.Random(1000 + reader_id)
    logger.info("Reader %d started", reader_id)

    local_reads = 0
    local_errors = 0
    local_violations = 0
    local_gen_skips = 0
    local_stale_row_skips = 0

    ### debug
    def dump_valid_values(ks: KeyState):
        return {
            v: {"status": st.status, "time": st.time}
            for v, st in sorted(ks.valid_values.items())
        }

    def dump_snapshot(values):
        return sorted(values)

    def dump_obj(obj):
        try:
            return {
                k: v
                for k, v in vars(obj).items()
                if not k.startswith("_")
            }
        except TypeError:
            return repr(obj)
    ###

    while not state.stop_event.is_set():
        pk = rng.randint(0, NUM_KEYS - 1)
        gen = state.generation
        ks = state.key_states[pk]
        valid_at_start = set(ks.valid_values.keys())

        try:
            rows = await cql.run_async(
                f"SELECT c, gen FROM {state.fqtn} WHERE pk = {pk}"
            )
            read_end = time.monotonic()

            if state.generation != gen:
                state.reader_gen_skips += 1
                local_gen_skips += 1
                continue

            if not rows:
                # Can happen briefly after DROP+RECREATE before row re-init.
                continue

            row = rows[0]
            # If an old-generation writer managed to land on the new table,
            # it carries the stale generation number. Ignore such rows.
            if row.gen != state.generation:
                state.reader_stale_row_skips += 1
                local_stale_row_skips += 1
                continue

            observed = row.c
            valid_now = set(ks.valid_values.keys())
            historical_successes = {
                value for value, ts in ks.success_history
                if ts <= read_end
            }

            allowed = valid_at_start | valid_now | historical_successes
            logger.debug(f'allowed vales: {allowed}')

            if observed not in allowed:
                state.consistency_violations += 1
                local_violations += 1
                logger.error(
                    "CONSISTENCY VIOLATION: pk=%d observed=%s row_gen=%s state_gen=%d "
                    "valid_at_start=%s valid_now=%s allowed=%s "
                    "next_value=%d valid_values=%s key_state=%s state=%s row=%s",
                    pk,
                    observed,
                    getattr(row, "gen", None),
                    state.generation,
                    dump_snapshot(valid_at_start),
                    dump_snapshot(valid_now),
                    dump_snapshot(allowed),
                    ks.next_value,
                    dump_valid_values(ks),
                    dump_obj(ks),
                    {
                        "generation": state.generation,
                        "current_c_type": state.current_c_type,
                        "schema_ops": state.schema_ops,
                        "writer_gen_skips": state.writer_gen_skips,
                        "reader_gen_skips": state.reader_gen_skips,
                        "reader_stale_row_skips": state.reader_stale_row_skips,
                    },
                    dump_obj(row),
                )
            state.read_success += 1
            local_reads += 1

        except READER_TRANSIENT_EXCEPTIONS:
            state.read_errors += 1
            local_errors += 1

        except Exception as exc:
            raise AssertionError(
                f"Reader {reader_id}: unexpected exception: {exc!r}"
            ) from exc

        await asyncio.sleep(rng.uniform(0.005, 0.02))

    logger.info(
        "Reader %d finished: reads=%d errors=%d violations=%d "
        "gen_skips=%d stale_row_skips=%d",
        reader_id,
        local_reads,
        local_errors,
        local_violations,
        local_gen_skips,
        local_stale_row_skips,
    )


SchemaOpHandler = Callable[[SCTestState, object, random.Random], Awaitable[Optional[str]]]

# should we somehow verify that column been added or dropped?
async def op_add_column(state: SCTestState, cql, rng: random.Random) -> Optional[str]:
    col_name = state.next_col_name()
    col_type = rng.choice(COLUMN_TYPES)
    await cql.run_async(f"ALTER TABLE {state.fqtn} ADD {col_name} {col_type}")
    state.added_columns.append(col_name)
    logger.info("DDL: ADD COLUMN %s %s", col_name, col_type)
    return "add_column"


async def op_drop_column(state: SCTestState, cql, rng: random.Random) -> Optional[str]:
    if not state.added_columns:
        return None

    col_name = rng.choice(state.added_columns)
    await cql.run_async(f"ALTER TABLE {state.fqtn} DROP {col_name}")
    state.added_columns.remove(col_name)
    logger.info("DDL: DROP COLUMN %s", col_name)
    return "drop_column"


async def op_alter_type_c(state: SCTestState, cql, rng: random.Random) -> Optional[str]:
    # Keep ALTER TYPE only as one-way int -> varint.
    # DROP+RECREATE resets the schema back to int.
    if state.current_c_type != "int":
        return None

    await cql.run_async(f"ALTER TABLE {state.fqtn} ALTER c TYPE {ALTER_TYPE_TARGET}")

    # Post-condition sanity check: table remains readable after schema change. Do we need it to make test more stable?
    # rows = await cql.run_async(f"SELECT c FROM {state.fqtn} LIMIT 1")
    # assert rows is not None

    state.current_c_type = ALTER_TYPE_TARGET
    logger.info("DDL: ALTER TYPE c int -> %s", ALTER_TYPE_TARGET)
    return "alter_type"


async def op_alter_properties(state: SCTestState, cql, rng: random.Random) -> Optional[str]:
    props = rng.choice([
        f"comment = 'stress_{rng.randint(0, 9999)}'",
        f"gc_grace_seconds = {rng.randint(100, 100000)}",
        f"default_time_to_live = {rng.randint(50, 3600)}",  # TTL should be > stress test duration
    ])
    await cql.run_async(f"ALTER TABLE {state.fqtn} WITH {props}")
    logger.info("DDL: ALTER TABLE WITH %s", props)
    return "alter_props"


async def op_drop_recreate(state: SCTestState, cql, rng: random.Random) -> Optional[str]:
    old_gen = state.generation
    logger.info("DDL: DROP+RECREATE starting (gen=%d)", old_gen)

    # 1. Bump generation before DDL. In-flight old-generation writers/readers
    # will detect mismatch and discard/skip their results.
    state.generation += 1
    new_gen = state.generation

    # 2. DROP TABLE
    try:
        await cql.run_async(f"DROP TABLE {state.fqtn}")
    except SCHEMA_EXPECTED_EXCEPTIONS as exc:
        logger.warning("DDL: DROP TABLE failed (continuing): %s", exc)

    # 3. RECREATE base schema (c type resets to int)
    await cql.run_async(f"CREATE TABLE {state.fqtn} {TABLE_SCHEMA}")

    # 4. Reset in-memory model BEFORE reinitializing rows, so new-generation
    # writers do not update a stale pre-reset model.
    for pk in range(NUM_KEYS):
        state.key_states[pk].reset(initial_value=0)

    # 5. Reset schema-change bookkeeping
    state.added_columns.clear()
    state.current_c_type = "int"

    # 6. Reinitialize base rows concurrently
    batch_size = 50
    for start in range(0, NUM_KEYS, batch_size):
        end = min(start + batch_size, NUM_KEYS)
        await asyncio.gather(*[
            cql.run_async(
                f"UPDATE {state.fqtn} "
                f"SET c = 0, gen = {new_gen} "
                f"WHERE pk = {pk}"
            )
            for pk in range(start, end)
        ])

    logger.info("DDL: DROP+RECREATE done (gen=%d)", new_gen)
    return "drop_recreate"


SCHEMA_OPS = [
    ("add_column", 3),
    ("drop_column", 2),
    ("alter_type", 1),
    ("alter_props", 2),
    ("drop_recreate", 1),
]


SCHEMA_OP_HANDLERS: dict[str, SchemaOpHandler] = {
    "add_column": op_add_column,
    "drop_column": op_drop_column,
    "alter_type": op_alter_type_c,
    "alter_props": op_alter_properties,
    "drop_recreate": op_drop_recreate,
}


async def run_schema_op(
    state: SCTestState,
    cql,
    op_name: str,
    rng: random.Random,
) -> Optional[str]:
    try:
        handler = SCHEMA_OP_HANDLERS[op_name]
    except KeyError as exc:
        raise ValueError(f"Unknown schema op: {op_name}") from exc
    return await handler(state, cql, rng)


async def schema_changer_task(state: SCTestState, cql) -> None:
    rng = random.Random(42)
    logger.info("Schema changer started")

    while not state.stop_event.is_set():
        available_ops = [
            (name, weight) for name, weight in SCHEMA_OPS
            if not (name == "alter_type" and state.current_c_type != "int")
        ]
        op_names = [name for name, _ in available_ops]
        op_weights = [weight for _, weight in available_ops]

        [op_name] = rng.choices(op_names, weights=op_weights)

        try:
            result = await run_schema_op(state, cql, op_name, rng)
            if result is not None:
                state.schema_ops += 1
                logger.info("Schema op #%d: %s", state.schema_ops, result)

        except SCHEMA_EXPECTED_EXCEPTIONS as exc:
            logger.warning("Schema op %s failed (expected): %s", op_name, exc)

        except Exception as exc:
            raise AssertionError(
                f"Schema changer: unexpected exception in {op_name}: {exc!r}"
            ) from exc

        await asyncio.sleep(rng.uniform(1.0, 3.0))

    logger.info("Schema changer finished: %d ops", state.schema_ops)


@pytest.mark.asyncio
async def test_sc_randomized_schema_changes(manager: ManagerClient):
    """Stress SC schema/apply path with randomized schema changes concurrent
    with read/write workload on a fixed key pool.
    """

    logger.info("Bootstrapping cluster")
    config = {
        "experimental_features": ["strongly-consistent-tables"],
    }
    cmdline = [
        "--logger-log-level", "sc_groups_manager=debug",
        "--logger-log-level", "sc_coordinator=debug",
    ]

    servers = await manager.servers_add(
        6,
        config=config,
        cmdline=cmdline,
        auto_rack_dc="my_dc",
    )

    cql = manager.get_cql()
    await wait_for_cql_and_get_hosts(cql, servers, time.time() + 60)

    ks_opts = (
        "WITH replication = "
        "{'class': 'NetworkTopologyStrategy', 'replication_factor': 3} "
        f"AND tablets = {{'initial': {NUM_TABLETS}}} "
        "AND consistency = 'local'"
    )

    async with new_test_keyspace(manager, ks_opts) as ks:
        table_name = "main"
        fqtn = f"{ks}.{table_name}"

        await cql.run_async(f"CREATE TABLE {fqtn} {TABLE_SCHEMA}")

        logger.info("Pre-initializing %d rows", NUM_KEYS)
        batch_size = 50
        for start in range(0, NUM_KEYS, batch_size):
            end = min(start + batch_size, NUM_KEYS)
            await asyncio.gather(*[
                cql.run_async(
                    f"UPDATE {fqtn} SET c = 0, gen = 0 WHERE pk = {pk}"
                )
                for pk in range(start, end)
            ])

        state = SCTestState(
            ks=ks,
            table_name=table_name,
            key_states={pk: KeyState(initial_value=0) for pk in range(NUM_KEYS)},
        )

        logger.info(
            "Starting stress phase (%ds): writers=%d readers=%d keys=%d tablets=%d",
            STRESS_DURATION_S, NUM_WRITERS, NUM_READERS, NUM_KEYS, NUM_TABLETS
        )

        async def stop_after_deadline() -> None:
            await asyncio.sleep(STRESS_DURATION_S)
            state.stop_event.set()

        tasks = [
            asyncio.create_task(stop_after_deadline()),
            *[
                asyncio.create_task(writer_task(state, cql, i))
                for i in range(NUM_WRITERS)
            ],
            *[
                asyncio.create_task(reader_task(state, cql, i))
                for i in range(NUM_READERS)
            ],
            asyncio.create_task(schema_changer_task(state, cql)),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        task_errors = [r for r in results if isinstance(r, Exception)]

        logger.info("Stress phase complete — running verification")

        # for diagnostic purposes, logging the valid-set sizes and some stats before assertions
        vs_sizes = [len(state.key_states[pk].valid_values) for pk in range(NUM_KEYS)]
        logger.info(
            "Stats: writes ok=%d err=%d timeout=%d | "
            "reads ok=%d err=%d violations=%d | "
            "schema_ops=%d generations=%d | "
            "writer_gen_skips=%d reader_gen_skips=%d stale_row_skips=%d",
            state.write_success,
            state.write_errors,
            state.write_timeouts,
            state.read_success,
            state.read_errors,
            state.consistency_violations,
            state.schema_ops,
            state.generation,
            state.writer_gen_skips,
            state.reader_gen_skips,
            state.reader_stale_row_skips,
        )
        logger.info(
            "Valid-set sizes: min=%d max=%d avg=%.1f",
            min(vs_sizes),
            max(vs_sizes),
            sum(vs_sizes) / len(vs_sizes),
        )

        # 1. No task errors
        assert not task_errors, f"Task(s) failed with unexpected exceptions: {task_errors}"

        # test approach may be changed to number of operations, but what number?
        # 2. No valid-set consistency violations
        assert state.consistency_violations == 0, (
            f"Detected {state.consistency_violations} valid-set consistency "
            f"violation(s) — see CONSISTENCY VIOLATION log lines above"
        )

        # 3. Minimum useful activity
        assert state.write_success >= 100, (
            f"Too few successful writes ({state.write_success}), expected >= 100"
        )
        assert state.schema_ops >= 5, (
            f"Too few schema operations ({state.schema_ops}), expected >= 5"
        )

    logger.info("Test passed")
