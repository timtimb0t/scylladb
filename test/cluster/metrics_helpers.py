#
# Copyright (C) 2026-present ScyllaDB
#
# SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
#
"""
Prometheus histogram helpers for LWT benchmark latency computation.

Parses Prometheus exposition format histograms from ScyllaDB metrics
endpoint and computes latency percentiles (avg, p50, p95, p99) using
linear interpolation within histogram buckets.

ScyllaDB latency histograms use **microseconds** as the unit.
All public functions return results in **milliseconds**.

Typical usage in a benchmark test:

    # Before measurement
    snap_before = await snapshot_latency_metrics(manager, servers)

    # ... run benchmark ...

    # After measurement
    snap_after = await snapshot_latency_metrics(manager, servers)

    # Compute latency stats from the delta
    read_stats = compute_latency_from_snapshots(snap_before, snap_after, "read")
    write_stats = compute_latency_from_snapshots(snap_before, snap_after, "write")
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from test.pylib.manager_client import ManagerClient

logger = logging.getLogger(__name__)

# ScyllaDB reports latency in microseconds
_US_TO_MS = 0.001

# The scheduling group for user CQL traffic (default service level).
# Internal driver metadata queries go to "sl:driver" — we exclude those
# to avoid diluting the benchmark signal.
_USER_SCHEDULING_GROUP = "sl:default"

# Prometheus metric names for coordinator-level latency histograms.
_READ_LATENCY_METRIC = "scylla_storage_proxy_coordinator_read_latency"
_WRITE_LATENCY_METRIC = "scylla_storage_proxy_coordinator_write_latency"


@dataclass
class HistogramData:
    """Parsed cumulative histogram from a Prometheus exposition block.

    Attributes:
        buckets: sorted list of (upper_bound, cumulative_count) pairs.
                 Does NOT include the +Inf bucket.
        total_sum: value of the ``_sum`` line.
        total_count: value of the ``_count`` line.
    """
    buckets: List[Tuple[float, int]]
    total_sum: float
    total_count: int


@dataclass
class LatencyStats:
    """Computed latency statistics in milliseconds."""
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    total_count: int = 0


@dataclass
class LatencySnapshot:
    """Raw histogram snapshots captured at a point in time.

    Keyed by ``(server_ip, "read"|"write")`` → ``HistogramData``.
    """
    histograms: Dict[Tuple[str, str], HistogramData]


def _parse_histogram(lines: List[str], metric_name: str,
                     scheduling_group: str) -> HistogramData:
    """Extract a single histogram series from Prometheus text lines.

    Filters by ``scheduling_group_name="<scheduling_group>"``.
    Returns aggregated histogram across all matching label combinations
    (i.e. summed across shards — the histogram already comes without a
    shard label in ScyllaDB's exposition format).
    """
    label_filter = f'scheduling_group_name="{scheduling_group}"'
    buckets_map: Dict[float, int] = {}
    total_sum = 0.0
    total_count = 0

    for line in lines:
        if not line or line.startswith('#'):
            continue

        # Bucket line:  metric_bucket{...,le="NNN",...} VALUE
        if f'{metric_name}_bucket{{' in line and label_filter in line:
            le_match = re.search(r'le="([^"]+)"', line)
            if le_match:
                le_str = le_match.group(1)
                if le_str == '+Inf':
                    continue
                le_val = float(le_str)
                count_val = int(float(line.rsplit(None, 1)[-1]))
                buckets_map[le_val] = buckets_map.get(le_val, 0) + count_val

        # Sum line:  metric_sum{...} VALUE
        elif f'{metric_name}_sum{{' in line and label_filter in line:
            total_sum += float(line.rsplit(None, 1)[-1])

        # Count line:  metric_count{...} VALUE
        elif f'{metric_name}_count{{' in line and label_filter in line:
            total_count += int(float(line.rsplit(None, 1)[-1]))

    buckets = sorted(buckets_map.items())
    return HistogramData(buckets=buckets, total_sum=total_sum,
                         total_count=total_count)


async def snapshot_latency_metrics(
    manager: ManagerClient,
    servers,
    scheduling_group: str = _USER_SCHEDULING_GROUP,
) -> LatencySnapshot:
    """Query every server and capture read/write latency histograms.

    Args:
        manager: the ManagerClient with a ``.metrics`` attribute.
        servers: list of server objects (each has ``.ip_addr``).
        scheduling_group: which scheduling group to capture
                          (default ``"sl:default"`` for user CQL traffic).

    Returns:
        A ``LatencySnapshot`` containing one histogram per (server, direction).
    """
    histograms: Dict[Tuple[str, str], HistogramData] = {}

    for srv in servers:
        ip = srv.ip_addr
        metrics = await manager.metrics.query(ip)
        for direction, metric_name in [("read", _READ_LATENCY_METRIC),
                                       ("write", _WRITE_LATENCY_METRIC)]:
            h = _parse_histogram(metrics.lines, metric_name, scheduling_group)
            histograms[(ip, direction)] = h

    return LatencySnapshot(histograms=histograms)


def _subtract_histograms(after: HistogramData,
                         before: HistogramData) -> HistogramData:
    """Compute the delta histogram (after − before).

    Both histograms must share the same bucket boundaries (ScyllaDB uses
    fixed boundaries, so this is guaranteed for the same metric).
    """
    before_map = dict(before.buckets)
    delta_buckets = []
    for le, after_count in after.buckets:
        before_count = before_map.get(le, 0)
        delta_buckets.append((le, max(after_count - before_count, 0)))

    return HistogramData(
        buckets=delta_buckets,
        total_sum=after.total_sum - before.total_sum,
        total_count=max(after.total_count - before.total_count, 0),
    )


def _merge_histograms(histograms: List[HistogramData]) -> HistogramData:
    """Merge multiple histograms (e.g. from different servers) by summing."""
    if not histograms:
        return HistogramData(buckets=[], total_sum=0.0, total_count=0)

    merged_map: Dict[float, int] = {}
    total_sum = 0.0
    total_count = 0

    for h in histograms:
        total_sum += h.total_sum
        total_count += h.total_count
        for le, count in h.buckets:
            merged_map[le] = merged_map.get(le, 0) + count

    return HistogramData(
        buckets=sorted(merged_map.items()),
        total_sum=total_sum,
        total_count=total_count,
    )


def _percentile_from_histogram(buckets: List[Tuple[float, int]],
                               count: int, quantile: float) -> float:
    """Compute a percentile from cumulative histogram buckets.

    Uses linear interpolation within the bucket where the target
    observation falls — the standard Prometheus ``histogram_quantile``
    algorithm.

    Args:
        buckets: sorted list of ``(upper_bound_us, cumulative_count)``.
        count:   total observation count (``_count``).
        quantile: target quantile, e.g. 0.50, 0.95, 0.99.

    Returns:
        Estimated latency in **microseconds**.
    """
    if count == 0 or not buckets:
        return 0.0

    target = quantile * count
    prev_le = 0.0
    prev_count = 0

    for le, cum_count in buckets:
        if cum_count >= target:
            bucket_width = le - prev_le
            bucket_count = cum_count - prev_count
            if bucket_count == 0:
                return prev_le
            fraction = (target - prev_count) / bucket_count
            return prev_le + fraction * bucket_width
        prev_le = le
        prev_count = cum_count

    # All observations beyond last finite bucket — return last boundary
    return buckets[-1][0] if buckets else 0.0


def _compute_stats(histogram: HistogramData) -> LatencyStats:
    """Compute avg / p50 / p95 / p99 from a histogram.  All in ms."""
    count = histogram.total_count
    if count == 0:
        return LatencyStats()

    avg_us = histogram.total_sum / count
    p50_us = _percentile_from_histogram(histogram.buckets, count, 0.50)
    p95_us = _percentile_from_histogram(histogram.buckets, count, 0.95)
    p99_us = _percentile_from_histogram(histogram.buckets, count, 0.99)

    return LatencyStats(
        avg_latency_ms=round(avg_us * _US_TO_MS, 3),
        p50_latency_ms=round(p50_us * _US_TO_MS, 3),
        p95_latency_ms=round(p95_us * _US_TO_MS, 3),
        p99_latency_ms=round(p99_us * _US_TO_MS, 3),
        total_count=count,
    )


def compute_latency_from_snapshots(
    before: LatencySnapshot,
    after: LatencySnapshot,
    direction: str,
) -> LatencyStats:
    """Compute latency percentiles for *direction* ("read" or "write")
    from two snapshots taken before and after a measurement window.

    Steps:
      1. For each server, compute delta histogram (after − before).
      2. Merge delta histograms across all servers.
      3. Compute percentiles from the merged delta.

    Returns:
        ``LatencyStats`` with avg/p50/p95/p99 in milliseconds.
    """
    deltas: List[HistogramData] = []

    # Collect all server IPs that have this direction in both snapshots
    after_keys = {k for k in after.histograms if k[1] == direction}

    for key in after_keys:
        h_after = after.histograms[key]
        h_before = before.histograms.get(key)
        if h_before is None:
            # Server was added between snapshots — use after as-is
            deltas.append(h_after)
        else:
            deltas.append(_subtract_histograms(h_after, h_before))

    merged = _merge_histograms(deltas)
    stats = _compute_stats(merged)

    logger.info(
        "Latency stats [%s]: avg=%.3fms p50=%.3fms p95=%.3fms p99=%.3fms "
        "(n=%d observations across %d server(s))",
        direction, stats.avg_latency_ms, stats.p50_latency_ms,
        stats.p95_latency_ms, stats.p99_latency_ms,
        stats.total_count, len(deltas),
    )
    return stats
