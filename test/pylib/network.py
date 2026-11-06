#
# Copyright (C) 2026-present ScyllaDB
#
# SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
#

"""Network partitions between the nodes of a test cluster.

The nodes of a test cluster are plain processes on the loopback device, one
127.X.Z.W address each, and they talk to each other over TCP (internode RPC on
port 7000). Cutting that traffic needs packet filter rules, and installing
those on the host loopback needs CAP_NET_ADMIN: the test harness runs as a
regular user, under dbuild it is dropped to the host uid with setpriv, so it
cannot do it there.

What it can do is bring its own network. An unprivileged user namespace plus a
network namespace gives the process tree a private loopback device over which
it holds CAP_NET_ADMIN, so nft rules work without root, and die with the
process tree, so nothing leaks onto the host. Every 127.X.Z.W address the
harness hands out exists in that namespace exactly as it does on the host, so
nothing about addressing changes. The namespace has no link to the outside by
itself; pasta, attached from outside, gives it one through the host's stack,
for the tests that download older Scylla versions.

``reexec_in_test_netns`` moves test.py into such a namespace before it starts
anything else, so the harness, its pytest workers and every Scylla process
share it. ``NetworkPartition`` then installs nft rules in it, the same rules on
a developer machine and on CI, since both build the namespace the same way
instead of depending on host configuration.

The harness also runs containers (the S3 mock, for one) with rootless podman,
and podman needs a user namespace with a whole range of ids, not just one: it
runs the container's root as id 0 and the container's other users as the
subordinate ids of ``/etc/subuid``. So the namespace maps the current user to
root and the user's subordinate range to the ids above it, which is the layout
rootless podman builds for itself, and podman is told through its own
environment variables that the namespace is already there. Under dbuild with
rootless podman the harness already is root of such a namespace, and the new
one maps that one's ids one to one.

Environment variables:

``SCYLLA_TEST_NETNS``
    Set by the re-exec: ``1`` inside the namespace, ``0`` when the namespace
    was disabled (``--no-netns``) or could not be created. Tests that need the
    namespace fail, not skip, in the latter case, so the gap is visible.
"""

from __future__ import annotations

import asyncio
import grp
import logging
import os
import pwd
import shutil
import signal
import subprocess
import sys
import tempfile
from typing import Iterable, Literal

from test.pylib.internal_types import IPAddress

logger = logging.getLogger(__name__)

NETNS_ENV = "SCYLLA_TEST_NETNS"
INTERNODE_PORT = 7000
# CQL, with and without TLS, plus the shard-aware variants the driver prefers.
CLIENT_PORTS = (9042, 9142, 19042, 19142)


def in_test_netns() -> bool:
    """Whether this process runs inside the namespace built by ``reexec_in_test_netns``."""
    return os.environ.get(NETNS_ENV) == "1"


def _subordinate_range(path: str, names: Iterable[str]) -> tuple[int, int] | None:
    """The first ``start:count`` range that /etc/subuid or /etc/subgid grants to one of ``names``."""
    try:
        with open(path) as f:
            for line in f:
                name, _, rest = line.strip().partition(":")
                if name in names:
                    start, _, count = rest.partition(":")
                    return int(start), int(count)
    except (OSError, ValueError):
        pass
    return None


def _mapped_ids(path: str) -> int:
    """How many ids the current user namespace maps, per /proc/self/uid_map or gid_map."""
    with open(path) as f:
        return sum(int(line.split()[2]) for line in f)


def _namespace_layout() -> tuple[list[str], dict[str, str]] | str:
    """The unshare options and environment for the namespace, or why it cannot be built."""
    uid, gid = os.getuid(), os.getgid()
    env: dict[str, str] = {}
    if uid == 0:
        # Root of a user namespace already (dbuild under rootless podman): map
        # its ids one to one, so that containers started inside see the same
        # layout as outside.
        uids, gids = _mapped_ids("/proc/self/uid_map") - 1, _mapped_ids("/proc/self/gid_map") - 1
        if uids <= 0 or gids <= 0:
            return "running as root with no subordinate ids mapped"
        users, groups = f"1:1:{uids}", f"1:1:{gids}"
    else:
        subuid = _subordinate_range("/etc/subuid", {pwd.getpwuid(uid).pw_name, str(uid)})
        subgid = _subordinate_range("/etc/subgid", {grp.getgrgid(gid).gr_name, str(gid)})
        if subuid is None or subgid is None:
            return f"no subordinate id range for uid {uid} in /etc/subuid and /etc/subgid"
        users, groups = f"1:{subuid[0]}:{subuid[1]}", f"1:{subgid[0]}:{subgid[1]}"
        # What rootless podman sets for itself once inside its namespace: it
        # then uses this one instead of building its own.
        env = {"_CONTAINERS_USERNS_CONFIGURED": "done",
               "_CONTAINERS_ROOTLESS_UID": str(uid),
               "_CONTAINERS_ROOTLESS_GID": str(gid),
               # Podman keeps per-session state here; the host session's refers to
               # a namespace this one cannot join.
               "XDG_RUNTIME_DIR": tempfile.mkdtemp(prefix="scylla-test-netns-")}
    # The mount namespace is for the containers, which mount their file systems;
    # as root of the namespace the process keeps its capabilities across exec.
    return (["--map-user=0", "--map-group=0", f"--map-users={users}", f"--map-groups={groups}",
             "--net", "--mount"], env)


def _probe_test_netns(unshare: str, options: list[str], env: dict[str, str], ip: str, nft: str) -> str | None:
    """Try to build the namespace and install one nft rule in it; return the error, if any."""
    probe = f"{ip} link set lo up && {nft} add table ip scylla_test_probe"
    try:
        res = subprocess.run([unshare, *options, "sh", "-c", probe], env={**os.environ, **env},
                             capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    if res.returncode != 0:
        return (res.stderr or res.stdout).strip() or f"exit code {res.returncode}"
    return None


def _start_uplink(pasta: str, child: int) -> subprocess.Popen | None:
    """Attach pasta to the child's namespace, from outside it: the namespace's way out to the host's network.

    pasta copies the host's addresses and routes into the namespace and forwards its traffic
    through the host's stack, as the user, with no capability on the host. It runs in the
    foreground here so that its failure is visible, and it exits by itself with the namespace.
    """
    cmd = [pasta, "--foreground", "--quiet", "--config-net",
           "--netns", f"/proc/{child}/ns/net", "--userns", f"/proc/{child}/ns/user"]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        proc.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        return proc  # still running after configuring the namespace: attached
    except OSError as exc:
        print(f"test.py: no uplink from the network namespace (pasta: {exc}); downloads will fail", file=sys.stderr)
        return None
    print(f"test.py: no uplink from the network namespace (pasta exited {proc.returncode}: {proc.stderr.read().strip()}); "
          f"downloads will fail", file=sys.stderr)
    return None


def _kill_namespace_leftovers(netns: str, child: int) -> None:
    """Kill whatever still runs in the namespace once test.py is done with it, e.g. podman's pause process."""
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) in (child, os.getpid()):
            continue
        try:
            if os.readlink(f"/proc/{entry}/ns/net") == netns:
                os.kill(int(entry), signal.SIGKILL)
        except OSError:
            continue


def _keep_namespace(child: int, ready_r: int, go_w: int, env: dict[str, str]) -> None:
    """The parent's side of the fork: attach the uplink, wait for the child, clean up, exit with its status."""
    if not os.read(ready_r, 1):
        # The child died before the namespace was up; its own error is on stderr.
        _, status = os.waitpid(child, 0)
        os._exit(os.waitstatus_to_exitcode(status) or 1)
    os.close(ready_r)
    netns = os.readlink(f"/proc/{child}/ns/net")
    pasta = shutil.which("pasta")
    if pasta is None:
        print("test.py: no uplink from the network namespace (pasta not installed); downloads will fail", file=sys.stderr)
    uplink = _start_uplink(pasta, child) if pasta else None
    os.write(go_w, b"\n")
    os.close(go_w)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: os.kill(child, signum))
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    while True:
        try:
            _, status = os.waitpid(child, 0)
            break
        except InterruptedError:
            continue
    _kill_namespace_leftovers(netns, child)
    if uplink is not None and uplink.poll() is None:
        uplink.terminate()
    if runtime_dir := env.get("XDG_RUNTIME_DIR"):
        shutil.rmtree(runtime_dir, ignore_errors=True)
    code = os.waitstatus_to_exitcode(status)
    os._exit(code if code >= 0 else 128 - code)


def reexec_in_test_netns() -> None:
    """Run the rest of test.py inside a private user+network namespace, with an uplink.

    Returns without doing anything when already inside one, or when the
    namespace was disabled with ``--no-netns``. When the namespace cannot be
    built (no unshare/ip/nft, no subordinate ids for the user, user namespaces
    disabled on the host, nf_tables not loaded) the run continues on the host
    network with a warning, and ``SCYLLA_TEST_NETNS=0`` tells the tests that
    need the namespace to fail.

    Otherwise this process forks. The child builds the namespace and becomes
    test.py again inside it; the parent stays on the host network as the
    namespace's keeper: it attaches pasta for the uplink, forwards SIGTERM and
    SIGHUP, waits, kills whatever the run left behind in the namespace, and
    exits with the child's status. SIGINT from the terminal reaches the child
    directly, as part of the foreground process group, so it is not forwarded.
    """
    if os.environ.get(NETNS_ENV) is not None:
        return
    if "--no-netns" in sys.argv:
        os.environ[NETNS_ENV] = "0"
        return

    tools = {name: shutil.which(name) for name in ("unshare", "ip", "nft")}
    if missing := [name for name, path in tools.items() if path is None]:
        error = f"missing tools: {', '.join(missing)}"
    else:
        layout = _namespace_layout()
        if isinstance(layout, str):
            error = layout
        else:
            options, env = layout
            error = _probe_test_netns(tools["unshare"], options, env, tools["ip"], tools["nft"])
    if error is not None:
        print(f"test.py: not running in a private network namespace ({error}); "
              f"tests that partition the network will fail, pass --no-netns to silence this",
              file=sys.stderr)
        os.environ[NETNS_ENV] = "0"
        return

    os.environ[NETNS_ENV] = "1"
    os.environ.update(env)
    # Whatever was printed so far would be lost with the child's process image.
    sys.stdout.flush()
    sys.stderr.flush()
    # Two pipes: the child reports the namespace ready, the parent tells it to go on once the uplink is up.
    ready_r, ready_w = os.pipe()
    go_r, go_w = os.pipe()
    os.set_inheritable(ready_w, True)
    os.set_inheritable(go_r, True)
    child = os.fork()
    if child == 0:
        os.close(ready_r)
        os.close(go_w)
        ip = tools["ip"]
        # The new namespace starts with its loopback device down. Once told to go on, wait a
        # little for pasta's default route, which may not come (no uplink on the host), then
        # hand over to test.py: "$0" "$@" are the interpreter and its arguments.
        bootstrap = (f'{ip} link set lo up && echo >&{ready_w} && exec {ready_w}>&- && '
                     f'read -r _ <&{go_r} && exec {go_r}<&- && '
                     f'i=0; while [ $i -lt 30 ] && ! {ip} route | grep -q default; do sleep 0.1; i=$((i+1)); done; '
                     f'exec "$0" "$@"')
        os.execv(tools["unshare"], [tools["unshare"], *options, "sh", "-c", bootstrap, sys.executable, *sys.orig_argv[1:]])

    os.close(ready_w)
    os.close(go_r)
    _keep_namespace(child, ready_r, go_w, env)


def require_test_netns() -> None:
    """Raise if the process is not inside the test network namespace."""
    if not in_test_netns():
        raise RuntimeError("this test cuts the network between nodes, which needs test.py to run inside its private "
                           "network namespace; it did not (see the warning test.py printed at startup, "
                           "or drop --no-netns)")


CutMode = Literal["drop", "reset"]


class NetworkPartition:
    """A set of cuts between node addresses, installed as nft rules.

    Each instance owns one nft table, so partitions of concurrently running
    tests (xdist workers share the namespace) do not interfere. The rules
    match on the source and destination addresses, which works because a
    node binds its outgoing RPC connections to its listen address, so every
    internode packet carries the two node addresses.

    A cut can be limited to ports: cut on the internode port only, a node is
    still reached by the test over CQL and the REST API, only its peers do
    not; cut on the client ports only (``block_clients``), its peers still
    see it while the driver does not. ``blackout`` cuts everything, as a
    machine that lost its network link would be.

    ``mode`` selects what the cut-off side observes: ``drop`` loses the
    packets silently, so connections hang until they time out; ``reset``
    answers with a TCP reset, so they fail at once. ``loss`` is the share of
    the matching packets the rule acts on, in percent: at 100, the default, the
    link is cut; below it the link is lossy, and TCP rides it with retransmits,
    so nodes see latency spikes and stalls rather than errors. Delaying packets
    is out of reach: nft decides whether a packet passes, it cannot hold one.
    """

    def __init__(self, name: str):
        self.table = f"scylla_test_{name}"
        self._rules: list[str] = []

    @staticmethod
    def _addresses(addrs: IPAddress | Iterable[IPAddress]) -> list[IPAddress]:
        return [addrs] if isinstance(addrs, str) else list(addrs)

    @staticmethod
    def _verdict(mode: CutMode, loss: int) -> str:
        if not 0 < loss <= 100:
            raise ValueError(f"loss must be in 1..100 percent, got {loss}")
        verdict = "drop" if mode == "drop" else "reject with tcp reset"
        if loss < 100:
            # numgen draws a fresh number per packet, so the rule acts on the given share of them.
            verdict = f"numgen random mod 100 < {loss} {verdict}"
        if mode == "reset":
            # The resets the rules send back carry the cut-off address as their source, so a
            # rule for the other direction would swallow them, and a reset would act as a drop.
            verdict = f"tcp flags & rst == 0 {verdict}"
        return verdict

    @staticmethod
    def _port_set(ports: int | Iterable[int]) -> str:
        return "{ " + ", ".join(str(p) for p in ([ports] if isinstance(ports, int) else ports)) + " }"

    def _rules_for(self, src: IPAddress, dst: IPAddress, ports: int | Iterable[int] | None, mode: CutMode, loss: int) -> list[str]:
        """The rules cutting packets from ``src`` to ``dst`` on ``ports`` (all, if None)."""
        match, verdict = f"ip saddr {src} ip daddr {dst}", self._verdict(mode, loss)
        if ports is None:
            return [f"{match} {verdict}"]
        # Both the packets to the peer's listening ports and the ones the peer
        # sends from them, so that connections in either direction are cut.
        port_set = self._port_set(ports)
        return [f"{match} tcp dport {port_set} {verdict}", f"{match} tcp sport {port_set} {verdict}"]

    def _rules_around(self, node: IPAddress, ports: int | Iterable[int] | None, mode: CutMode, loss: int) -> list[str]:
        """The rules cutting packets from anywhere to ``node`` and back, on ``ports`` of ``node`` (all, if None)."""
        verdict = self._verdict(mode, loss)
        if ports is None:
            return [f"ip daddr {node} {verdict}", f"ip saddr {node} {verdict}"]
        port_set = self._port_set(ports)
        return [f"ip daddr {node} tcp dport {port_set} {verdict}", f"ip saddr {node} tcp sport {port_set} {verdict}"]

    async def cut(self, src: IPAddress | Iterable[IPAddress], dst: IPAddress | Iterable[IPAddress], *,
                  ports: int | Iterable[int] | None = INTERNODE_PORT, mode: CutMode = "drop", loss: int = 100) -> None:
        """Stop packets from every ``src`` address to every ``dst`` address.

        Packets the other way still arrive, so ``dst`` keeps hearing from ``src`` while ``src`` hears
        nothing back: an asymmetric link. No TCP connection between the two survives it in either
        direction, since each needs packets both ways.
        """
        for s in self._addresses(src):
            for d in self._addresses(dst):
                self._rules.extend(self._rules_for(s, d, ports, mode, loss))
        await self._apply()

    async def partition(self, side_a: IPAddress | Iterable[IPAddress], side_b: IPAddress | Iterable[IPAddress], *,
                        ports: int | Iterable[int] | None = INTERNODE_PORT, mode: CutMode = "drop", loss: int = 100) -> None:
        """Cut both directions between the two sides."""
        a, b = self._addresses(side_a), self._addresses(side_b)
        for s in a:
            for d in b:
                self._rules.extend(self._rules_for(s, d, ports, mode, loss))
                self._rules.extend(self._rules_for(d, s, ports, mode, loss))
        await self._apply()

    async def isolate(self, node: IPAddress, peers: Iterable[IPAddress], *,
                      ports: int | Iterable[int] | None = INTERNODE_PORT, mode: CutMode = "drop", loss: int = 100) -> None:
        """Cut ``node`` off from all ``peers`` in both directions."""
        await self.partition(node, peers, ports=ports, mode=mode, loss=loss)

    async def block_clients(self, node: IPAddress, *, ports: Iterable[int] = CLIENT_PORTS, mode: CutMode = "drop",
                            loss: int = 100) -> None:
        """Cut ``node`` off from every client, wherever it connects from; its peers and the REST API are untouched."""
        self._rules.extend(self._rules_around(node, ports, mode, loss))
        await self._apply()

    async def blackout(self, node: IPAddress, *, mode: CutMode = "drop", loss: int = 100) -> None:
        """Cut all traffic to and from ``node``: peers, clients and the REST API alike."""
        self._rules.extend(self._rules_around(node, None, mode, loss))
        await self._apply()

    async def heal(self) -> None:
        """Remove every cut."""
        self._rules.clear()
        # A table that was never created is not an error here.
        await self._nft(f"delete table ip {self.table}", check=False)

    async def _apply(self) -> None:
        # The whole table is replaced atomically, so there is no window with a
        # partial rule set between two cuts.
        rules = "\n".join(f"        {rule}" for rule in self._rules)
        ruleset = (f"table ip {self.table} {{}}\n"
                   f"flush table ip {self.table}\n"
                   f"table ip {self.table} {{\n"
                   f"    chain input {{\n"
                   f"        type filter hook input priority 0; policy accept;\n"
                   f"{rules}\n"
                   f"    }}\n"
                   f"}}\n")
        logger.info("Installing nft ruleset for %s:\n%s", self.table, ruleset)
        await self._nft("-f", "-", stdin=ruleset)

    async def _nft(self, *args: str, stdin: str | None = None, check: bool = True) -> None:
        proc = await asyncio.create_subprocess_exec(
            "nft", *args,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        out, _ = await proc.communicate(stdin.encode() if stdin is not None else None)
        if check and proc.returncode != 0:
            raise RuntimeError(f"nft {' '.join(args)} failed ({proc.returncode}): {out.decode()}")
