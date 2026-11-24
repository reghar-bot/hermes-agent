"""A recycled worker PID is never mistaken for our worker.

``tasks.worker_pid`` survives a reboot; the number can then belong to an unrelated process. Every
liveness decision (extend/defer the claim) and every kill (SIGTERM/SIGKILL on timeout or reclaim)
must require the spawn-time start fingerprint to match, never bare PID existence.
"""

import os
import signal
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


def _claimed_running(conn, *, pid: int, started_at, max_runtime=None) -> str:
    tid = kb.create_task(conn, title="job", assignee="worker", max_runtime_seconds=max_runtime)
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, pid)
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_started_at = ?, started_at = ?, claim_expires = ? WHERE id = ?",
                     (started_at, old, old, tid))
        conn.execute("UPDATE task_runs SET started_at = ? WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                     (old, tid))
    return tid


def test_recycled_pid_is_reclaimed_without_being_signalled(board):
    """Our own live PID with a foreign fingerprint models a post-reboot recycle: the claim is released
    (dead worker), no signal is sent, and max-runtime enforcement does not SIGTERM the stranger either."""
    conn = board
    killed = []
    stranger_fingerprint = 1  # no live process started at tick 1
    tid = _claimed_running(conn, pid=os.getpid(), started_at=stranger_fingerprint, max_runtime=1)

    assert kbd._worker_alive(os.getpid(), stranger_fingerprint) is False
    assert tid in kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: killed.append((pid, sig)))
    assert killed == []
    task = kb.get_task(conn, tid)
    assert task.status == "ready" and task.worker_pid is None

    tid2 = _claimed_running(conn, pid=os.getpid(), started_at=stranger_fingerprint)
    assert kb.release_stale_claims(conn, signal_fn=lambda pid, sig: killed.append((pid, sig))) == 1
    assert killed == []
    assert kb.get_task(conn, tid2).status == "ready"


def test_matching_fingerprint_keeps_the_live_worker(board):
    """The same PID with ITS OWN fingerprint (recorded at spawn) is our worker: the expired claim is
    extended rather than reclaimed, and the timeout path signals it."""
    from gateway.status import get_process_start_time

    conn = board
    killed = []
    tid = _claimed_running(conn, pid=os.getpid(), started_at=get_process_start_time(os.getpid()))
    assert kbd._worker_alive(os.getpid(), get_process_start_time(os.getpid())) is True
    assert kb.release_stale_claims(conn) == 0
    assert kb.get_task(conn, tid).status == "running"
    kinds = [e.kind for e in kb.list_events(conn, tid)]
    assert "claim_extended" in kinds

    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET max_runtime_seconds = 1 WHERE id = ?", (tid,))
    kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: killed.append((pid, sig)))
    assert killed and killed[0] == (os.getpid(), signal.SIGTERM)


def test_same_pid_and_start_tick_on_another_boot_is_foreign(board, monkeypatch):
    """A row that survived a reboot: the PID AND the boot-relative start tick both match a process on
    this boot (the Linux start time is clock ticks since boot, so that recurs), but the persisted
    instantiation epoch does not. The worker is foreign: claim released, zero signals."""
    from gateway import drain_control

    conn = board
    killed = []
    live_fingerprint = kbd._process_fingerprint(os.getpid())
    assert live_fingerprint is not None and live_fingerprint.split("|", 1)[1] == str(
        __import__("gateway.status", fromlist=["x"]).get_process_start_time(os.getpid()))
    tid = _claimed_running(conn, pid=os.getpid(), started_at=live_fingerprint, max_runtime=1)
    assert kbd._worker_alive(os.getpid(), live_fingerprint) is True

    # Same PID, same start tick, different boot identity.
    other_boot = "deadbeef-boot:1|" + live_fingerprint.split("|", 1)[1]
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_started_at = ? WHERE id = ?", (other_boot, tid))
    assert kbd._worker_alive(os.getpid(), other_boot) is False
    assert tid in kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: killed.append((pid, sig)))
    assert killed == []
    task = kb.get_task(conn, tid)
    assert task.status == "ready" and task.worker_pid is None

    # The same value re-derived on THIS boot still identifies our worker (the witness is stable
    # within a boot, unlike the recorded epoch of a previous one).
    drain_control.current_instantiation_epoch.cache_clear()
    assert kbd._process_fingerprint(os.getpid()) == live_fingerprint


def test_unverified_fingerprint_capture_never_authorizes_a_signal(board, monkeypatch):
    """Fingerprint capture fails for a new spawn: the row is NOT a legacy NULL row. A live PID under
    it is never SIGTERM/SIGKILLed by any reclaim/timeout path, and the claim is held (not released
    beside the live process); once the PID is gone the claim is reclaimed normally."""
    import gateway.status as status

    conn = board
    killed = []
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: None)
    tid = kb.create_task(conn, title="job", assignee="worker", max_runtime_seconds=1)
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, os.getpid())
    row = conn.execute("SELECT worker_started_at FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["worker_started_at"] == kbd.UNVERIFIED_WORKER_FINGERPRINT
    monkeypatch.undo()
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET started_at = ?, claim_expires = ? WHERE id = ?", (old, old, tid))
        conn.execute("UPDATE task_runs SET started_at = ? WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                     (old, tid))

    sig = lambda pid, s: killed.append((pid, s))  # noqa: E731
    assert kbd.enforce_max_runtime(conn, signal_fn=sig) == []
    assert kb.release_stale_claims(conn, signal_fn=sig) == 0
    assert killed == []
    assert kb.get_task(conn, tid).status == "running"
    # An explicit operator reclaim releases the claim (human override) but still sends nothing.
    assert kb.reclaim_task(conn, tid, reason="operator", signal_fn=sig) is True
    assert killed == []

    # The process is gone (a dead PID): the row is reclaimed like any dead worker, still no signal.
    tid2 = kb.create_task(conn, title="job2", assignee="worker")
    kb.claim_task(conn, tid2)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_pid = ?, worker_started_at = ?, claim_expires = ? WHERE id = ?",
                     (os.getpid(), kbd.UNVERIFIED_WORKER_FINGERPRINT, old, tid2))
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    assert kb.release_stale_claims(conn, signal_fn=sig) == 1
    assert killed == [] and kb.get_task(conn, tid2).status == "ready"


# --- macOS: psutil's adjusted create_time differs per reading process ------------------------------
# Measured on a real Mac (psutil 7.2.2): the raw kernel create_time of a process is FIXED, kern.boottime
# drifts (2 s in 30 min), and psutil's create_time() shifts the raw value by the reading process's
# (INIT_BOOT_TIME - boot_time()), so two processes disagree about the same live PID.

_RAW_CTIME = 1_800_000_000.34
_BOOT = 1_799_000_000.0


class _FakeMacPsutil:
    """Shaped like psutil 7.2.2's ``_psosx``: ``cext.proc_kinfo_oneshot`` + ``kinfo_proc_map``."""

    kinfo_proc_map = {"ppid": 0, "ctime": 1}

    class _Cext:
        def __init__(self, world):
            self.world = world

        def proc_kinfo_oneshot(self, pid):
            if self.world.raw_error:
                raise OSError("boom")
            return (1, self.world.raw_ctime)

    def __init__(self, world, real):
        self.cext = self._Cext(world)
        self._real = real  # everything else psutil asks of its platform module (pid_exists, ...)

    def __getattr__(self, name):
        return getattr(self._real, name)


class _World:
    """Fake macOS host. The raw create_time never changes; ``reader(init_boot, boot_now)`` selects which
    process is reading (its psutil import-time boottime and the current kern.boottime)."""

    def __init__(self):
        self.raw_ctime = _RAW_CTIME
        self.raw_error = False
        self.init_boot = _BOOT
        self.boot_now = _BOOT

    def reader(self, init_boot, boot_now):
        self.init_boot, self.boot_now = float(init_boot), float(boot_now)

    def adjusted(self):  # psutil._psosx.adjust_proc_create_time, verbatim logic
        diff = self.init_boot - self.boot_now
        if abs(diff) < 1:
            return self.raw_ctime
        return self.raw_ctime - diff if diff < 0 else self.raw_ctime + diff


@pytest.fixture
def darwin_world(monkeypatch):
    import gateway.status as status
    import psutil

    w = _World()
    monkeypatch.setattr(kbd, "_on_darwin", lambda: True, raising=False)
    monkeypatch.setattr(psutil, "_psplatform", _FakeMacPsutil(w, psutil._psplatform), raising=False)
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: round(w.adjusted() * 100))
    monkeypatch.setattr("gateway.drain_control.current_instantiation_epoch", lambda: "")
    return w


@pytest.mark.parametrize("drift", [1, 5, 30])
def test_darwin_boottime_drift_does_not_recycle_a_live_worker(board, darwin_world, drift):
    """Recorded by a process that imported psutil at boottime B; checked later by one that imported it
    ``drift`` s after kern.boottime moved. Same live PID: it must not be reclaimed (a duplicate worker
    would be spawned)."""
    conn = board
    recorded = kbd._process_fingerprint(os.getpid())
    assert recorded == f"|r:{round(_RAW_CTIME * 100)}"
    tid = _claimed_running(conn, pid=os.getpid(), started_at=recorded)
    darwin_world.reader(_BOOT + drift, _BOOT + drift)  # a fresh process, boottime has drifted
    assert kbd._pid_recycled(os.getpid(), recorded) is False
    darwin_world.reader(_BOOT, _BOOT + drift)  # the recording process itself, still running
    assert kbd._pid_recycled(os.getpid(), recorded) is False
    assert kbd._worker_alive(os.getpid(), recorded) is True
    sweep = kbd._reclaim_dead_workers(conn)
    assert sweep.crash_details == []
    assert kb.get_task(conn, tid).status == "running"


def test_darwin_raw_start_far_off_is_recycled(darwin_world):
    """A different process on the same PID (started 10 s later) is foreign."""
    recorded = kbd._process_fingerprint(os.getpid())
    darwin_world.raw_ctime += 10
    assert kbd._pid_recycled(os.getpid(), recorded) is True


def test_epoch_mismatch_is_recycled_even_when_start_matches(darwin_world, monkeypatch):
    recorded = kbd._process_fingerprint(os.getpid())
    monkeypatch.setattr("gateway.drain_control.current_instantiation_epoch", lambda: "other-boot:1")
    assert kbd._pid_recycled(os.getpid(), recorded) is True


def test_darwin_unreadable_current_reading_is_recycled(darwin_world):
    recorded = kbd._process_fingerprint(os.getpid())
    darwin_world.raw_error = True
    assert kbd._pid_recycled(os.getpid(), recorded) is True


@pytest.mark.parametrize("which,delta_s,recycled", [
    ("adjusted", 0.0, False), ("adjusted", 1.5, False), ("adjusted", 10, True),
    ("raw", 1.5, False), ("raw", 10, True),
])
def test_legacy_wallclock_row_tolerates_adjusted_or_raw_reading(darwin_world, which, delta_s, recycled):
    """Rows written before the raw format: ``|<wall-clock cs>``. Accepted within 2 s of either psutil's
    adjusted reading or the raw one (the writer's adjustment is unknown)."""
    darwin_world.reader(_BOOT + 3, _BOOT)  # we read an adjusted value 3 s off the raw one
    base = darwin_world.adjusted() if which == "adjusted" else darwin_world.raw_ctime
    assert abs(darwin_world.adjusted() - darwin_world.raw_ctime) >= 2.9
    assert kbd._pid_recycled(os.getpid(), f"|{round((base + delta_s) * 100)}") is recycled


def test_raw_create_time_wrapper_reads_the_psutil_7_2_2_attribute_path(darwin_world):
    """Exercises ``psutil._psplatform.cext.proc_kinfo_oneshot(pid)[kinfo_proc_map['ctime']]`` against a
    fake shaped like psutil 7.2.2's ``_psosx`` (runs on every host)."""
    assert kbd._darwin_raw_create_time(os.getpid()) == _RAW_CTIME


@pytest.mark.platforms("macos")
def test_private_psutil_api_we_rely_on_exists_on_macos():
    """On a real Mac the private names ``_darwin_raw_create_time`` reads must exist; a psutil upgrade
    that drops them silently degrades to the legacy fingerprint, and this says so."""
    import psutil

    assert "ctime" in psutil._psplatform.kinfo_proc_map
    raw = kbd._darwin_raw_create_time(os.getpid())
    assert raw is not None and raw > 0


@pytest.mark.parametrize("breakage", ["missing", "raises"])
def test_private_psutil_failure_falls_back_to_legacy_format(darwin_world, monkeypatch, breakage):
    import gateway.status as status
    import psutil

    if breakage == "missing":
        monkeypatch.delattr(psutil, "_psplatform")
    else:
        darwin_world.raw_error = True
    assert kbd._darwin_raw_create_time(os.getpid()) is None
    fp = kbd._process_fingerprint(os.getpid())
    assert fp is not None and fp.split("|", 1)[1] == str(status.get_process_start_time(os.getpid()))
    assert "r:" not in fp
    assert kbd._pid_recycled(os.getpid(), fp) is False


def test_retried_run_gets_launch_grace_from_its_own_start(board, monkeypatch):
    """tasks.started_at is the FIRST claim ever; a retry whose worker PID is not yet visible must still
    get the launch grace measured from the current run's start, not be reclaimed as crashed."""
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "30")
    conn = board
    tid = _claimed_running(conn, pid=os.getpid(), started_at=1)  # foreign fingerprint = "dead" worker
    assert kbd._reclaim_dead_workers(conn).crash_details  # old run: outside grace, reclaimed
    kb.claim_task(conn, tid)  # retry: new run, tasks.started_at stays hours old
    kbd._set_worker_pid(conn, tid, os.getpid())
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_started_at = 1, started_at = ? WHERE id = ?",
                     (int(time.time()) - 3600, tid))
    assert kb.get_task(conn, tid).status == "running"
    assert kbd._reclaim_dead_workers(conn).crash_details == []
    assert kb.get_task(conn, tid).status == "running"
