"""Behavior tests for profile/skill compatibility before Kanban spawn."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _install_skill(
    home: Path,
    profile: str,
    skill: str,
    *,
    environments: list[str] | None = None,
) -> None:
    skill_dir = home / "profiles" / profile / "skills" / skill
    skill_dir.mkdir(parents=True, exist_ok=True)
    environment_line = (
        f"environments: [{', '.join(environments)}]\n" if environments else ""
    )
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {skill}\ndescription: Test routing skill.\n"
        f"{environment_line}---\n\n"
        f"# {skill}\n",
        encoding="utf-8",
    )


def test_missing_ready_skill_blocks_before_spawn_without_failure_retry(
    kanban_home: Path,
) -> None:
    _install_skill(kanban_home, "worker", "available-skill")
    spawned: list[str] = []

    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="needs unavailable capability",
            assignee="worker",
            skills=["missing-skill"],
        )
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, _workspace: spawned.append(task.id),
        )
        task = kb.get_task(conn, task_id)

    assert spawned == []
    assert result.spawned == []
    assert result.capability_blocked == [
        (task_id, "worker", ["missing-skill"])
    ]
    assert task is not None
    assert task.status == "blocked"
    assert task.block_kind == "capability"
    assert task.consecutive_failures == 0


def test_available_ready_skill_spawns_normally(kanban_home: Path) -> None:
    _install_skill(kanban_home, "worker", "available-skill")
    spawned: list[str] = []

    def spawn(task, _workspace):
        spawned.append(task.id)
        return 4242

    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="has compatible capability",
            assignee="worker",
            skills=["available-skill"],
        )
        result = kb.dispatch_once(conn, spawn_fn=spawn)
        task = kb.get_task(conn, task_id)

    assert spawned == [task_id]
    assert [entry[0] for entry in result.spawned] == [task_id]
    assert result.capability_blocked == []
    assert task is not None
    assert task.status == "running"


def test_review_lane_preflights_mandatory_reviewer_skill(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_skill(kanban_home, "reviewer", "domain-review")
    monkeypatch.setattr(kb, "review_dispatch_enabled", lambda: True)
    spawned: list[str] = []

    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="review without reviewer bundle",
            assignee="reviewer",
            skills=["domain-review"],
        )
        implementation = kb.claim_task(conn, task_id)
        assert implementation is not None
        assert kb.request_review(
            conn,
            task_id,
            summary="ready for review",
            expected_run_id=implementation.current_run_id,
        )

        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, _workspace: spawned.append(task.id),
        )
        task = kb.get_task(conn, task_id)

    assert spawned == []
    assert result.spawned == []
    assert result.capability_blocked == [
        (task_id, "reviewer", ["sdlc-review"])
    ]
    assert task is not None
    assert task.status == "blocked"
    assert task.block_kind == "capability"
    assert task.consecutive_failures == 0


def test_review_lane_accepts_explicit_kanban_environment_skill(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_skill(kanban_home, "reviewer", "domain-review")
    _install_skill(
        kanban_home,
        "reviewer",
        "sdlc-review",
        environments=["kanban"],
    )
    monkeypatch.setattr(kb, "review_dispatch_enabled", lambda: True)
    captured: list[list[str]] = []

    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="review with reviewer bundle",
            assignee="reviewer",
            skills=["domain-review"],
        )
        implementation = kb.claim_task(conn, task_id)
        assert implementation is not None
        assert kb.request_review(
            conn,
            task_id,
            summary="ready for review",
            expected_run_id=implementation.current_run_id,
        )

        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, _workspace: captured.append(task.skills),
        )

    assert result.capability_blocked == []
    assert [entry[0] for entry in result.spawned] == [task_id]
    assert captured == [["domain-review", "sdlc-review"]]


def test_dry_run_reports_incompatibility_without_mutating_task(
    kanban_home: Path,
) -> None:
    _install_skill(kanban_home, "worker", "available-skill")

    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="dry-run incompatible capability",
            assignee="worker",
            skills=["missing-skill"],
        )
        result = kb.dispatch_once(conn, dry_run=True)
        task = kb.get_task(conn, task_id)

    assert result.capability_blocked == [
        (task_id, "worker", ["missing-skill"])
    ]
    assert result.spawned == []
    assert task is not None
    assert task.status == "ready"
