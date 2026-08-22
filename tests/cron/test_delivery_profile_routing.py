from pathlib import Path
from unittest.mock import patch

from cron import scheduler


def test_named_delivery_profile_scopes_send_and_drops_foreign_adapter(tmp_path):
    secret_scope = object()
    secret_token = object()
    home_token = object()

    with (
        patch.object(scheduler, "_cron_delivery_profile", return_value="telegram-lite"),
        patch("hermes_cli.profiles.resolve_profile_env", return_value=tmp_path),
        patch("agent.secret_scope.build_profile_secret_scope", return_value=secret_scope),
        patch("agent.secret_scope.set_secret_scope", return_value=secret_token) as set_secret,
        patch("agent.secret_scope.reset_secret_scope") as reset_secret,
        patch("hermes_constants.set_hermes_home_override", return_value=home_token) as set_home,
        patch("hermes_constants.reset_hermes_home_override") as reset_home,
        patch.object(
            scheduler,
            "_deliver_result_active_profile",
            return_value=None,
        ) as deliver,
    ):
        assert scheduler._deliver_result(
            {"id": "job-1"}, "hello", adapters={"foreign": object()}, loop=object()
        ) is None

    set_home.assert_called_once_with(Path(tmp_path))
    set_secret.assert_called_once_with(secret_scope)
    deliver.assert_called_once_with(
        {"id": "job-1"}, "hello", adapters=None, loop=None
    )
    reset_secret.assert_called_once_with(secret_token)
    reset_home.assert_called_once_with(home_token)


def test_missing_delivery_profile_fails_closed():
    with (
        patch.object(scheduler, "_cron_delivery_profile", return_value="missing"),
        patch(
            "hermes_cli.profiles.resolve_profile_env",
            side_effect=FileNotFoundError("not found"),
        ),
        patch.object(scheduler, "_deliver_result_active_profile") as deliver,
    ):
        result = scheduler._deliver_result({"id": "job-1"}, "hello")

    assert "delivery profile 'missing' is unavailable" in result
    deliver.assert_not_called()


def test_default_delivery_profile_preserves_live_adapter_path():
    adapters = {"telegram": object()}
    loop = object()
    with (
        patch.object(scheduler, "_cron_delivery_profile", return_value="default"),
        patch.object(
            scheduler,
            "_deliver_result_active_profile",
            return_value=None,
        ) as deliver,
    ):
        assert scheduler._deliver_result(
            {"id": "job-1"}, "hello", adapters=adapters, loop=loop
        ) is None

    deliver.assert_called_once_with(
        {"id": "job-1"}, "hello", adapters=adapters, loop=loop
    )
