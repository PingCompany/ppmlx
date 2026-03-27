from __future__ import annotations

from types import SimpleNamespace


def _cfg(**overrides):
    defaults = {
        "enabled": True,
        "provider": "posthog",
        "host": "https://stats.example.com",
        "project_api_key": "phc_test_123",
        "respect_do_not_track": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(analytics=SimpleNamespace(**defaults))


def test_track_skips_when_sink_not_configured(monkeypatch):
    from ppmlx import config as config_module
    from ppmlx.analytics import track

    monkeypatch.setattr(config_module, "load_config", lambda: _cfg(host="", project_api_key=""))
    assert track("serve_started") is False


def test_track_respects_do_not_track(monkeypatch):
    from ppmlx import config as config_module
    from ppmlx.analytics import track

    called = {"value": False}

    def fake_client(*args, **kwargs):
        called["value"] = True
        raise AssertionError("_get_client should not be called when DNT=1")

    monkeypatch.setattr(config_module, "load_config", lambda: _cfg())
    monkeypatch.setattr("ppmlx.analytics._get_client", fake_client)
    monkeypatch.setenv("DNT", "1")

    assert track("serve_started") is False
    assert called["value"] is False


def test_track_posts_minimal_posthog_event(monkeypatch):
    from ppmlx import config as config_module
    from ppmlx.analytics import track

    captured = {}

    class FakeClient:
        def capture(self, event, **kwargs):
            captured["event"] = event
            captured["kwargs"] = kwargs
            return "evt_123"

    monkeypatch.setattr(config_module, "load_config", lambda: _cfg())
    monkeypatch.setattr("ppmlx.analytics._get_client", lambda host, project_api_key: FakeClient())

    ok = track("serve_started", {"interactive": True, "ignored": "secret-string"}, context="server")

    assert ok is True
    assert captured["event"] == "serve_started"
    properties = captured["kwargs"]["properties"]
    assert properties["interactive"] is True
    assert properties["$process_person_profile"] is False
    assert properties["platform"]
    assert properties["arch"]
    assert "ignored" not in properties
