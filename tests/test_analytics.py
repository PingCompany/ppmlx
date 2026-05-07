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

    def fake_post(*args, **kwargs):
        called["value"] = True
        raise AssertionError("_post_capture should not be called when DNT=1")

    monkeypatch.setattr(config_module, "load_config", lambda: _cfg())
    monkeypatch.setattr("ppmlx.analytics._post_capture", fake_post)
    monkeypatch.setenv("DNT", "1")

    assert track("serve_started") is False
    assert called["value"] is False


def test_track_posts_minimal_posthog_event(monkeypatch, tmp_home):
    from ppmlx import config as config_module
    from ppmlx.analytics import track

    captured = {}

    def fake_post_capture(host, project_api_key, event, properties):
        captured["host"] = host
        captured["project_api_key"] = project_api_key
        captured["event"] = event
        captured["properties"] = properties
        return True

    monkeypatch.setattr(config_module, "load_config", lambda: _cfg())
    monkeypatch.setattr("ppmlx.analytics._post_capture", fake_post_capture)
    monkeypatch.setattr("ppmlx.analytics._anonymous_distinct_id", lambda: "ppmlx-test-id")

    ok = track("serve_started", {"interactive": True, "ignored": "secret-string"}, context="server")

    assert ok is True
    assert captured["host"] == "https://stats.example.com"
    assert captured["project_api_key"] == "phc_test_123"
    assert captured["event"] == "serve_started"
    properties = captured["properties"]
    assert properties["distinct_id"] == "ppmlx-test-id"
    assert properties["interactive"] is True
    assert properties["context_server"] is True
    assert properties["$process_person_profile"] is False
    assert properties["platform"]
    assert properties["arch"]
    assert "ignored" not in properties
