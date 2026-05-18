"""Tests for checkpointed quality benchmark pipeline."""
from __future__ import annotations

import json

from ppmlx.quality_bench import build_quality_probes, classify_probe, run_quality_bench


def _messages():
    messages = [{"role": "system", "content": "You are helpful."}]
    for idx in range(6):
        messages.extend([
            {"role": "user", "content": f"Question {idx}: confirm app_{idx}.py changed, tests passed, and next run validation."},
            {"role": "assistant", "content": f"File app_{idx}.py changed and tests passed. Next run validation."},
        ])
    return messages


def test_build_quality_probes_uses_holdout_without_expected_leak():
    probes, skipped = build_quality_probes(_messages(), split=0.6, max_probes=2)

    assert len(probes) == 2
    assert skipped == []
    assert probes[0].user_message["role"] == "user"
    assert "File" in probes[0].expected_answer
    rendered_prefix = "\n".join(str(message.get("content", "")) for message in [*probes[0].prefix_messages, probes[0].user_message])
    assert probes[0].expected_answer not in rendered_prefix


def test_quality_bench_with_fake_responder(tmp_path):
    path = tmp_path / "pi.jsonl"
    lines = []
    for message in _messages():
        lines.append(json.dumps({"type": "message", "message": {"role": message["role"], "content": [{"type": "text", "text": message["content"]}]}}))
    path.write_text("\n".join(lines) + "\n")

    def responder(messages, max_tokens, metadata):
        user = next(message for message in reversed(messages) if message["role"] == "user")
        idx = user["content"].split()[1].rstrip(":")
        answer = f"File app_{idx}.py changed and tests passed. Next run validation."
        return answer, {"prompt_tokens": 100, "completion_tokens": 20}, 0.01

    report = run_quality_bench(
        path=path,
        source="pi",
        base_url="http://unused/v1",
        model="test-model",
        split=0.6,
        max_probes=2,
        responder=responder,
    )

    data = report.to_dict()
    assert data["summary"]["probes"] == 2
    assert data["summary"]["passed"] == 2
    assert data["summary"]["wrong_facts_total"] == 0
    assert data["summary"]["skipped"] == 0
    assert data["summary"]["failure_buckets"] == {"passed": 2}
    assert data["probes"][0]["context_fact_coverage"] == 1.0
    assert "graph_hot" in data["probes"][0]["ablations"]


def test_quality_bench_skips_oracle_facts_missing_from_context(tmp_path):
    path = tmp_path / "pi.jsonl"
    messages = [{"role": "system", "content": "You are helpful."}]
    for idx in range(5):
        messages.extend([
            {"role": "user", "content": f"Question {idx}: what changed?"},
            {"role": "assistant", "content": f"File future_{idx}.py changed and tests passed."},
        ])
    path.write_text("\n".join(
        json.dumps({"type": "message", "message": {"role": message["role"], "content": [{"type": "text", "text": message["content"]}]}})
        for message in messages
    ) + "\n")

    def responder(messages, max_tokens, metadata):
        user = next(message for message in reversed(messages) if message["role"] == "user")
        idx = user["content"].split()[1].rstrip(":")
        return f"File future_{idx}.py changed and tests passed.", {"prompt_tokens": 100, "completion_tokens": 12}, 0.01

    report = run_quality_bench(
        path=path,
        source="pi",
        base_url="http://unused/v1",
        model="test-model",
        split=0.6,
        max_probes=1,
        responder=responder,
    )

    data = report.to_dict()
    assert data["summary"]["probes"] == 0
    assert data["summary"]["skipped_by_type"] == {"oracle_unavailable_in_context": 1}
    assert data["skipped_probes"][0]["reason"] == "no expected-answer oracle facts are recoverable from compact/replay context"


def test_probe_classifier_skips_tool_and_code_action_turns():
    tool_type, _ = classify_probe(
        "Find the file and edit it",
        "Tool call bash {\"command\": \"rg payment src\"}",
    )
    code_type, _ = classify_probe(
        "Fix the failing pytest in the repo",
        "I'll inspect the failing test and patch the file.",
    )
    delegated_type, _ = classify_probe(
        "zrób to za mnie",
        "I will run uv sync and reinstall the tool.",
    )
    feedback_type, _ = classify_probe(
        "z HF_HUB_DISABLE_XET=1 działa płynniej",
        "Great, that confirms Xet progress callbacks are the likely cause.",
    )
    answer_type, _ = classify_probe(
        "What changed?",
        "The config file changed and tests passed. Next run validation.",
    )

    assert tool_type == "tool_action_required"
    assert code_type == "code_edit_required"
    assert delegated_type == "code_edit_required"
    assert feedback_type == "ambiguous_skip"
    assert answer_type == "answerable_text"
