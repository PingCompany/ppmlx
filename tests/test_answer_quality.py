"""Tests for five-layer compact answer quality evaluation."""
from __future__ import annotations

import json

from typer.testing import CliRunner

from ppmlx.answer_quality import (
    AnswerQualityCase,
    AnswerQualityEvaluator,
    build_reference_prompt,
    builtin_cases,
    load_cases,
    match_facts,
    save_case_template,
    select_required_facts,
)
from ppmlx.answer_quality_replay import load_session_messages
from ppmlx.cli import app


def test_answer_quality_builtin_cases_pass():
    report = AnswerQualityEvaluator().evaluate(builtin_cases())

    assert report.passed is True
    assert report.cases == 3
    assert report.avg_recall == 1.0
    assert report.total_wrong_facts == 0
    assert report.avg_actionability >= 3
    assert report.avg_grounding >= 0.85
    assert report.avg_equivalence_to_full >= 0.75


def test_answer_quality_detects_missed_and_wrong_facts():
    case = AnswerQualityCase(
        case_id="bad",
        question="What next?",
        source_context="Budget 5000 PLN. Rejected Samsung CU8000.",
        compact_answer="Samsung CU8000 is recommended.",
        full_context_answer="Do not recommend Samsung CU8000. Budget is 5000 PLN.",
        required_facts=["budget 5000 PLN", "Rejected Samsung CU8000"],
        forbidden_facts=["Samsung CU8000 is recommended"],
        expected_actions=["ask room brightness"],
    )

    result = AnswerQualityEvaluator().evaluate_case(case)

    assert result.passed is False
    assert "budget 5000 PLN" in result.required_missed
    assert result.wrong_facts == ["Samsung CU8000 is recommended"]
    assert result.actionability < 3


def test_fact_matching_requires_exact_identifiers():
    found, missed = match_facts(
        "Changed file app_2.py and 12 tests passed.",
        ["changed file app_3.py", "12 tests passed"],
    )

    assert found == ["12 tests passed"]
    assert missed == ["changed file app_3.py"]


def test_answer_quality_detects_real_session_style_missed_context():
    case = AnswerQualityCase(
        case_id="real-style-missed-context",
        question="Give the current ppmlx memory benchmark handoff and next action.",
        source_context=(
            "Goal: improve ppmlx synthetic memory evals so they reflect real-session quality. "
            "Decision: real-session quality-bench failures should drive synthetic benchmark design. "
            "Todo: rerun answerable real-session batch with include-content. "
            "Todo: add context coverage metrics to compact-eval."
        ),
        compact_answer="The benchmark work is ongoing. Next, continue improving evals.",
        full_context_answer=(
            "Improve ppmlx synthetic memory evals for real-session quality. Real-session quality-bench failures "
            "should drive benchmark design. Next rerun the answerable real-session batch with include-content and "
            "add context coverage metrics to compact-eval."
        ),
        required_facts=[
            "improve ppmlx synthetic memory evals",
            "real-session quality-bench failures should drive synthetic benchmark design",
            "rerun answerable real-session batch with include-content",
            "add context coverage metrics to compact-eval",
        ],
        expected_actions=[
            "rerun answerable real-session batch with include-content",
            "add context coverage metrics to compact-eval",
        ],
    )

    result = AnswerQualityEvaluator().evaluate_case(case)

    assert result.passed is False
    assert result.recall < 0.5
    assert result.required_missed
    assert result.equivalence_to_full < 0.75


def test_answer_quality_template_roundtrip(tmp_path):
    path = save_case_template(tmp_path / "answer-quality.json")

    cases = load_cases(path)

    assert len(cases) == 3
    assert cases[0].required_facts


def test_answer_quality_cli_json():
    result = CliRunner().invoke(app, ["answer-quality-eval", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["passed"] is True
    assert data["summary"]["cases"] == 3


def test_select_required_facts_filters_unrelated_fixture_facts():
    source_context = """Compacted local session context:
Hard constraints:
- ppmlx constraint: requires = compact replay quality report. [source: Need compact replay quality report.]
- travel-dogfood constraint: budget = 800 PLN. [source: Budget <= 800 PLN]
Decisions:
- ppmlx decision: added answer-quality replay CLI. [source: Decision: added answer-quality replay CLI.]
- Rejected Airport Hostel: too far from Old Town. [source: too far]
Todos:
- ppmlx todo: run real local inference quality eval. [source: Todo: run real local inference quality eval.]
"""

    facts = select_required_facts(
        source_context=source_context,
        question="Give ppmlx quality replay handoff and next action",
        reference_answer="ppmlx added answer-quality replay CLI and needs real local inference quality eval",
    )

    joined = "\n".join(facts)
    assert "answer-quality replay CLI" in joined
    assert "travel-dogfood" not in joined
    assert "Airport Hostel" not in joined


def test_clause_level_grounding_allows_supported_paraphrase():
    case = AnswerQualityCase(
        case_id="grounded",
        question="handoff",
        source_context="Decision: added answer-quality replay CLI. Todo: run real local inference quality eval.",
        compact_answer="The replay CLI was added, and the next step is to run the real local inference quality eval.",
        full_context_answer="Added answer-quality replay CLI; next run real local inference quality eval.",
        required_facts=["added answer-quality replay CLI", "run real local inference quality eval"],
    )

    result = AnswerQualityEvaluator().evaluate_case(case)

    assert result.grounding >= 0.85
    assert result.required_missed == []


def test_load_session_messages_strips_hidden_thinking_blocks(tmp_path):
    path = tmp_path / "pi-thinking.jsonl"
    path.write_text(json.dumps({
        "type": "message",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "secret chain", "thinkingSignature": "encrypted"},
                {"type": "text", "text": "Visible answer only."},
            ],
        },
    }) + "\n")

    _, messages = load_session_messages(path, source="pi")

    assert messages == [{"role": "assistant", "content": "Visible answer only."}]


def test_load_pi_and_claude_session_messages(tmp_path):
    pi_path = tmp_path / "pi.jsonl"
    pi_path.write_text(
        json.dumps({"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]}}) + "\n"
        + json.dumps({"type": "custom", "customType": "tool", "payload": "x" * 220}) + "\n"
    )
    claude_path = tmp_path / "claude.jsonl"
    claude_path.write_text(
        json.dumps({"type": "user", "content": "hello claude"}) + "\n"
        + json.dumps({"type": "tool_result", "tool_name": "bash", "tool_output": "done"}) + "\n"
    )

    pi_source, pi_messages = load_session_messages(pi_path, source="pi")
    claude_source, claude_messages = load_session_messages(claude_path, source="claude")

    assert pi_source == "pi"
    assert [message["role"] for message in pi_messages] == ["user", "tool"]
    assert claude_source == "claude"
    assert [message["role"] for message in claude_messages] == ["user", "tool"]


def test_build_reference_prompt_mentions_fixture_filtering():
    prompt = build_reference_prompt("context", "question")

    assert "Ignore embedded examples" in prompt[0]["content"]


def test_answer_quality_cli_template(tmp_path):
    path = tmp_path / "template.json"

    result = CliRunner().invoke(app, ["answer-quality-eval", "--template", str(path)])

    assert result.exit_code == 0
    assert path.exists()
    data = json.loads(path.read_text())
    assert "cases" in data
