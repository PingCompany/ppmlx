"""Answer quality evaluation for compact-memory responses.

This module scores whether an answer produced from compact context preserves task
state and remains grounded.  The default judge is deterministic and local-only;
it is intentionally simple so it can run in CI without a model.  A later LLM
judge can consume the same case/result schema.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AnswerQualityCase:
    case_id: str
    question: str
    source_context: str
    compact_answer: str
    full_context_answer: str
    required_facts: list[str]
    forbidden_facts: list[str] = field(default_factory=list)
    expected_actions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnswerQualityCase":
        return cls(
            case_id=str(data["case_id"]),
            question=str(data.get("question") or ""),
            source_context=str(data.get("source_context") or ""),
            compact_answer=str(data.get("compact_answer") or ""),
            full_context_answer=str(data.get("full_context_answer") or ""),
            required_facts=[str(item) for item in data.get("required_facts", [])],
            forbidden_facts=[str(item) for item in data.get("forbidden_facts", [])],
            expected_actions=[str(item) for item in data.get("expected_actions", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "source_context": self.source_context,
            "compact_answer": self.compact_answer,
            "full_context_answer": self.full_context_answer,
            "required_facts": self.required_facts,
            "forbidden_facts": self.forbidden_facts,
            "expected_actions": self.expected_actions,
        }


@dataclass
class AnswerQualityResult:
    case_id: str
    passed: bool
    recall: float
    required_found: list[str]
    required_missed: list[str]
    wrong_facts: list[str]
    wrong_fact_count: int
    actionability: int
    actionability_reasons: list[str]
    grounding: float
    unsupported_terms: list[str]
    equivalence_to_full: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "recall": self.recall,
            "required_found": self.required_found,
            "required_missed": self.required_missed,
            "wrong_facts": self.wrong_facts,
            "wrong_fact_count": self.wrong_fact_count,
            "actionability": self.actionability,
            "actionability_reasons": self.actionability_reasons,
            "grounding": self.grounding,
            "unsupported_terms": self.unsupported_terms,
            "equivalence_to_full": self.equivalence_to_full,
            "notes": self.notes,
        }


@dataclass
class AnswerQualitySummary:
    timestamp: str
    passed: bool
    cases: int
    passed_cases: int
    avg_recall: float
    total_wrong_facts: int
    avg_actionability: float
    avg_grounding: float
    avg_equivalence_to_full: float
    results: list[AnswerQualityResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "passed": self.passed,
            "summary": {
                "cases": self.cases,
                "passed_cases": self.passed_cases,
                "avg_recall": self.avg_recall,
                "total_wrong_facts": self.total_wrong_facts,
                "avg_actionability": self.avg_actionability,
                "avg_grounding": self.avg_grounding,
                "avg_equivalence_to_full": self.avg_equivalence_to_full,
            },
            "results": [result.to_dict() for result in self.results],
        }


@dataclass
class AnswerQualityThresholds:
    min_recall: float = 0.85
    max_wrong_facts: int = 0
    min_actionability: int = 3
    min_grounding: float = 0.85
    min_equivalence_to_full: float = 0.75


class AnswerQualityEvaluator:
    """Deterministic five-layer quality judge for compact-memory answers."""

    def __init__(self, thresholds: AnswerQualityThresholds | None = None):
        self.thresholds = thresholds or AnswerQualityThresholds()

    def evaluate_case(self, case: AnswerQualityCase) -> AnswerQualityResult:
        required_found, required_missed = match_facts(case.compact_answer, case.required_facts)
        recall = _safe_ratio(len(required_found), len(case.required_facts))

        wrong_facts = [fact for fact in case.forbidden_facts if _contains_wrong_fact(case.compact_answer, fact)]
        actionability, action_reasons = _score_actionability(case.compact_answer, case.expected_actions)
        grounding, unsupported = _score_grounding(
            answer=case.compact_answer,
            source_context=case.source_context,
            required_facts=case.required_facts,
            forbidden_facts=case.forbidden_facts,
        )
        equivalence = _score_equivalence(
            compact_answer=case.compact_answer,
            full_context_answer=case.full_context_answer,
            required_facts=case.required_facts,
        )
        notes: list[str] = []
        if required_missed:
            notes.append("required facts missing from compact answer")
        if wrong_facts:
            notes.append("forbidden/wrong facts present")
        if unsupported:
            notes.append("answer contains unsupported salient terms")
        passed = (
            recall >= self.thresholds.min_recall
            and len(wrong_facts) <= self.thresholds.max_wrong_facts
            and actionability >= self.thresholds.min_actionability
            and grounding >= self.thresholds.min_grounding
            and equivalence >= self.thresholds.min_equivalence_to_full
        )
        return AnswerQualityResult(
            case_id=case.case_id,
            passed=passed,
            recall=round(recall, 4),
            required_found=required_found,
            required_missed=required_missed,
            wrong_facts=wrong_facts,
            wrong_fact_count=len(wrong_facts),
            actionability=actionability,
            actionability_reasons=action_reasons,
            grounding=round(grounding, 4),
            unsupported_terms=unsupported,
            equivalence_to_full=round(equivalence, 4),
            notes=notes,
        )

    def evaluate(self, cases: list[AnswerQualityCase]) -> AnswerQualitySummary:
        results = [self.evaluate_case(case) for case in cases]
        passed_cases = sum(1 for result in results if result.passed)
        return AnswerQualitySummary(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            passed=passed_cases == len(results),
            cases=len(results),
            passed_cases=passed_cases,
            avg_recall=_round_avg([result.recall for result in results]),
            total_wrong_facts=sum(result.wrong_fact_count for result in results),
            avg_actionability=_round_avg([float(result.actionability) for result in results]),
            avg_grounding=_round_avg([result.grounding for result in results]),
            avg_equivalence_to_full=_round_avg([result.equivalence_to_full for result in results]),
            results=results,
        )


# Built-in cases intentionally mirror compact-memory failure modes: missed state,
# wrong rejected option, low actionability, unsupported invention, and A/B drift.
def builtin_cases() -> list[AnswerQualityCase]:
    return [
        AnswerQualityCase(
            case_id="tv_handoff_good",
            question="Which TV should we pick and what is next?",
            source_context=(
                "Budget <= 5000 PLN. Need HDMI 2.1 for PS5. Candidate: LG OLED C4. "
                "LG OLED C4 price: 4599 PLN. Rejected Samsung CU8000: 60Hz and no HDMI 2.1. "
                "Todo: ask room brightness."
            ),
            compact_answer=(
                "Pick LG OLED C4: it is within the 5000 PLN budget at 4599 PLN and has HDMI 2.1 for PS5. "
                "Do not recommend Samsung CU8000 because it was rejected for 60Hz and no HDMI 2.1. "
                "Next: ask room brightness before final purchase."
            ),
            full_context_answer=(
                "LG OLED C4 is the best choice under 5000 PLN because it costs 4599 PLN and supports HDMI 2.1 for PS5. "
                "Samsung CU8000 should stay rejected due to 60Hz and no HDMI 2.1. Ask about room brightness next."
            ),
            required_facts=[
                "LG OLED C4",
                "5000 PLN",
                "4599 PLN",
                "HDMI 2.1 for PS5",
                "Rejected Samsung CU8000: 60Hz and no HDMI 2.1",
                "ask room brightness",
            ],
            forbidden_facts=["Samsung CU8000 is recommended", "budget = 8000 PLN"],
            expected_actions=["ask room brightness"],
        ),
        AnswerQualityCase(
            case_id="incident_handoff_good",
            question="Give the incident handoff.",
            source_context=(
                "Incident checkout latency spike severity = P1. Root cause = Redis connection pool exhaustion. "
                "Mitigation = increase pool to 200 and restart checkout workers. "
                "Rejected database outage: Postgres metrics normal. Todo: add Redis pool saturation alert."
            ),
            compact_answer=(
                "P1 checkout latency spike: root cause is Redis connection pool exhaustion. "
                "Mitigation is to increase the pool to 200 and restart checkout workers. "
                "Database outage is rejected because Postgres metrics were normal. Next: add a Redis pool saturation alert."
            ),
            full_context_answer=(
                "The incident is P1. Redis connection pool exhaustion caused checkout latency. "
                "Increase pool to 200, restart checkout workers, and add Redis pool saturation alert. "
                "Database outage was rejected because Postgres metrics were normal."
            ),
            required_facts=[
                "P1",
                "Redis connection pool exhaustion",
                "increase pool to 200",
                "restart checkout workers",
                "Rejected database outage",
                "add Redis pool saturation alert",
            ],
            forbidden_facts=["database outage is root cause", "severity = P3"],
            expected_actions=["add Redis pool saturation alert"],
        ),
        AnswerQualityCase(
            case_id="ppmlx_bench_handoff_good",
            question="Give the current ppmlx memory benchmark handoff and next action.",
            source_context=(
                "Goal: improve ppmlx synthetic memory evals so they reflect real-session quality. "
                "Need exact benchmark identifiers in reports. "
                "Decision: real-session quality-bench failures should drive synthetic benchmark design. "
                "Rejected synthetic-only PASS as sufficient because real sessions showed low recall. "
                "Todo: rerun answerable real-session batch with include-content. "
                "Todo: add context coverage metrics to compact-eval."
            ),
            compact_answer=(
                "Current ppmlx goal: improve synthetic memory evals so they reflect real-session quality. "
                "Reports must keep exact benchmark identifiers. The key decision is that real-session quality-bench "
                "failures should drive synthetic benchmark design; synthetic-only PASS was rejected because real sessions "
                "showed low recall. Next: rerun the answerable real-session batch with include-content and add context "
                "coverage metrics to compact-eval."
            ),
            full_context_answer=(
                "Improve ppmlx synthetic memory evals to mirror real-session quality. Keep exact benchmark IDs in reports. "
                "Use real-session quality-bench failures to design synthetic cases, because synthetic-only PASS was rejected "
                "after low real-session recall. Next rerun answerable real-session batch with include-content and add compact-eval "
                "context coverage metrics."
            ),
            required_facts=[
                "improve ppmlx synthetic memory evals",
                "real-session quality",
                "exact benchmark identifiers",
                "real-session quality-bench failures should drive synthetic benchmark design",
                "synthetic-only PASS was rejected",
                "real sessions showed low recall",
                "rerun answerable real-session batch with include-content",
                "add context coverage metrics to compact-eval",
            ],
            forbidden_facts=["synthetic-only PASS is sufficient", "real-session failures can be ignored"],
            expected_actions=[
                "rerun answerable real-session batch with include-content",
                "add context coverage metrics to compact-eval",
            ],
        ),
    ]


def select_required_facts(
    *,
    source_context: str,
    question: str,
    reference_answer: str | None = None,
    max_facts: int = 8,
) -> list[str]:
    """Select relevant required facts for real-session quality evals.

    Real agent traces often contain embedded examples, synthetic fixtures, and
    unrelated tool-test facts.  This selector ranks source-context bullets by
    relevance to the user's question and optional reference answer instead of
    treating every graph fact as mandatory.
    """
    candidates = _context_fact_candidates(source_context)
    if not candidates:
        candidates = _answer_clauses(reference_answer or source_context)
    topic_terms = set(_topic_terms(question))
    reference_terms = set(_topic_terms(reference_answer or ""))
    ranked: list[tuple[float, int, str]] = []
    for index, fact in enumerate(candidates):
        fact_terms = set(_fact_tokens(fact))
        if not fact_terms:
            continue
        topic_overlap = len(fact_terms & topic_terms)
        reference_overlap = len(fact_terms & reference_terms)
        if topic_terms and topic_overlap == 0 and reference_overlap == 0:
            continue
        score = topic_overlap * 3.0 + reference_overlap * 1.5
        score += _fact_signal_score(fact)
        if _looks_like_embedded_fixture(fact) and topic_overlap == 0:
            continue
        if score <= 0 and topic_terms:
            continue
        ranked.append((score, -index, fact))
    ranked.sort(reverse=True)
    selected: list[str] = []
    for _, __, fact in ranked:
        if fact not in selected:
            selected.append(fact)
        if len(selected) >= max_facts:
            break
    return selected


def build_reference_prompt(source_context: str, question: str) -> list[dict[str, str]]:
    """Build a stricter local-model reference prompt for real replay quality evals."""
    return [
        {
            "role": "system",
            "content": (
                "You are a concise evaluator assistant. Answer only from the provided source context. "
                "Focus on facts relevant to the user's question. Ignore embedded examples, synthetic fixtures, "
                "or unrelated test scenarios unless the question explicitly asks about them. "
                "Do not describe your analysis process. Do not invent details."
            ),
        },
        {
            "role": "user",
            "content": f"Source context:\n\n{source_context}\n\nQuestion: {question}",
        },
    ]


def load_cases(path: Path | str) -> list[AnswerQualityCase]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        raw_cases = data["cases"]
    elif isinstance(data, list):
        raw_cases = data
    else:
        raise ValueError("Answer quality dataset must be a list or {cases:[...]}")
    return [AnswerQualityCase.from_dict(item) for item in raw_cases]


def save_case_template(path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {"cases": [case.to_dict() for case in builtin_cases()]}
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return out


def match_facts(text: str, facts: list[str]) -> tuple[list[str], list[str]]:
    """Return facts found/missed in text using strict identifier-aware matching."""
    found = [fact for fact in facts if _contains_fact(text, fact)]
    missed = [fact for fact in facts if fact not in found]
    return found, missed


def _contains_wrong_fact(text: str, fact: str) -> bool:
    """Wrong-fact matching is intentionally stricter than recall matching."""
    return _normalize(fact) in _normalize(text)


def _contains_fact(text: str, fact: str) -> bool:
    text_norm = _normalize(text)
    fact_norm = _normalize(fact)
    if fact_norm in text_norm:
        return True
    if _missing_identifier_tokens(text_norm, fact_norm):
        return False
    fact_tokens = _fact_tokens(fact)
    if not fact_tokens:
        return True
    text_tokens = set(_fact_tokens(text))
    overlap = sum(1 for token in fact_tokens if token in text_tokens)
    threshold = 0.72 if len(fact_tokens) >= 5 else 0.9
    return (overlap / len(fact_tokens)) >= threshold


def _score_actionability(answer: str, expected_actions: list[str]) -> tuple[int, list[str]]:
    normalized = _normalize(answer)
    reasons: list[str] = []
    score = 1
    if len(_tokens(answer)) >= 8:
        score += 1
        reasons.append("substantive answer")
    if any(marker in normalized for marker in ("next", "todo", "ask", "do ", "increase", "restart", "send", "schedule", "add ")):
        score += 1
        reasons.append("contains next-step language")
    if expected_actions and any(_normalize(action) in normalized for action in expected_actions):
        score += 1
        reasons.append("mentions expected action")
    if re.search(r"\b\d+\b|[A-Z][A-Za-z0-9_.-]{2,}", answer):
        score += 1
        reasons.append("contains concrete identifiers or values")
    if any(refusal in normalized for refusal in ("i don't have", "cannot access", "not enough context")):
        score -= 2
        reasons.append("contains avoidable context refusal")
    return max(1, min(5, score)), reasons


def _score_grounding(
    *,
    answer: str,
    source_context: str,
    required_facts: list[str],
    forbidden_facts: list[str],
) -> tuple[float, list[str]]:
    unsupported_clauses: list[str] = []
    clauses = [clause for clause in _answer_clauses(answer) if len(_fact_tokens(clause)) >= 3]
    source_terms = set(_fact_tokens(source_context))
    required_supported = [fact for fact in required_facts if _contains_fact(answer, fact) and _contains_fact(source_context, fact)]
    grounded_clauses = 0
    for clause in clauses:
        if any(_contains_fact(clause, fact) and _contains_fact(source_context, fact) for fact in required_facts):
            grounded_clauses += 1
            continue
        clause_terms = set(_fact_tokens(clause))
        if not clause_terms:
            grounded_clauses += 1
            continue
        overlap = len(clause_terms & source_terms) / len(clause_terms)
        if overlap >= 0.55 or _is_low_risk_handoff_clause(clause):
            grounded_clauses += 1
        else:
            unsupported_clauses.append(clause)
    clause_score = _safe_ratio(grounded_clauses, len(clauses))
    required_score = _safe_ratio(len(required_supported), len([fact for fact in required_facts if _contains_fact(answer, fact)]) or len(required_facts))
    forbidden_hits = [fact for fact in forbidden_facts if _contains_wrong_fact(answer, fact)]
    forbidden_penalty = min(0.5, len(forbidden_hits) * 0.25)
    return max(0.0, 0.65 * clause_score + 0.35 * required_score - forbidden_penalty), unsupported_clauses[:8]


def _score_equivalence(*, compact_answer: str, full_context_answer: str, required_facts: list[str]) -> float:
    compact_required = set(match_facts(compact_answer, required_facts)[0])
    full_required = set(match_facts(full_context_answer, required_facts)[0])
    if full_required:
        fact_score = len(compact_required & full_required) / len(full_required)
    else:
        fact_score = _safe_ratio(len(compact_required), len(required_facts))
    compact_terms = set(_tokens(compact_answer))
    full_terms = set(_tokens(full_context_answer))
    lexical = _safe_ratio(len(compact_terms & full_terms), len(compact_terms | full_terms))
    return max(0.0, min(1.0, 0.75 * fact_score + 0.25 * lexical))


def _context_fact_candidates(source_context: str) -> list[str]:
    facts: list[str] = []
    for line in source_context.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            core = stripped[2:].split(" [source:", 1)[0].split(" (scope=", 1)[0].strip().rstrip(".")
            core = _strip_context_namespace(core)
            if 8 <= len(core) <= 220:
                facts.append(core)
    return facts


def _strip_context_namespace(fact: str) -> str:
    match = re.match(r"^[A-Za-z0-9_.:-]+\s+(constraint|decision|todo):\s*(.+)$", fact, flags=re.IGNORECASE)
    if match:
        label = match.group(1).lower()
        rest = match.group(2).strip()
        if label == "constraint":
            return rest
        if label == "decision":
            return rest if rest.lower().startswith("decision") else rest
        if label == "todo":
            return rest if rest.lower().startswith("todo") else f"todo: {rest}"
    return fact


def _answer_clauses(text: str) -> list[str]:
    clauses: list[str] = []
    for raw in re.split(r"\n+|(?<=[.!?])\s+", text or ""):
        cleaned = raw.strip(" -•\t")
        if 8 <= len(cleaned) <= 260:
            clauses.append(cleaned.rstrip("."))
    return clauses


def _missing_identifier_tokens(text_norm: str, fact_norm: str) -> list[str]:
    """Identifiers/numbers must match exactly; fuzzy overlap is too lenient for them."""
    important: list[str] = []
    for token in re.findall(r"[a-z0-9_./:-]+", fact_norm):
        cleaned = token.strip(".,;:()[]{}")
        if not cleaned or cleaned in {"http", "https"}:
            continue
        if any(char.isdigit() for char in cleaned) or any(char in cleaned for char in ("_", "/", ".")):
            important.append(cleaned)
    return [token for token in important if token not in text_norm]


def _topic_terms(text: str) -> list[str]:
    stop = {
        "give", "concise", "factual", "handoff", "current", "important", "status",
        "action", "actions", "answer", "from", "context", "session", "state", "what",
        "which", "this", "that", "with", "next", "goal", "goals", "decision",
        "decisions", "validation", "user", "task",
    }
    return [token for token in _fact_tokens(text) if token not in stop]


def _fact_signal_score(fact: str) -> float:
    lowered = _normalize(fact)
    score = 0.0
    for marker in ("goal", "decision", "todo", "requires", "constraint", "validation", "passed", "failed", "next"):
        if marker in lowered:
            score += 1.0
    if re.search(r"\bppmlx\b|memory|compact|replay|eval|trace|server|model", lowered):
        score += 2.0
    return score


def _looks_like_embedded_fixture(fact: str) -> bool:
    lowered = _normalize(fact)
    fixture_markers = (
        "dogfood", "synthetic", "fixture", "test scenario", "tv-shopping",
        "travel-dogfood", "incident-dogfood", "screen_size", "budget =", "hotel", "oled",
    )
    return any(marker in lowered for marker in fixture_markers)


def _is_low_risk_handoff_clause(clause: str) -> bool:
    lowered = _normalize(clause)
    return any(marker in lowered for marker in ("next", "todo", "recommend", "continue", "verify", "check"))


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9_]+", _normalize(text)) if len(token) >= 3]


def _fact_tokens(text: str) -> list[str]:
    stop = {
        "the", "and", "for", "with", "was", "were", "because", "that", "this",
        "from", "into", "under", "over", "next", "todo", "should", "stay", "has",
        "have", "had", "is", "are", "not", "due", "its", "it", "to", "a", "an",
    }
    return [token for token in _tokens(text) if token not in stop]


def _normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


def _safe_ratio(num: int, den: int) -> float:
    if den <= 0:
        return 1.0
    return num / den


def _round_avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)
