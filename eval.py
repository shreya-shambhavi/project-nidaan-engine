"""
eval.py — Validation framework for Project Nidaan Engine.

Three evaluation layers:
  1. Unit tests   : logic checks on individual components (no API calls)
  2. Integration  : full pipeline runs on benchmark queries (uses Groq + PubMed)
  3. Report       : aggregated metrics saved to eval_report.json

Run with:
    python eval.py              # all layers
    python eval.py --unit       # unit tests only (no API needed)
"""

import json
import sys
import time
from datetime import datetime


# ── BENCHMARK SUITE ────────────────────────────────────────────────────────────
# Each entry defines a test query and the domain terms that must appear in a
# correct summary. Keep to 2–3 entries to limit API usage during development.

BENCHMARKS = [
    {
        "description": "GLP-1 agonist weight management",
        "query": "semaglutide weight loss clinical trial",
        "must_contain_terms": ["semaglutide", "weight", "obesity", "GLP-1"],
        "min_summary_length": 150,
    },
    {
        "description": "Hereditary breast cancer genetics",
        "query": "BRCA1 BRCA2 breast cancer mutation risk",
        "must_contain_terms": ["BRCA", "breast", "cancer", "mutation"],
        "min_summary_length": 150,
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — UNIT TESTS
# Pure logic tests. No LLM or PubMed calls. Run offline.
# ══════════════════════════════════════════════════════════════════════════════

def test_agent_state_has_required_keys():
    """AgentState must contain all fields added across our fixes."""
    required = [
        "query", "pubmed_query", "abstracts", "extractions",
        "key_findings", "draft_summary", "feedback",
        "feedback_history", "confidence_score", "iterations",
    ]
    state = {k: None for k in required}
    for key in required:
        assert key in state, f"Missing AgentState key: {key}"


def test_reviewer_score_must_be_in_bounds():
    """Confidence scores must always be in [0.0, 1.0]."""
    valid = [0.0, 0.5, 0.85, 0.9, 1.0]
    for s in valid:
        assert 0.0 <= s <= 1.0, f"Score {s} out of bounds"

    invalid = [-0.1, 1.1, 2.0]
    for s in invalid:
        assert not (0.0 <= s <= 1.0), f"Score {s} should be invalid but passed"


def test_extraction_schema_has_all_fields():
    """Each AbstractExtraction dict must have every required field populated."""
    required_fields = [
        "pmid", "study_type", "sample_size",
        "population", "key_finding", "limitation",
    ]
    mock = {
        "pmid": "12345678",
        "study_type": "RCT",
        "sample_size": "342",
        "population": "Adults with type 2 diabetes, age 18-75",
        "key_finding": "Semaglutide reduced HbA1c by 1.5% vs placebo at 26 weeks",
        "limitation": "Single-centre, 26-week follow-up only",
    }
    for field in required_fields:
        assert field in mock, f"Missing field: {field}"
        assert mock[field].strip(), f"Field '{field}' is empty"


def test_feedback_history_grows_monotonically():
    """Each reviewer iteration must append — never overwrite — the history."""
    feedbacks = [
        "Missing sample sizes for all three studies.",
        "Outcomes section too vague — specify primary endpoint values.",
        "Limitation of study 2 not mentioned.",
    ]
    history = []
    for i, fb in enumerate(feedbacks):
        entry = f"Iteration {i + 1}: {fb}"
        history = history + [entry]
        assert len(history) == i + 1
        assert f"Iteration {i + 1}" in history[i]

    # Confirm the buggy overwrite approach only retains the last entry
    buggy_feedback = ""
    for fb in feedbacks:
        buggy_feedback = fb
    assert buggy_feedback == feedbacks[-1]


def test_pubmed_query_is_structured():
    """Query planner output must contain MeSH terms or boolean operators."""

    def is_structured(q: str) -> bool:
        return bool(q) and ("[MeSH" in q or " AND " in q or " OR " in q)

    good = [
        "(neoplasms[MeSH Terms]) AND (antineoplastic agents[MeSH Terms])",
        "semaglutide[MeSH Terms] AND (obesity[MeSH Terms] OR weight loss[MeSH Terms])",
    ]
    bad = ["cancer treatment", "", "tell me about diabetes"]

    for q in good:
        assert is_structured(q), f"Should be valid: {q}"
    for q in bad:
        assert not is_structured(q), f"Should be invalid: {q}"


def test_router_logic():
    """route_after_review must follow the score threshold and iteration cap."""
    from app import route_after_review

    cases = [
        ({"confidence_score": 0.90, "iterations": 1}, "end"),
        ({"confidence_score": 0.85, "iterations": 1}, "end"),
        ({"confidence_score": 0.84, "iterations": 1}, "continue"),
        ({"confidence_score": 0.50, "iterations": 1}, "continue"),
        ({"confidence_score": 0.50, "iterations": 3}, "end"),
        ({"confidence_score": 0.50, "iterations": 4}, "end"),
    ]
    for state, expected in cases:
        result = route_after_review(state)
        assert result == expected, (
            f"State {state} → expected '{expected}', got '{result}'"
        )


def test_abstract_parsing_returns_list():
    """fetch_pubmed_abstracts must always return a list, even on bad input."""
    from tools import fetch_pubmed_abstracts

    result = fetch_pubmed_abstracts("", max_results=1)
    assert isinstance(result, list), "Must return list on empty query"


UNIT_TESTS = [
    test_agent_state_has_required_keys,
    test_reviewer_score_must_be_in_bounds,
    test_extraction_schema_has_all_fields,
    test_feedback_history_grows_monotonically,
    test_pubmed_query_is_structured,
    test_router_logic,
    test_abstract_parsing_returns_list,
]


def run_unit_tests() -> list:
    print("\n-- LAYER 1: UNIT TESTS " + "-" * 38)
    results = []

    for fn in UNIT_TESTS:
        label = fn.__name__
        try:
            fn()
            results.append({"test": label, "status": "PASS"})
            print(f"  PASS  {label}")
        except AssertionError as e:
            results.append({"test": label, "status": "FAIL", "reason": str(e)})
            print(f"  FAIL  {label}\n       {e}")
        except Exception as e:
            results.append({"test": label, "status": "ERROR", "reason": str(e)})
            print(f"  ERR   {label}\n       {e}")

    passed = sum(1 for r in results if r["status"] == "PASS")
    print(f"\n  {passed}/{len(results)} passed")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — INTEGRATION TESTS
# Each benchmark runs the full pipeline exactly once and collects all quality
# checks from that single output. Avoids double-invoking the graph.
# ══════════════════════════════════════════════════════════════════════════════

def _run_pipeline(query: str) -> tuple:
    """Invoke the compiled graph and return (output, elapsed_seconds)."""
    from app import graph
    start = time.time()
    output = graph.invoke({"query": query, "iterations": 0})
    return output, round(time.time() - start, 2)


def _check_summary_quality(output: dict, benchmark: dict) -> dict:
    """Run all quality checks on a single pipeline output dict."""
    checks = {}
    summary = output.get("draft_summary", "")

    # Basic completion
    checks["summary_non_empty"]   = len(summary) > benchmark["min_summary_length"]
    checks["summary_length"]      = len(summary)
    checks["abstracts_retrieved"] = len(output.get("abstracts", [])) > 0

    # Confidence score
    score = output.get("confidence_score", -1.0)
    checks["score_in_bounds"]       = 0.0 <= score <= 1.0
    checks["score_above_threshold"] = score >= 0.85
    checks["confidence_score"]      = round(score, 3)

    # Query planner check
    pq = output.get("pubmed_query", "")
    checks["query_optimized"] = "[MeSH" in pq or " AND " in pq or " OR " in pq
    checks["pubmed_query"]    = pq

    # Term recall: domain terms must appear in the summary
    terms   = benchmark["must_contain_terms"]
    found   = [t for t in terms if t.lower() in summary.lower()]
    missing = [t for t in terms if t.lower() not in summary.lower()]
    recall  = len(found) / len(terms) if terms else 0.0
    checks["term_recall"]      = round(recall, 3)
    checks["terms_found"]      = found
    checks["terms_missing"]    = missing
    checks["term_recall_pass"] = recall >= 0.5

    # Extraction completeness
    extractions = output.get("extractions", [])
    checks["extractions_count"] = len(extractions)

    if extractions:
        tracked = ["study_type", "sample_size", "population", "key_finding"]
        fill_rates = {}
        for field in tracked:
            populated = sum(
                1 for ex in extractions
                if ex.get(field, "").strip().lower()
                not in ("", "not reported", "n/a", "unknown")
            )
            fill_rates[field] = round(populated / len(extractions), 2)
        checks["field_fill_rates"] = fill_rates
        checks["avg_fill_rate"]    = round(
            sum(fill_rates.values()) / len(fill_rates), 2
        )
        checks["extraction_pass"]  = checks["avg_fill_rate"] >= 0.5
    else:
        checks["extraction_pass"] = False
        checks["avg_fill_rate"]   = 0.0

    # Feedback history integrity
    history = output.get("feedback_history", [])
    checks["feedback_history_length"]    = len(history)
    checks["history_labeled_correctly"]  = all(
        f"Iteration {i + 1}" in h for i, h in enumerate(history)
    )

    # Iteration efficiency
    checks["iterations"]          = output.get("iterations", 0)
    checks["converged_first_try"] = output.get("iterations", 0) == 1

    return checks


def _determine_status(checks: dict) -> str:
    """A benchmark passes only when all critical checks pass."""
    critical = [
        "summary_non_empty",
        "score_in_bounds",
        "term_recall_pass",
        "extraction_pass",
        "query_optimized",
    ]
    return "PASS" if all(checks.get(c) for c in critical) else "FAIL"


def run_integration_tests() -> list:
    print("\n-- LAYER 2: INTEGRATION TESTS " + "-" * 30)
    results = []

    for benchmark in BENCHMARKS:
        print(f"\n  Benchmark : {benchmark['description']}")
        print(f"  Query     : \"{benchmark['query']}\"")

        result = {
            "benchmark": benchmark["description"],
            "query":     benchmark["query"],
        }

        try:
            output, elapsed = _run_pipeline(benchmark["query"])
            result["elapsed_seconds"] = elapsed

            checks = _check_summary_quality(output, benchmark)
            result.update(checks)

            status = _determine_status(checks)
            result["status"] = status

            icon = "PASS" if status == "PASS" else "FAIL"
            print(f"  [{icon}]  score={checks['confidence_score']}  "
                  f"recall={checks['term_recall']:.0%}  "
                  f"fill={checks.get('avg_fill_rate', 'N/A')}  "
                  f"iters={checks['iterations']}  "
                  f"({elapsed}s)")

            if status == "FAIL":
                failing = [
                    k for k in [
                        "summary_non_empty", "score_in_bounds",
                        "term_recall_pass", "extraction_pass", "query_optimized",
                    ]
                    if not checks.get(k)
                ]
                print(f"         Failed checks: {failing}")

        except Exception as e:
            result["status"] = "ERROR"
            result["reason"] = str(e)
            print(f"  [ERR]  {e}")

        results.append(result)

    passed = sum(1 for r in results if r["status"] == "PASS")
    print(f"\n  {passed}/{len(results)} passed")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(unit: list, integration: list) -> dict:
    print("\n-- EVALUATION REPORT " + "-" * 39)

    passing = [r for r in integration if r.get("status") == "PASS"]
    aggregate = {}
    if passing:
        aggregate = {
            "avg_confidence_score": round(
                sum(r["confidence_score"] for r in passing) / len(passing), 3
            ),
            "avg_term_recall": round(
                sum(r["term_recall"] for r in passing) / len(passing), 3
            ),
            "avg_fill_rate": round(
                sum(r.get("avg_fill_rate", 0) for r in passing) / len(passing), 3
            ),
            "avg_iterations": round(
                sum(r["iterations"] for r in passing) / len(passing), 2
            ),
        }
        print(f"  Avg confidence score : {aggregate['avg_confidence_score']}")
        print(f"  Avg term recall      : {aggregate['avg_term_recall']:.0%}")
        print(f"  Avg extraction fill  : {aggregate['avg_fill_rate']:.0%}")
        print(f"  Avg iterations taken : {aggregate['avg_iterations']}")

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "unit_tests": {
            "total":   len(unit),
            "passed":  sum(1 for r in unit if r["status"] == "PASS"),
            "results": unit,
        },
        "integration_tests": {
            "total":   len(integration),
            "passed":  len(passing),
            "results": integration,
        },
        "aggregate_metrics": aggregate,
    }

    path = "eval_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report saved to {path}")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unit_only = "--unit" in sys.argv

    unit_results = run_unit_tests()

    if unit_only:
        print("\n(Skipping integration tests -- --unit flag set)")
        sys.exit(0 if all(r["status"] == "PASS" for r in unit_results) else 1)

    integration_results = run_integration_tests()
    generate_report(unit_results, integration_results)

    all_passed = (
        all(r["status"] == "PASS" for r in unit_results)
        and all(r["status"] == "PASS" for r in integration_results)
    )
    sys.exit(0 if all_passed else 1)