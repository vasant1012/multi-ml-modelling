"""
Hotel Review Intelligence
Central Test Runner & Dashboard
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.append(os.getcwd())
from backend.config import TESTS_DIR
from backend.logger import logger

# ==========================================================
# Configuration
# ==========================================================

JSON_REPORT = Path(f"{TESTS_DIR}/pytest_report.json")
FAILED_TESTS_FILE = Path(f"{TESTS_DIR}/last_failed_tests.json")
FAILED_TESTS_FILE.parent.mkdir(parents=True, exist_ok=True)


def _suite_files():
    test_dir = Path(TESTS_DIR)
    return sorted(
        str(path)
        for path in test_dir.glob("test_*.py")
        if path.name != "run_tests.py"
    )


SUITES = {
    "api": [
        str(Path(TESTS_DIR) / "test_api.py"),
        str(Path(TESTS_DIR) / "test_batch_api.py"),
    ],
    "graph": [
        str(Path(TESTS_DIR) / "test_full_graph.py"),
        str(Path(TESTS_DIR) / "test_reconciliation.py"),
        str(Path(TESTS_DIR) / "test_workflow.py"),
        str(Path(TESTS_DIR) / "test_model_registry.py"),
    ],
    "absa": [
        str(Path(TESTS_DIR) / "test_aspects.py"),
    ],
    "emotion": [
        str(Path(TESTS_DIR) / "test_emotion_graph.py"),
    ],
    "summary": [
        str(Path(TESTS_DIR) / "test_summarization.py"),
    ],
    "integration": [
        str(Path(TESTS_DIR) / "test_api.py"),
        str(Path(TESTS_DIR) / "test_full_graph.py"),
    ],
    "all": _suite_files(),
}


# ==========================================================
# Failed Test Storage
# ==========================================================

def save_failed_tests(failed_tests):
    FAILED_TESTS_FILE.write_text(json.dumps(failed_tests, indent=4), encoding="utf-8")


def load_failed_tests():
    if not FAILED_TESTS_FILE.exists():
        return []
    return json.loads(FAILED_TESTS_FILE.read_text(encoding="utf-8"))


# ==========================================================
# Execute Pytest
# ==========================================================

def _suite_label_for(nodeid: str) -> str:
    path = Path(nodeid.split("::")[0])
    file_name = path.stem.upper()
    mapping = {
        "TEST_API": "API",
        "TEST_BATCH_API": "API",
        "TEST_FULL_GRAPH": "GRAPH",
        "TEST_RECONCILIATION": "GRAPH",
        "TEST_WORKFLOW": "GRAPH",
        "TEST_MODEL_REGISTRY": "GRAPH",
        "TEST_ASPECTS": "ABSA",
        "TEST_EMOTION_GRAPH": "EMOTION",
        "TEST_SUMMARIZATION": "SUMMARY",
    }
    return mapping.get(
        file_name,
        file_name.replace("TEST_", "").replace("_", " ").title().upper(),
    )


def run_pytest(targets):
    start = time.time()
    cmd = [
        "pytest",
        "-q",
        "--json-report",
        f"--json-report-file={JSON_REPORT}",
        *targets,
    ]
    process = subprocess.run(cmd, capture_output=True, text=True, check=False)
    duration = time.time() - start
    return process, duration


# ==========================================================
# Parse JSON Report
# ==========================================================

def parse_report():
    if not JSON_REPORT.exists():
        return {}
    try:
        return json.loads(JSON_REPORT.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


# ==========================================================
# Process Results
# ==========================================================

def summarize_results(report):
    suite_results = {}
    failed_tests = []

    for test in report.get("tests", []):
        nodeid = test.get("nodeid", "")
        outcome = test.get("outcome", "")
        if not nodeid:
            continue

        suite = _suite_label_for(nodeid)
        suite_results.setdefault(suite, {"passed": 0, "failed": 0})

        if outcome == "passed":
            suite_results[suite]["passed"] += 1
            # logger.info("PASS %s", nodeid)
        elif outcome in {"failed", "error"}:
            suite_results[suite]["failed"] += 1
            reason = (
                test.get("longrepr")
                or test.get("call", {}).get("crash", {}).get("message")
                or test.get("message")
                or "Unknown error"
            )
            failed_tests.append({"test": nodeid, "reason": str(reason).strip()})
            # logger.error("FAIL %s", nodeid)

    save_failed_tests([item["test"] for item in failed_tests])
    return suite_results, failed_tests


# ==========================================================
# Dashboard
# ==========================================================

def print_dashboard(suite_results, failed_tests, duration, total_passed=None, total_failed=None):
    if total_passed is None:
        total_passed = sum(stats["passed"] for stats in suite_results.values())
    if total_failed is None:
        total_failed = sum(stats["failed"] for stats in suite_results.values())

    print()
    print("=" * 70)
    print("HOTEL REVIEW INTELLIGENCE TEST DASHBOARD")
    print("=" * 70)
    print()
    print("EXECUTION SUMMARY")
    print()
    print(f"{'Suite':<18}{'Status':<12}{'Passed':>8}{'Failed':>8}")
    print()

    ordered = ["API", "GRAPH", "ABSA", "EMOTION", "SUMMARY"]
    suites = [suite for suite in ordered if suite in suite_results]
    suites.extend(suite for suite in suite_results if suite not in ordered)

    for suite in suites:
        stats = suite_results[suite]
        passed = stats["passed"]
        failed = stats["failed"]
        status = "✅ PASS" if failed == 0 else "❌ FAIL"
        print(f"{suite:<18}{status:<12}{passed:>8}{failed:>8}")

    print()
    print("-" * 70)
    print()
    print("FAILED TESTS")
    print()

    if not failed_tests:
        print("No failures detected.")
    else:
        for item in failed_tests:
            print(f"❌ {item['test']}")
            print()
            print("Reason:")
            print(item["reason"])
            print()

    print("-" * 70)
    print()
    print(f"Total Tests : {total_passed + total_failed}")
    print(f"Passed      : {total_passed}")
    print(f"Failed      : {total_failed}")
    print(f"Duration    : {duration:.2f} sec")
    print()
    print("=" * 70)
    print()


# ==========================================================
# Main
# ==========================================================

def main():
    selected = sys.argv[1:]
    if not selected:
        selected = ["all"]

    if selected == ["failed"]:
        targets = load_failed_tests()
        if not targets:
            print("\nNo failed tests found.\n")
            return
    else:
        targets = []
        for suite in selected:
            if suite not in SUITES:
                print(f"\nUnknown suite: {suite}\n")
                return
            targets.extend(SUITES[suite])

    logger.info("Executing suites: %s", ", ".join(selected))
    process, duration = run_pytest(targets)
    report = parse_report()
    suite_results, failed_tests = summarize_results(report)
    total_passed = sum(stats["passed"] for stats in suite_results.values())
    total_failed = sum(stats["failed"] for stats in suite_results.values())
    print_dashboard(suite_results, failed_tests, duration, total_passed, total_failed)
    sys.exit(process.returncode)


if __name__ == "__main__":
    main()