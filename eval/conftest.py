from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_CASES_PATH = REPO_ROOT / "eval" / "categorization_test_cases.json"
REPORTS_DIR = REPO_ROOT / "eval" / "reports"
REPORT_JSON_PATH = REPORTS_DIR / "categorization_comparison_report.json"
REPORT_MD_PATH = REPORTS_DIR / "categorization_comparison_report.md"


def load_cases() -> list[dict[str, Any]]:
    payload = json.loads(TEST_CASES_PATH.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    case_id = 1
    for category_group in payload["categories"]:
        expected_category = category_group["category"]
        for question in category_group["questions"]:
            cases.append(
                {
                    "id": case_id,
                    "question": question,
                    "expected_category": expected_category,
                }
            )
            case_id += 1
    return cases


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("categorization-eval")
    group.addoption(
        "--api-base-url",
        action="store",
        default="http://localhost:8090",
        help="Base URL for the categorization API.",
    )
    group.addoption(
        "--finetuned-model",
        action="store",
        default="our-house-qwen3-0.6b",
        help="Fine-tuned model name to evaluate.",
    )
    group.addoption(
        "--num-predict",
        action="store",
        type=int,
        default=50,
        help="num_predict value sent to the API for each evaluation request.",
    )
    group.addoption(
        "--think",
        action="store_true",
        default=False,
        help="Enable model thinking mode during evaluation requests.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config._categorization_eval_results = []
    config._categorization_eval_started_at = datetime.now(timezone.utc)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "case" in metafunc.fixturenames:
        cases = load_cases()
        metafunc.parametrize(
            "case",
            cases,
            ids=[f"{case['id']:03d}-{case['expected_category']}" for case in cases],
        )


@pytest.fixture(scope="session")
def api_base_url(pytestconfig: pytest.Config) -> str:
    return str(pytestconfig.getoption("--api-base-url")).rstrip("/")


@pytest.fixture(scope="session")
def finetuned_model(pytestconfig: pytest.Config) -> str:
    return str(pytestconfig.getoption("--finetuned-model"))


@pytest.fixture(scope="session")
def num_predict(pytestconfig: pytest.Config) -> int:
    return int(pytestconfig.getoption("--num-predict"))


@pytest.fixture(scope="session")
def think(pytestconfig: pytest.Config) -> bool:
    return bool(pytestconfig.getoption("--think"))


def write_report(config: pytest.Config) -> None:
    results = list(config._categorization_eval_results)
    started_at = config._categorization_eval_started_at
    finished_at = datetime.now(timezone.utc)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    correct_count = sum(1 for result in results if result["correct"])
    total_count = len(results)
    durations = [result["duration_ms"] for result in results]
    summary = {
        "model_name": results[0]["model_name"] if results else "",
        "total": total_count,
        "correct": correct_count,
        "incorrect": total_count - correct_count,
        "accuracy": round(correct_count / total_count, 4) if total_count else 0.0,
        "average_duration_ms": round(mean(durations), 2) if durations else 0.0,
        "max_duration_ms": round(max(durations), 2) if durations else 0.0,
    }

    report = {
        "metadata": {
            "api_base_url": str(config.getoption("--api-base-url")),
            "finetuned_model": str(config.getoption("--finetuned-model")),
            "num_predict": int(config.getoption("--num-predict")),
            "think": bool(config.getoption("--think")),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": round((finished_at - started_at).total_seconds(), 2),
        },
        "summary": summary,
        "results": results,
    }

    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_MD_PATH.write_text(render_markdown_report(report), encoding="utf-8")


def render_markdown_report(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    summary = report["summary"]
    results = report["results"]

    lines = [
        "# Categorization Evaluation Report",
        "",
        f"- API base URL: `{metadata['api_base_url']}`",
        f"- Fine-tuned model: `{metadata['finetuned_model']}`",
        f"- num_predict: `{metadata['num_predict']}`",
        f"- think: `{metadata['think']}`",
        f"- Started: `{metadata['started_at']}`",
        f"- Finished: `{metadata['finished_at']}`",
        f"- Duration seconds: `{metadata['duration_seconds']}`",
        "",
        "## Summary",
        "",
        "| Model | Correct | Incorrect | Accuracy | Avg ms | Max ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        "| {model_name} | {correct} | {incorrect} | {accuracy:.2%} | {avg_ms} | {max_ms} |".format(
            model_name=summary.get("model_name", "finetuned"),
            correct=summary.get("correct", 0),
            incorrect=summary.get("incorrect", 0),
            accuracy=summary.get("accuracy", 0.0),
            avg_ms=summary.get("average_duration_ms", 0.0),
            max_ms=summary.get("max_duration_ms", 0.0),
        ),
        "",
        f"## {summary.get('model_name', 'finetuned')}",
        "",
        "### Correct Responses",
        "",
        "| Case | Question | Expected | Predicted | Code | Time ms |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]

    correct_results = [result for result in results if result["correct"]]
    for result in correct_results:
        lines.append(
            "| {case_id} | {question} | {expected} | {predicted} | {code} | {duration_ms} |".format(
                case_id=result["case_id"],
                question=escape_markdown_cell(result["question"]),
                expected=escape_markdown_cell(result["expected_category"]),
                predicted=escape_markdown_cell(result["predicted_category"] or ""),
                code=escape_markdown_cell(result["predicted_code"] or ""),
                duration_ms=result["duration_ms"],
            )
        )
    if not correct_results:
        lines.append("| - | No correct responses | - | - | - | - |")

    lines.extend(
        [
            "",
            "### Incorrect Responses",
            "",
            "| Case | Question | Expected | Predicted | Code | Status | Time ms | Error |",
            "| --- | --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )

    incorrect_results = [result for result in results if not result["correct"]]
    for result in incorrect_results:
        lines.append(
            "| {case_id} | {question} | {expected} | {predicted} | {code} | {status} | {duration_ms} | {error} |".format(
                case_id=result["case_id"],
                question=escape_markdown_cell(result["question"]),
                expected=escape_markdown_cell(result["expected_category"]),
                predicted=escape_markdown_cell(result["predicted_category"] or ""),
                code=escape_markdown_cell(result["predicted_code"] or ""),
                status=result["status_code"] if result["status_code"] is not None else "",
                duration_ms=result["duration_ms"],
                error=escape_markdown_cell(result["error"] or ""),
            )
        )
    if not incorrect_results:
        lines.append("| - | No incorrect responses | - | - | - | - | - | - |")

    return "\n".join(lines) + "\n"


def escape_markdown_cell(value: Any) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False)
    return text.replace("|", "\\|").replace("\n", "<br>")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    write_report(session.config)
