from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_CASES_PATH = REPO_ROOT / "eval" / "categorization_test_cases.json"
ADDITIONAL_TEST_CASES_PATH = REPO_ROOT / "eval" / "categorization_test_cases_additional.json"
REPORTS_DIR = REPO_ROOT / "eval" / "reports"
REPORT_JSON_PATH = REPORTS_DIR / "categorization_comparison_report.json"
REPORT_MD_PATH = REPORTS_DIR / "categorization_comparison_report.md"


def load_cases_from_path(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    for category_group in payload["categories"]:
        expected_category = category_group["category"]
        for question in category_group["questions"]:
            cases.append(
                {
                    "question": question,
                    "expected_category": expected_category,
                }
            )
    return cases


def load_cases() -> list[dict[str, Any]]:
    raw_cases = load_cases_from_path(TEST_CASES_PATH) + load_cases_from_path(
        ADDITIONAL_TEST_CASES_PATH
    )
    cases: list[dict[str, Any]] = []
    for case_id, raw_case in enumerate(raw_cases, start=1):
        cases.append(
            {
                "id": case_id,
                "question": raw_case["question"],
                "expected_category": raw_case["expected_category"],
            }
        )
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
        "--finetuned-code-model",
        action="store",
        default="our-house-qwen3-0.6b",
        help="Fine-tuned model that emits opaque short codes.",
    )
    group.addoption(
        "--finetuned-category-model",
        action="store",
        default="our-house-qwen3-0.6b-category-names",
        help="Fine-tuned model that emits full category names.",
    )
    group.addoption(
        "--baseline-model",
        action="store",
        default="qwen3:0.6b",
        help="Base model name used for both opaque-code and full-category-name scenarios.",
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
    if "model_target" in metafunc.fixturenames:
        baseline_model = str(metafunc.config.getoption("--baseline-model"))
        finetuned_code_model = str(metafunc.config.getoption("--finetuned-code-model"))
        finetuned_category_model = str(
            metafunc.config.getoption("--finetuned-category-model")
        )

        model_targets: list[dict[str, str]] = [
            {
                "scenario": "finetuned-code",
                "kind": "finetuned",
                "name": finetuned_code_model,
                "label_mode": "code",
            },
            {
                "scenario": "finetuned-category",
                "kind": "finetuned",
                "name": finetuned_category_model,
                "label_mode": "category",
            },
            {
                "scenario": "baseline-category",
                "kind": "baseline",
                "name": baseline_model,
                "label_mode": "category",
            },
            {
                "scenario": "baseline-code",
                "kind": "baseline",
                "name": baseline_model,
                "label_mode": "code",
            },
        ]

        metafunc.parametrize(
            "model_target",
            model_targets,
            ids=[
                f"{target['scenario']}-{target['name']}"
                for target in model_targets
            ],
        )


@pytest.fixture(scope="session")
def api_base_url(pytestconfig: pytest.Config) -> str:
    return str(pytestconfig.getoption("--api-base-url")).rstrip("/")


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

    summaries = build_model_summaries(results)
    summary = summaries[0] if len(summaries) == 1 else {}

    report = {
        "metadata": {
            "api_base_url": str(config.getoption("--api-base-url")),
            "baseline_model": str(config.getoption("--baseline-model")),
            "finetuned_code_model": str(config.getoption("--finetuned-code-model")),
            "finetuned_category_model": str(
                config.getoption("--finetuned-category-model")
            ),
            "scenarios_evaluated": [
                "finetuned-code",
                "finetuned-category",
                "baseline-category",
                "baseline-code",
            ],
            "num_predict": int(config.getoption("--num-predict")),
            "think": bool(config.getoption("--think")),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": round((finished_at - started_at).total_seconds(), 2),
        },
        "summary": summary,
        "summaries": summaries,
        "results": results,
    }

    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_MD_PATH.write_text(render_markdown_report(report), encoding="utf-8")


def render_markdown_report(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    summaries = report.get("summaries", [])
    results = report["results"]

    lines = [
        "# Categorization Evaluation Report",
        "",
        f"- API base URL: `{metadata['api_base_url']}`",
        f"- Fine-tuned code model: `{metadata['finetuned_code_model']}`",
        f"- Fine-tuned category model: `{metadata['finetuned_category_model']}`",
        f"- Baseline model: `{metadata['baseline_model']}`",
        f"- Scenarios evaluated: `{', '.join(metadata['scenarios_evaluated'])}`",
        f"- num_predict: `{metadata['num_predict']}`",
        f"- think: `{metadata['think']}`",
        f"- Started: `{metadata['started_at']}`",
        f"- Finished: `{metadata['finished_at']}`",
        f"- Duration seconds: `{metadata['duration_seconds']}`",
        "",
        "## Summary",
        "",
    ]

    lines.extend(
        [
            "| Scenario | Model | Kind | Label Mode | Correct | Incorrect | Accuracy | Avg ms | Max ms |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for item in summaries:
        lines.append(
            "| {scenario} | {model_name} | {model_kind} | {label_mode} | {correct} | {incorrect} | {accuracy:.2%} | {avg_ms} | {max_ms} |".format(
                scenario=escape_markdown_cell(item.get("scenario", "")),
                model_name=escape_markdown_cell(item.get("model_name", "")),
                model_kind=escape_markdown_cell(item.get("model_kind", "")),
                label_mode=escape_markdown_cell(item.get("label_mode", "")),
                correct=item.get("correct", 0),
                incorrect=item.get("incorrect", 0),
                accuracy=item.get("accuracy", 0.0),
                avg_ms=item.get("average_duration_ms", 0.0),
                max_ms=item.get("max_duration_ms", 0.0),
            )
        )

    for item in summaries:
        scenario = item.get("scenario", "")
        model_name = item.get("model_name", "")
        model_kind = item.get("model_kind", "")
        label_mode = item.get("label_mode", "")
        model_results = [
            result
            for result in results
            if result["scenario"] == scenario
            and result["model_name"] == model_name
            and result["model_kind"] == model_kind
            and result["label_mode"] == label_mode
        ]

        lines.extend(
            [
                "",
                f"## {scenario}: {model_name} ({model_kind}, {label_mode})",
                "",
                "### Correct Responses",
                "",
                "| Case | Question | Expected | Predicted | Code | Time ms |",
                "| --- | --- | --- | --- | --- | ---: |",
            ]
        )

        correct_results = [result for result in model_results if result["correct"]]
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

        incorrect_results = [result for result in model_results if not result["correct"]]
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


def build_model_summaries(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped_results: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for result in results:
        key = (
            str(result["scenario"]),
            str(result["model_kind"]),
            str(result["model_name"]),
            str(result["label_mode"]),
        )
        grouped_results.setdefault(key, []).append(result)

    summaries: list[dict[str, Any]] = []
    for (scenario, model_kind, model_name, label_mode), model_results in grouped_results.items():
        correct_count = sum(1 for result in model_results if result["correct"])
        total_count = len(model_results)
        durations = [result["duration_ms"] for result in model_results]
        summaries.append(
            {
                "scenario": scenario,
                "model_kind": model_kind,
                "model_name": model_name,
                "label_mode": label_mode,
                "total": total_count,
                "correct": correct_count,
                "incorrect": total_count - correct_count,
                "accuracy": round(correct_count / total_count, 4) if total_count else 0.0,
                "average_duration_ms": round(mean(durations), 2) if durations else 0.0,
                "max_duration_ms": round(max(durations), 2) if durations else 0.0,
            }
        )

    return sorted(
        summaries,
        key=lambda item: (
            item["scenario"],
            item["model_kind"],
            item["model_name"],
            item["label_mode"],
        ),
    )


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
