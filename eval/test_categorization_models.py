from __future__ import annotations

import json
from time import perf_counter

import httpx


def test_categorize_question(
    case: dict[str, object],
    finetuned_model: str,
    api_base_url: str,
    num_predict: int,
    think: bool,
    request,
) -> None:
    payload = {
        "model_name": finetuned_model,
        "question": case["question"],
        "num_predict": num_predict,
        "think": think,
    }

    start = perf_counter()
    status_code = None
    predicted_category = None
    predicted_code = None
    error = None

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{api_base_url}/categorize", json=payload)
        status_code = response.status_code
        response_json = response.json()
        duration_ms = round((perf_counter() - start) * 1000, 2)

        if response.is_success:
            predicted_category = response_json.get("category")
            predicted_code = response_json.get("code")
        else:
            detail = response_json.get("detail", response.text)
            error = detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False)
    except httpx.HTTPError as exc:
        duration_ms = round((perf_counter() - start) * 1000, 2)
        error = str(exc)
    except ValueError as exc:
        duration_ms = round((perf_counter() - start) * 1000, 2)
        error = f"Unable to decode JSON response: {exc}"

    correct = predicted_category == case["expected_category"]

    request.config._categorization_eval_results.append(
        {
            "case_id": case["id"],
            "question": case["question"],
            "expected_category": case["expected_category"],
            "model_kind": "finetuned",
            "model_name": finetuned_model,
            "predicted_category": predicted_category,
            "predicted_code": predicted_code,
            "correct": correct,
            "duration_ms": duration_ms,
            "status_code": status_code,
            "error": error,
        }
    )

    assert error is None, error
    assert predicted_category == case["expected_category"], (
        f"Expected {case['expected_category']!r} but got {predicted_category!r} "
        f"for question {case['question']!r} using model {finetuned_model!r}"
    )
