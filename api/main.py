import os
import sys
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException


REPO_ROOT = Path(__file__).resolve().parent.parent
FINE_TUNING_ROOT = REPO_ROOT / "fine-tuning"
if str(FINE_TUNING_ROOT) not in sys.path:
    sys.path.insert(0, str(FINE_TUNING_ROOT))

from domain.categories import CATEGORY_OUTPUT_CODES
from handlers.prompt_handler import PromptHandler
from api.models import CategorizeQuestionRequest, CategorizeQuestionResponse


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11499")
OUTPUT_CODE_TO_CATEGORY = {
    code: category for category, code in CATEGORY_OUTPUT_CODES.items()
}

app = FastAPI(title="Question Categorization API", version="1.0.0")


def normalize_prediction(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    first_token = cleaned.split()[0].rstrip(":").strip().lower()
    if first_token in OUTPUT_CODE_TO_CATEGORY:
        return first_token
    return cleaned.lower()


@app.post("/categorize", response_model=CategorizeQuestionResponse)
async def categorize_question(
    request: CategorizeQuestionRequest,
) -> CategorizeQuestionResponse:
    prompt = PromptHandler.create_categorize_query_prompt(
        request.question,
        CATEGORY_OUTPUT_CODES,
    )

    payload = {
        "model": request.model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 3,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        if exc.response is not None:
            try:
                error_message = exc.response.json().get("error", detail)
            except ValueError:
                error_message = detail

            if exc.response.status_code == 404 and "not found" in error_message.lower():
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Ollama model '{request.model_name}' was not found in the configured "
                        f"Ollama instance at {OLLAMA_BASE_URL}."
                    ),
                ) from exc

        raise HTTPException(status_code=502, detail=f"Ollama request failed: {detail}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Unable to reach Ollama: {exc}") from exc

    raw_response = response.json().get("response", "")
    code = normalize_prediction(raw_response)
    category = OUTPUT_CODE_TO_CATEGORY.get(code)
    if category is None:
        raise HTTPException(
            status_code=422,
            detail=f"Ollama returned an unknown category code: {raw_response!r}",
        )

    return CategorizeQuestionResponse(
        model_name=request.model_name,
        question=request.question,
        code=code,
        category=category,
    )
