# fine-tuned-classifier

This repo contains a self-contained workflow for fine-tuning a small classifier model with Unsloth, exporting merged weights, converting to GGUF, quantizing, and creating an Ollama model.

## Repo layout

- `fine-tuning/data/category_train.json`: static training dataset
- `fine-tuning/train_categories.py`: QLoRA training script
- `fine-tuning/export_merged.py`: merges the LoRA adapter into the base model
- `fine-tuning/Dockerfile`: training image with Unsloth and llama.cpp
- `docker-compose.yml`: local `unsloth`, `ollama`, and API services
- `fine-tuning/run-finetune-pipeline.ps1`: end-to-end training wrapper
- `api/main.py`: FastAPI service for Ollama-backed categorization

## Prerequisites

- Docker Desktop with NVIDIA GPU support enabled
- PowerShell

## Build the training image

From the repo root:

```powershell
docker compose -f docker-compose.yml build unsloth
```

## Run the full pipeline

From the repo root:

```powershell
.\fine-tuning\run-finetune-pipeline.ps1 -ModelName our-house-qwen3-0.6b -BaseModel "Qwen/Qwen3-0.6B"
```

Train a parallel model that emits full category names instead of opaque codes:

```powershell
.\fine-tuning\run-finetune-pipeline-category-names.ps1 -ModelName our-house-qwen3-0.6b-category-names -BaseModel "Qwen/Qwen3-0.6B"
```

This wrapper will:

1. start the local `unsloth` and `ollama` containers
2. train the classifier
3. export merged weights
4. convert the merged model to `f16` GGUF
5. quantize the GGUF, default `Q4_K_M`
6. write `fine-tuning/outputs/<model-name>/Modelfile`
7. create the Ollama model

The default pipeline trains the model to emit opaque two-letter labels. The category-name pipeline keeps the same dataset and export flow, but trains the model to emit full category names such as `hvac` or `water heater`, and should use a distinct model name.

## Useful options

Use a different quantization:

```powershell
.\fine-tuning\run-finetune-pipeline.ps1 -ModelName our-house-qwen3-0.6b -BaseModel "Qwen/Qwen3-0.6B" -Quantization "Q5_K_M"
```

Use the main pipeline directly with explicit label mode selection:

```powershell
.\fine-tuning\run-finetune-pipeline.ps1 -ModelName our-house-qwen3-0.6b-category-names -BaseModel "Qwen/Qwen3-0.6B" -LabelMode category
```

## Outputs

The pipeline writes artifacts under:

- `fine-tuning/outputs/<model-name>/lora`
- `fine-tuning/outputs/<model-name>/merged`
- `fine-tuning/outputs/<model-name>/gguf`
- `fine-tuning/outputs/<model-name>/reports`

The final Ollama model is created with the same name you pass via `-ModelName`.

## API

Install the API dependencies:

```powershell
pip install -r api/requirements.txt
```

Run the API from the repo root:

```powershell
uvicorn api.main:app --reload
```

The API expects Ollama to be reachable at `http://localhost:11434` by default. Override that with `OLLAMA_BASE_URL` if needed.

Run the API in Docker with Ollama:

```powershell
docker compose -f docker-compose.yml up -d --build ollama api
```

The `unsloth` training container is under the `training` profile, so a plain `docker compose up` starts the API-facing services without trying to launch the GPU training container.

Example request:

```json
{
  "model_name": "our-house-qwen3-0.6b",
  "question": "When did we replace the downstairs AC unit?"
}
```

Example response:

```json
{
  "model_name": "our-house-qwen3-0.6b",
  "question": "When did we replace the downstairs AC unit?",
  "code": "hv",
  "category": "hvac"
}
```

## Evaluation

Install the API dependencies, which now include `pytest`:

```powershell
pip install -r api/requirements.txt
```

With `ollama` and `api` running locally, execute the comparison suite:

```powershell
python -m pytest eval --baseline-model qwen3:0.6b --finetuned-code-model our-house-qwen3-0.6b --finetuned-category-model our-house-qwen3-0.6b-category-names --num-predict 8
```

The suite will:

1. enumerate every question in `eval/categorization_test_cases.json`
2. run each case against all four scenarios:
   - fine-tuned with opaque short codes
   - fine-tuned with full category names
   - base model with full category names
   - base model with opaque short codes
3. assert the predicted category matches the expected category
4. generate comparison reports with accuracy and timing at:
   - `eval/reports/categorization_comparison_report.json`
   - `eval/reports/categorization_comparison_report.md`

Useful options:

```powershell
python -m pytest eval --api-base-url http://localhost:8090 --think
python -m pytest eval --baseline-model qwen3:0.6b --finetuned-code-model my-code-model --finetuned-category-model my-category-model
```
