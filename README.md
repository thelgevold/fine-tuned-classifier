# fine-tuned-classifier

This repo contains a self-contained workflow for fine-tuning a small classifier model with Unsloth, exporting merged weights, converting to GGUF, quantizing, and creating an Ollama model.

## Repo layout

- `fine-tuning/category_train.json`: static training dataset
- `fine-tuning/train_categories_unsloth.py`: QLoRA training script
- `fine-tuning/export_merged.py`: merges the LoRA adapter into the base model
- `fine-tuning/Dockerfile`: training image with Unsloth and llama.cpp
- `fine-tuning/docker-compose.yml`: local `unsloth` and `ollama` services
- `fine-tuning/run-finetune-pipeline.ps1`: end-to-end training wrapper

## Prerequisites

- Docker Desktop with NVIDIA GPU support enabled
- PowerShell

## Build the training image

From the repo root:

```powershell
docker compose -f fine-tuning/docker-compose.yml build
```

## Run the full pipeline

From the repo root:

```powershell
.\fine-tuning\run-finetune-pipeline.ps1 -ModelName our-house-qwen3-0.6b -BaseModel "Qwen/Qwen3-0.6B"
```

This wrapper will:

1. start the local `unsloth` and `ollama` containers
2. train the classifier
3. export merged weights
4. convert the merged model to `f16` GGUF
5. quantize the GGUF, default `Q4_K_M`
6. write `fine-tuning/outputs/<model-name>/Modelfile`
7. create the Ollama model

## Useful options

Run the held-out test set and skip the final full-dataset fit:

```powershell
.\fine-tuning\run-finetune-pipeline.ps1 -ModelName our-house-qwen3-0.6b -BaseModel "Qwen/Qwen3-0.6B" -RunFinalTest -FinalFitEpochs 0
```

Use a different quantization:

```powershell
.\fine-tuning\run-finetune-pipeline.ps1 -ModelName our-house-qwen3-0.6b -BaseModel "Qwen/Qwen3-0.6B" -Quantization "Q5_K_M"
```

## Outputs

The pipeline writes artifacts under:

- `fine-tuning/outputs/<model-name>/lora`
- `fine-tuning/outputs/<model-name>/merged`
- `fine-tuning/outputs/<model-name>/gguf`
- `fine-tuning/outputs/<model-name>/reports`

The final Ollama model is created with the same name you pass via `-ModelName`.
