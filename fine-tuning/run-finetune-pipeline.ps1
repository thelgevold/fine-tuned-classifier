param(
    [Parameter(Mandatory = $true)]
    [string]$ModelName,

    [string]$BaseModel = "Qwen/Qwen3-0.6B",

    [string]$Quantization = "Q4_K_M"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$outputDir = "/workspace/fine-tuning/outputs/$ModelName"
$ggufDir = "$outputDir/gguf"
$f16GgufFile = "$ggufDir/$ModelName-f16.gguf"
$quantSuffix = $Quantization.ToLower()
$quantizedGgufFile = "$ggufDir/$ModelName-$quantSuffix.gguf"
$modelfilePath = Join-Path $scriptDir "outputs\$ModelName\Modelfile"
$modelfileContainerPath = "/workspace/fine-tuning/outputs/$ModelName/Modelfile"

Write-Host "Running fine-tuning pipeline for $ModelName"

Push-Location $repoRoot
try {
    Write-Host "Step 1"
    docker compose -f fine-tuning/docker-compose.yml up -d unsloth ollama
    Write-Host "Step 2"
    $trainCommand = @(
        "python",
        "/workspace/fine-tuning/train_categories.py",
        "--base-model", $BaseModel,
        "--data-path", "/workspace/fine-tuning/data/category_train.json",
        "--output-dir", $outputDir
    )
    docker compose -f fine-tuning/docker-compose.yml exec unsloth @trainCommand
    Write-Host "Step 3"
    docker compose -f fine-tuning/docker-compose.yml exec unsloth python /workspace/fine-tuning/export_merged.py --base-model $BaseModel --output-dir $outputDir
    Write-Host "Step 4"
    docker compose -f fine-tuning/docker-compose.yml exec unsloth mkdir -p $ggufDir
    Write-Host "Step 5"
    docker compose -f fine-tuning/docker-compose.yml exec unsloth python /opt/llama.cpp/convert_hf_to_gguf.py "$outputDir/merged" --outfile $f16GgufFile --outtype f16
    Write-Host "Step 6"
    docker compose -f fine-tuning/docker-compose.yml exec unsloth /opt/llama.cpp/build/bin/llama-quantize $f16GgufFile $quantizedGgufFile $Quantization

    $modelfileDir = Split-Path -Parent $modelfilePath
    New-Item -ItemType Directory -Force -Path $modelfileDir | Out-Null
    Set-Content -Path $modelfilePath -Value "FROM ./gguf/$ModelName-$quantSuffix.gguf`n`nPARAMETER temperature 0"

    Write-Host "Step 7"
    docker compose -f fine-tuning/docker-compose.yml exec ollama ollama create $ModelName -f $modelfileContainerPath
}
finally {
    Pop-Location
}
