param(
    [Parameter(Mandatory = $true)]
    [string]$ModelName,

    [string]$BaseModel = "Qwen/Qwen3-0.6B",

    [string]$Quantization = "Q4_K_M",

    [ValidateSet("code", "category")]
    [string]$LabelMode = "code"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$localOutputDir = Join-Path $scriptDir "outputs\$ModelName"
$localLoraDir = Join-Path $localOutputDir "lora"
$localMergedDir = Join-Path $localOutputDir "merged"
$localGgufDir = Join-Path $localOutputDir "gguf"
$localReportsDir = Join-Path $localOutputDir "reports"
$outputDir = "/workspace/fine-tuning/outputs/$ModelName"
$loraDir = "$outputDir/lora"
$mergedDir = "$outputDir/merged"
$ggufDir = "$outputDir/gguf"
$reportsDir = "$outputDir/reports"
$f16GgufFile = "$ggufDir/$ModelName-f16.gguf"
$quantSuffix = $Quantization.ToLower()
$quantizedGgufFile = "$ggufDir/$ModelName-$quantSuffix.gguf"
$modelfilePath = Join-Path $localOutputDir "Modelfile"
$modelfileContainerPath = "/workspace/fine-tuning/outputs/$ModelName/Modelfile"

function Invoke-DockerComposeOrThrow {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [Parameter(Mandatory = $true)]
        [string]$FailureMessage
    )

    & docker compose @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw $FailureMessage
    }
}

Write-Host "Running fine-tuning pipeline for $ModelName"
Write-Host "Label mode: $LabelMode"

Push-Location $repoRoot
try {
    Write-Host "Preparing output folders"
    @(
        $localOutputDir,
        $localLoraDir,
        $localMergedDir,
        $localGgufDir,
        $localReportsDir
    ) | ForEach-Object {
        New-Item -ItemType Directory -Force -Path $_ | Out-Null
    }

    Write-Host "Step 1"
    Invoke-DockerComposeOrThrow -Arguments @("-f", "docker-compose.yml", "--profile", "training", "up", "-d", "unsloth", "ollama") -FailureMessage "Unable to start the training containers. Check Docker and GPU runtime health."
    Invoke-DockerComposeOrThrow -Arguments @("-f", "docker-compose.yml", "exec", "unsloth", "mkdir", "-p", $outputDir, $loraDir, $mergedDir, $ggufDir, $reportsDir) -FailureMessage "Unable to prepare output folders inside the unsloth container."
    Write-Host "Step 2"
    $trainCommand = @(
        "python",
        "/workspace/fine-tuning/train_categories.py",
        "--base-model", $BaseModel,
        "--data-path", "/workspace/fine-tuning/data/category_train.json",
        "--output-dir", $outputDir,
        "--label-mode", $LabelMode
    )
    Invoke-DockerComposeOrThrow -Arguments (@("-f", "docker-compose.yml", "exec", "unsloth") + $trainCommand) -FailureMessage "Training failed inside the unsloth container."
    Write-Host "Step 3"
    Invoke-DockerComposeOrThrow -Arguments @("-f", "docker-compose.yml", "exec", "unsloth", "python", "/workspace/fine-tuning/export_merged.py", "--base-model", $BaseModel, "--output-dir", $outputDir) -FailureMessage "Merged model export failed."
    Write-Host "Step 4"
    Invoke-DockerComposeOrThrow -Arguments @("-f", "docker-compose.yml", "exec", "unsloth", "python", "/opt/llama.cpp/convert_hf_to_gguf.py", $mergedDir, "--outfile", $f16GgufFile, "--outtype", "f16") -FailureMessage "GGUF conversion failed."
    Write-Host "Step 5"
    Invoke-DockerComposeOrThrow -Arguments @("-f", "docker-compose.yml", "exec", "unsloth", "/opt/llama.cpp/build/bin/llama-quantize", $f16GgufFile, $quantizedGgufFile, $Quantization) -FailureMessage "GGUF quantization failed."

    Write-Host "Step 6"
    $modelfileDir = Split-Path -Parent $modelfilePath
    New-Item -ItemType Directory -Force -Path $modelfileDir | Out-Null
    Set-Content -Path $modelfilePath -Value "FROM ./gguf/$ModelName-$quantSuffix.gguf`n`nPARAMETER temperature 0"

    Write-Host "Step 7"
    Invoke-DockerComposeOrThrow -Arguments @("-f", "docker-compose.yml", "exec", "ollama", "ollama", "create", $ModelName, "-f", $modelfileContainerPath) -FailureMessage "Ollama model creation failed."
}
finally {
    Pop-Location
}
