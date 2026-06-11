param(
    [Parameter(Mandatory = $true)]
    [string]$ModelName,

    [string]$BaseModel = "Qwen/Qwen3-0.6B",

    [string]$Quantization = "Q4_K_M"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

& (Join-Path $scriptDir "run-finetune-pipeline.ps1") `
    -ModelName $ModelName `
    -BaseModel $BaseModel `
    -Quantization $Quantization `
    -LabelMode "category"
