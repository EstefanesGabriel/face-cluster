$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Criando ambiente virtual (.venv)..." -ForegroundColor Cyan
    python -m venv .venv
}

Write-Host "Verificando dependencias..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m pip install -q -r requirements.txt

Write-Host "Iniciando Face Cluster..." -ForegroundColor Green
.\.venv\Scripts\python.exe main.py
