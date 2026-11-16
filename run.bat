@echo off
setlocal

if not exist ".venv\Scripts\python.exe" (
    echo Criando ambiente virtual ^(.venv^)...
    python -m venv .venv
)

echo Verificando dependencias...
.\.venv\Scripts\python.exe -m pip install -q -r requirements.txt

echo Iniciando Face Cluster...
.\.venv\Scripts\python.exe main.py
