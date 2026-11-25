@echo off
setlocal
rem Executa o exercicio do KNN e mostra o resultado.

rem Garante que o script rode a partir da pasta do repositorio
cd /d "%~dp0"

rem Acentos corretos no console do Windows
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

rem Procura o Python: launcher "py", depois "python", depois "python3"
set "PYTHON="
where py >nul 2>&1 && set "PYTHON=py -3"
if not defined PYTHON (
    where python >nul 2>&1 && set "PYTHON=python"
)
if not defined PYTHON (
    where python3 >nul 2>&1 && set "PYTHON=python3"
)

if not defined PYTHON (
    echo [ERRO] Python nao encontrado.
    echo Instale o Python 3.9 ou superior em https://www.python.org/downloads/
    echo e marque a opcao "Add Python to PATH" durante a instalacao.
    echo.
    pause
    exit /b 1
)

echo Usando: %PYTHON%
echo.

%PYTHON% exercicio.py
set "CODIGO=%ERRORLEVEL%"

echo.
if not "%CODIGO%"=="0" (
    echo [ERRO] A execucao terminou com codigo %CODIGO%.
)
pause
exit /b %CODIGO%
