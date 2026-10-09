@echo off
rem Starts the IPED embedding server using the virtualenv in this folder.
rem Extra arguments are passed through, e.g.: start_embedding_server.bat --port 8691 --dim 768
setlocal
set HERE=%~dp0
if not exist "%HERE%.venv\Scripts\python.exe" (
    echo Virtualenv not found. See %HERE%README.md to create it.
    exit /b 1
)
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
"%HERE%.venv\Scripts\python.exe" "%HERE%embedding_server.py" %*
