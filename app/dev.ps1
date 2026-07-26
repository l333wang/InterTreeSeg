# Launch backend (FastAPI on :8000) and frontend (Vite on :5173) for development.
# Usage:  cd <repo>/app ;  ./dev.ps1
# Optional env vars: ANNO_BACKEND_ENV (conda env for backend, default "intertreeseg"),
#   ANNO_INFERENCE_BACKEND ("ptv3" | "mock"), NPM_CMD (path to npm), ANNO_PTV3_CKPT.
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot

# Backend conda env (needs torch + PTv3 deps + web stack) running the real model.
# Set ANNO_INFERENCE_BACKEND=mock to run without the model / GPU.
$backendEnv = $env:ANNO_BACKEND_ENV; if (-not $backendEnv) { $backendEnv = "intertreeseg" }
$backend = $env:ANNO_INFERENCE_BACKEND; if (-not $backend) { $backend = "ptv3" }
Write-Host "Starting backend ($backendEnv env, uvicorn :8000, inference=$backend)..." -ForegroundColor Cyan
$env:ANNO_INFERENCE_BACKEND = $backend
Start-Process -FilePath "conda" `
  -ArgumentList "run", "-n", $backendEnv, "--no-capture-output", "python", "-m", "uvicorn", "backend.main:app", "--port", "8000" `
  -WorkingDirectory $here

Start-Sleep -Seconds 2

Write-Host "Starting frontend (vite :5173)..." -ForegroundColor Cyan
$npm = $env:NPM_CMD; if (-not $npm) { $npm = "npm" }
Start-Process -FilePath $npm -ArgumentList "run", "dev" -WorkingDirectory (Join-Path $here "frontend")

Start-Sleep -Seconds 3
Write-Host "`nOpen http://127.0.0.1:5173" -ForegroundColor Green
