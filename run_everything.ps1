# AgenticSpeedRAG Startup Script

Write-Host "🚀 Starting AgenticSpeedRAG Stack..." -ForegroundColor Green

# 1. Fix Docker Path (if needed)
$env:Path = "C:\Program Files\Docker\Docker\resources\bin;" + $env:Path

# 2. Start Docker Services
Write-Host "docker-compose up -d..."
& docker-compose up -d

# 3. Initialize Database (Run Pipeline)
Write-Host "Initializing Database Schema..."
$env:PYTHONPATH = "."
python -m backend.ingestion.pipeline

# 4. Start API Server
Write-Host "Starting API Endpoint..."
python -m backend.app.main
