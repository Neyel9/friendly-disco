# Friendly Disco Deployment Script
param(
    [string]$Mode = "production",
    [string]$Port = "8888",
    [string]$Tag = "latest"
)

Write-Host "🚀 Deploying Friendly Disco Sentiment Analysis App..." -ForegroundColor Green

# Stop and remove existing container
Write-Host "🛑 Stopping existing containers..." -ForegroundColor Yellow
docker stop sentiment-app-debug 2>$null
docker rm sentiment-app-debug 2>$null

# Build new image
Write-Host "🔨 Building Docker image..." -ForegroundColor Blue
docker build -t friendly_disco:$Tag .

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker build failed!" -ForegroundColor Red
    exit 1
}

# Deploy based on mode
if ($Mode -eq "test") {
    Write-Host "🧪 Deploying in TEST mode..." -ForegroundColor Cyan
    docker run -d `
        --name sentiment-app-debug `
        --restart unless-stopped `
        -e TESTING=true `
        -p ${Port}:5000 `
        friendly_disco:$Tag
} else {
    Write-Host "🚀 Deploying in PRODUCTION mode..." -ForegroundColor Green
    docker run -d `
        --name sentiment-app-debug `
        --restart unless-stopped `
        --env-file docker.env `
        -p ${Port}:5000 `
        --memory=1g `
        --cpus=1 `
        friendly_disco:$Tag
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Deployment successful!" -ForegroundColor Green
    Write-Host "🌐 App is running at: http://localhost:$Port" -ForegroundColor Cyan
    Write-Host "🔍 Health check: http://localhost:$Port/health" -ForegroundColor Cyan
    Write-Host "📊 Metrics: http://localhost:$Port/metrics" -ForegroundColor Cyan
    
    # Show container status
    Write-Host "`n📋 Container Status:" -ForegroundColor Yellow
    docker ps --filter name=sentiment-app-debug
} else {
    Write-Host "❌ Deployment failed!" -ForegroundColor Red
    exit 1
}
