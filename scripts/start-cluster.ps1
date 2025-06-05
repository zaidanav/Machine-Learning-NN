# start-cluster.ps1 - Windows PowerShell version

Write-Host "==========================================" -ForegroundColor Blue
Write-Host "  Starting Raft TypeScript Cluster" -ForegroundColor Blue  
Write-Host "==========================================" -ForegroundColor Blue

# Check if Docker is running
try {
    docker info | Out-Null
    Write-Host "✓ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "✗ Error: Docker is not running" -ForegroundColor Red
    Write-Host "Please start Docker Desktop first" -ForegroundColor Yellow
    exit 1
}

# Check if project files exist
if (!(Test-Path "package.json")) {
    Write-Host "✗ Error: package.json not found" -ForegroundColor Red
    Write-Host "Please run this from the project root directory" -ForegroundColor Yellow
    exit 1
}

# Install dependencies if needed
if (!(Test-Path "node_modules")) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    npm install
}

# Build TypeScript if needed  
if (!(Test-Path "dist")) {
    Write-Host "Building TypeScript..." -ForegroundColor Yellow
    npm run build
}

# Clean up existing containers
Write-Host "Cleaning up existing containers..." -ForegroundColor Yellow
docker-compose down --remove-orphans 2>$null

# Build and start cluster
Write-Host "Building and starting cluster..." -ForegroundColor Blue
docker-compose up --build -d

# Wait for containers
Write-Host "Waiting for containers to start..." -ForegroundColor Blue
Start-Sleep 10

# Check status
Write-Host "Container Status:" -ForegroundColor Blue
docker-compose ps

Write-Host "==========================================" -ForegroundColor Green
Write-Host "  Cluster Started Successfully!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Node Access:"
Write-Host "  Node 1 (Leader):  http://localhost:8001"
Write-Host "  Node 2:           http://localhost:8002"
Write-Host "  Node 3:           http://localhost:8003" 
Write-Host "  Node 4:           http://localhost:8004"
Write-Host ""
Write-Host "Client Access:"
Write-Host "  Interactive:      docker exec -it raft-client npm run client"
Write-Host "  Direct commands:  docker exec raft-client npm run client status"
Write-Host ""
Write-Host "Useful Commands:"
Write-Host "  docker-compose logs -f        # View all logs"
Write-Host "  docker logs raft-node-1       # View specific node"
Write-Host "  .\scripts\test-demo.ps1       # Run demo tests"
Write-Host "  docker-compose down           # Stop cluster"

---

# test-demo.ps1 - Windows PowerShell demo script

Write-Host "==========================================" -ForegroundColor Blue
Write-Host "  Raft TypeScript Demo Script" -ForegroundColor Blue
Write-Host "==========================================" -ForegroundColor Blue

function Pause {
    Write-Host "Press Enter to continue..." -ForegroundColor Yellow
    Read-Host
}

function Run-Client {
    param([string[]]$Arguments)
    docker exec raft-client npm run client @Arguments
}

Write-Host "This script will demonstrate the Raft implementation step by step."
Write-Host ""

# Check git status
Write-Host "=== Git Status ===" -ForegroundColor Blue
try {
    git status --short
} catch {
    Write-Host "Not in git repository" -ForegroundColor Yellow
}
Write-Host ""
Pause

# Demo Setup
Write-Host "=== Demo Setup ===" -ForegroundColor Blue
Write-Host "1. Checking cluster status..."
Run-Client "status"
Write-Host ""

Write-Host "2. Finding current leader..."
Run-Client "leader" 
Write-Host ""
Pause

# Heartbeat Demo
Write-Host "=== HEARTBEAT DEMO (10 points) ===" -ForegroundColor Blue
Write-Host "1. Testing heartbeat monitoring for 10 seconds..."
Run-Client "heartbeat", "10"
Write-Host ""

Write-Host "2. Testing ping connectivity..."
Run-Client "ping"
Write-Host ""
Pause

# Log Replication Demo
Write-Host "=== LOG REPLICATION DEMO (10 points) ===" -ForegroundColor Blue
Write-Host "1. Executing commands on leader..."

$commands = @(
    'set "1" "A"',
    'append "1" "BC"', 
    'set "2" "SI"',
    'append "2" "S"',
    'get "1"'
)

foreach ($cmd in $commands) {
    Write-Host "   Executing: $cmd"
    Run-Client "execute", $cmd
    Start-Sleep 1
}

Write-Host ""
Write-Host "2. Testing additional commands..."
Run-Client "execute", 'set "ruby-chan" "choco-minto"'
Run-Client "execute", 'set "ayumu-chan" "strawberry-flavor"'

Write-Host ""
Write-Host "3. Verifying results..."
Run-Client "execute", 'get "ruby-chan"'
Run-Client "execute", 'get "ayumu-chan"'
Write-Host ""
Pause

# Leader Election Demo
Write-Host "=== LEADER ELECTION DEMO (10 points) ===" -ForegroundColor Blue
Write-Host "1. Current cluster status:"
Run-Client "status"
Write-Host ""

Write-Host "2. To test leader election, stop the current leader container:"
Write-Host "   In another terminal, run: docker stop raft-node-1" -ForegroundColor Yellow
Write-Host ""

Write-Host "3. Monitoring election process..."
Run-Client "election"
Write-Host ""
Pause

# Demo Cleanup
Write-Host "=== Demo Completed! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Final cluster status:"
Run-Client "status"

Write-Host ""
Write-Host "Next steps for team integration:"
Write-Host "- Farhan: Implement log replication in RaftNode.execute()"
Write-Host "- Dhinto: Enhance membership change and RPC system"
Write-Host "- All: Integration testing and performance tuning"

---

# cleanup.ps1 - Windows cleanup script

Write-Host "Cleaning up Raft TypeScript cluster resources..." -ForegroundColor Blue

# Stop and remove containers
Write-Host "Stopping containers..."
docker-compose down --remove-orphans 2>$null

# Remove additional containers
Write-Host "Removing additional containers..."
docker rm -f raft-node-5 2>$null
docker rm -f raft-node-6 2>$null

# Clean up networks
Write-Host "Cleaning up networks..."
docker network rm raft-consensus_raft-network 2>$null

# Clean up volumes
Write-Host "Cleaning up volumes..."
docker volume rm raft-consensus_logs 2>$null

# Clean up build artifacts
$response = Read-Host "Remove build artifacts (dist/, node_modules/)? [y/N]"
if ($response -eq 'y' -or $response -eq 'Y') {
    Write-Host "Removing build artifacts..."
    if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
    if (Test-Path "node_modules") { Remove-Item -Recurse -Force "node_modules" }
}

# Clean up Docker images
$response = Read-Host "Remove Docker images? [y/N]"
if ($response -eq 'y' -or $response -eq 'Y') {
    Write-Host "Removing images..."
    docker rmi raft-consensus_raft-node-1 2>$null
    docker rmi raft-consensus_raft-node-2 2>$null  
    docker rmi raft-consensus_raft-node-3 2>$null
    docker rmi raft-consensus_raft-node-4 2>$null
    docker rmi raft-consensus_raft-client 2>$null
}

# System cleanup
Write-Host "Running Docker system cleanup..."
docker system prune -f

Write-Host "Cleanup completed!" -ForegroundColor Green