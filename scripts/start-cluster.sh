#!/bin/bash
# start-cluster.sh - Start Raft cluster

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=========================================="
echo "  Starting Raft TypeScript Cluster"
echo "==========================================${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    exit 1
fi

# Check if Node.js project is built
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    npm install
fi

if [ ! -d "dist" ]; then
    echo -e "${YELLOW}Building TypeScript...${NC}"
    npm run build
fi

# Clean up any existing containers
echo -e "${YELLOW}Cleaning up existing containers...${NC}"
docker-compose down --remove-orphans 2>/dev/null || true

# Build and start the cluster
echo -e "${BLUE}Building and starting cluster...${NC}"
docker-compose up --build -d

# Wait for containers to be ready
echo -e "${BLUE}Waiting for containers to start...${NC}"
sleep 10

# Check container status
echo -e "${BLUE}Container Status:${NC}"
docker-compose ps

# Setup network conditions
echo -e "${BLUE}Setting up network simulation...${NC}"
chmod +x ./scripts/setup-network.sh
./scripts/setup-network.sh setup

# Show cluster information
echo -e "${GREEN}=========================================="
echo "  Cluster Started Successfully!"
echo "==========================================${NC}"
echo ""
echo "Node Access:"
echo "  Node 1 (Leader):  http://localhost:8001"
echo "  Node 2:           http://localhost:8002" 
echo "  Node 3:           http://localhost:8003"
echo "  Node 4:           http://localhost:8004"
echo ""
echo "Client Access:"
echo "  Interactive:      docker exec -it raft-client npm run client"
echo "  Direct commands:  docker exec raft-client npm run client status"
echo ""
echo "Useful Commands:"
echo "  docker-compose logs -f                    # View all logs"
echo "  docker logs raft-node-1                  # View specific node logs"
echo "  docker exec -it raft-client bash         # Access client container"
echo "  ./scripts/test-demo.sh                   # Run demo tests"
echo "  docker-compose down                      # Stop cluster"

---

#!/bin/bash
# setup-network.sh - Network simulation for TypeScript Raft

set -e

echo "=========================================="
echo "  Raft Network Simulation Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to check if container is running
check_container() {
    local container_name=$1
    if docker ps | grep -q "$container_name"; then
        echo -e "${GREEN}✓${NC} $container_name is running"
        return 0
    else
        echo -e "${RED}✗${NC} $container_name is not running"
        return 1
    fi
}

# Function to add network conditions to container
add_network_conditions() {
    local container_name=$1
    local delay=${2:-"100ms"}
    local delay_variation=${3:-"50ms"}
    local loss=${4:-"2%"}
    local corruption=${5:-"1%"}
    local reorder=${6:-"5%"}
    local duplicate=${7:-"1%"}
    
    echo -e "${BLUE}Setting up network conditions for $container_name:${NC}"
    echo "  - Delay: $delay ± $delay_variation"
    echo "  - Packet Loss: $loss"
    echo "  - Corruption: $corruption"
    echo "  - Reorder: $reorder"
    echo "  - Duplicate: $duplicate"
    
    # Check if container is running
    if ! check_container "$container_name"; then
        echo -e "${YELLOW}Warning: Skipping $container_name (not running)${NC}"
        return 1
    fi
    
    # Apply network conditions using tc (traffic control)
    docker exec "$container_name" bash -c "
        # Remove any existing qdisc
        tc qdisc del dev eth0 root 2>/dev/null || true
        
        # Add network emulation
        tc qdisc add dev eth0 root netem \
            delay $delay $delay_variation \
            loss $loss \
            corrupt $corruption \
            reorder $reorder 25% \
            duplicate $duplicate
            
        echo 'Network conditions applied successfully'
    " 2>/dev/null || {
        echo -e "${RED}Failed to apply network conditions to $container_name${NC}"
        return 1
    }
    
    echo -e "${GREEN}✓ Network conditions applied to $container_name${NC}"
    echo ""
}

# Function to remove network conditions
remove_network_conditions() {
    local container_name=$1
    
    echo -e "${BLUE}Removing network conditions from $container_name${NC}"
    
    docker exec "$container_name" bash -c "
        tc qdisc del dev eth0 root 2>/dev/null || true
        echo 'Network conditions removed'
    " 2>/dev/null || {
        echo -e "${YELLOW}Warning: Could not remove conditions from $container_name${NC}"
    }
}

# Main script logic
case "${1:-setup}" in
    "setup")
        echo "Setting up realistic network conditions for testing..."
        echo ""
        
        # Apply different network conditions to each node
        add_network_conditions "raft-node-1" "50ms" "25ms" "1%" "0.5%" "3%" "0.5%"
        add_network_conditions "raft-node-2" "75ms" "30ms" "2%" "1%" "4%" "1%"
        add_network_conditions "raft-node-3" "60ms" "20ms" "1.5%" "0.5%" "3%" "0.5%"
        add_network_conditions "raft-node-4" "80ms" "40ms" "2.5%" "1%" "5%" "1%"
        
        echo -e "${GREEN}Network simulation setup complete!${NC}"
        ;;
        
    "demo")
        echo "Setting up DEMO network conditions (harsh):"
        echo ""
        
        # Apply harsh conditions for demo
        add_network_conditions "raft-node-1" "1000ms" "50ms" "5%" "5%" "8%" "2%"
        add_network_conditions "raft-node-2" "1000ms" "50ms" "5%" "5%" "8%" "2%"
        add_network_conditions "raft-node-3" "1000ms" "50ms" "5%" "5%" "8%" "2%"
        add_network_conditions "raft-node-4" "1000ms" "50ms" "5%" "5%" "8%" "2%"
        
        echo -e "${GREEN}Demo network conditions applied!${NC}"
        echo -e "${YELLOW}Warning: These conditions are harsh and may cause timeouts.${NC}"
        ;;
        
    "cleanup")
        echo "Removing all network conditions..."
        echo ""
        
        remove_network_conditions "raft-node-1"
        remove_network_conditions "raft-node-2" 
        remove_network_conditions "raft-node-3"
        remove_network_conditions "raft-node-4"
        
        echo -e "${GREEN}Network conditions cleaned up!${NC}"
        ;;
        
    "status")
        echo "Current network conditions:"
        echo ""
        
        for i in {1..4}; do
            echo -e "${BLUE}raft-node-$i:${NC}"
            docker exec "raft-node-$i" tc qdisc show dev eth0 2>/dev/null || echo "No conditions"
            echo ""
        done
        ;;
        
    *)
        echo "Usage: $0 [setup|demo|cleanup|status]"
        echo ""
        echo "Commands:"
        echo "  setup   - Apply realistic network conditions for testing"
        echo "  demo    - Apply harsh conditions matching demo specification"
        echo "  cleanup - Remove all network conditions"
        echo "  status  - Show current network conditions"
        ;;
esac

---

#!/bin/bash
# test-demo.sh - Complete demo script for TypeScript Raft

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=========================================="
echo "  Raft TypeScript Demo Script"
echo "==========================================${NC}"

# Function to run client command
run_client() {
    docker exec raft-client npm run client "$@"
}

# Function to pause for user input
pause() {
    echo -e "${YELLOW}Press Enter to continue...${NC}"
    read
}

echo "This script will demonstrate the Raft implementation step by step."
echo ""

# Check git status
echo -e "${BLUE}=== Git Status ===${NC}"
git status --short || echo "Not in git repository"
echo ""
pause

# Demo Setup
echo -e "${BLUE}=== Demo Setup ===${NC}"
echo "1. Checking cluster status..."
run_client status
echo ""

echo "2. Finding current leader..."
run_client leader
echo ""

echo "3. Applying demo network conditions..."
./scripts/setup-network.sh demo
echo ""
pause

# Heartbeat Demo
echo -e "${BLUE}=== HEARTBEAT DEMO (10 points) ===${NC}"
echo "1. Testing heartbeat monitoring for 10 seconds..."
run_client heartbeat 10
echo ""

echo "2. Testing ping connectivity..."
run_client ping
echo ""
pause

# Log Replication Demo
echo -e "${BLUE}=== LOG REPLICATION DEMO (10 points) ===${NC}"
echo "1. Executing commands on leader..."

commands=(
    'set "1" "A"'
    'append "1" "BC"'
    'set "2" "SI"'
    'append "2" "S"'
    'get "1"'
)

for cmd in "${commands[@]}"; do
    echo "   Executing: $cmd"
    run_client execute $cmd
    sleep 1
done

echo ""
echo "2. Testing concurrent commands..."
echo "   (This will be enhanced when Farhan implements log replication)"

# Simulate concurrent commands
run_client execute 'set "ruby-chan" "choco-minto"' &
sleep 0.5
run_client execute 'set "ayumu-chan" "strawberry-flavor"' &
wait

echo ""
echo "3. Verifying results..."
run_client execute 'get "ruby-chan"'
run_client execute 'get "ayumu-chan"'
echo ""
pause

# Leader Election Demo
echo -e "${BLUE}=== LEADER ELECTION DEMO (10 points) ===${NC}"
echo "1. Current cluster status:"
run_client status
echo ""

echo "2. To test leader election, stop the current leader:"
leader_container=$(docker exec raft-client npm run client leader 2>/dev/null | grep "Leader:" | cut -d' ' -f2 | cut -d':' -f1)
if [[ $leader_container == "172.20.0.10" ]]; then
    echo "   docker stop raft-node-1"
elif [[ $leader_container == "172.20.0.11" ]]; then
    echo "   docker stop raft-node-2"
elif [[ $leader_container == "172.20.0.12" ]]; then
    echo "   docker stop raft-node-3"
elif [[ $leader_container == "172.20.0.13" ]]; then
    echo "   docker stop raft-node-4"
fi

echo ""
echo "3. Monitoring election process..."
run_client election
echo ""
pause

# Membership Change Demo (placeholder)
echo -e "${BLUE}=== MEMBERSHIP CHANGE DEMO (10 points) ===${NC}"
echo "Note: This will be fully implemented by Dhinto"
echo ""

echo "1. Current cluster size:"
run_client status
echo ""

echo "2. Adding new node (placeholder)..."
echo "   docker run --name raft-node-5 --network raft-consensus_raft-network ..."
echo "   (Implementation pending)"
echo ""

echo "3. Verifying membership change..."
echo "   (Will show log replication to new node when implemented)"
echo ""
pause

# Demo Cleanup
echo -e "${BLUE}=== Demo Cleanup ===${NC}"
echo "Removing network simulation conditions..."
./scripts/setup-network.sh cleanup

echo ""
echo -e "${GREEN}=========================================="
echo "  Demo Completed!"
echo "==========================================${NC}"
echo ""
echo "Final cluster status:"
run_client status

echo ""
echo "Next steps for team integration:"
echo "- Farhan: Implement log replication in RaftNode.execute()"
echo "- Dhinto: Enhance membership change and RPC system"
echo "- All: Integration testing and performance tuning"

---

#!/bin/bash
# cleanup.sh - Cleanup all resources

set -e

echo "Cleaning up Raft TypeScript cluster resources..."

# Stop and remove all containers
echo "Stopping containers..."
docker-compose down --remove-orphans 2>/dev/null || true

# Remove any additional containers
echo "Removing additional containers..."
docker rm -f raft-node-5 2>/dev/null || true
docker rm -f raft-node-6 2>/dev/null || true

# Clean up networks
echo "Cleaning up networks..."
docker network rm raft-consensus_raft-network 2>/dev/null || true

# Clean up volumes (optional)
echo "Cleaning up volumes..."
docker volume rm raft-consensus_logs 2>/dev/null || true

# Clean up build artifacts
read -p "Remove build artifacts (dist/, node_modules/)? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Removing build artifacts..."
    rm -rf dist/
    rm -rf node_modules/
fi

# Clean up Docker images (optional)
read -p "Remove Docker images? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Removing images..."
    docker rmi raft-consensus_raft-node-1 2>/dev/null || true
    docker rmi raft-consensus_raft-node-2 2>/dev/null || true
    docker rmi raft-consensus_raft-node-3 2>/dev/null || true
    docker rmi raft-consensus_raft-node-4 2>/dev/null || true
    docker rmi raft-consensus_raft-client 2>/dev/null || true
fi

# System cleanup
echo "Running Docker system cleanup..."
docker system prune -f

echo "Cleanup completed!"