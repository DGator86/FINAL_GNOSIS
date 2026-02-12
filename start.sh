#!/bin/bash
# =============================================================================
# GNOSIS Trading System - Quick Start Script
# =============================================================================
#
# Usage:
#   ./start.sh              # Start with Docker Compose
#   ./start.sh local        # Start locally (no Docker)
#   ./start.sh train        # Train models first, then start
#   ./start.sh dry-run      # Start in dry-run mode (no orders)
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "============================================================"
echo "  GNOSIS TRADING SYSTEM"
echo "  Starting Up..."
echo "============================================================"
echo -e "${NC}"

# Check for .env file
if [ ! -f ".env" ]; then
    echo -e "${RED}ERROR: .env file not found!${NC}"
    echo "Please create a .env file with your API credentials."
    echo "See .env.example for reference."
    exit 1
fi

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Validate required credentials
if [ -z "$ALPACA_API_KEY" ] || [ "$ALPACA_API_KEY" = "your_key_here" ]; then
    echo -e "${RED}ERROR: ALPACA_API_KEY not configured in .env${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Configuration loaded${NC}"

# Parse command line arguments
MODE="${1:-docker}"

case "$MODE" in
    "docker")
        echo -e "${BLUE}Starting with Docker Compose...${NC}"
        
        # Check if Docker is running
        if ! docker info > /dev/null 2>&1; then
            echo -e "${RED}ERROR: Docker is not running!${NC}"
            exit 1
        fi
        
        # Build and start
        docker-compose up -d
        
        echo ""
        echo -e "${GREEN}✓ Services started!${NC}"
        echo ""
        echo "Services:"
        echo "  - Trading API:  http://localhost:8000"
        echo "  - API Docs:     http://localhost:8000/docs"
        echo "  - Prometheus:   http://localhost:9090"
        echo "  - Grafana:      http://localhost:3000 (admin/gnosis123)"
        echo ""
        echo "Commands:"
        echo "  docker-compose logs -f app    # View trading logs"
        echo "  docker-compose down           # Stop all services"
        ;;
        
    "local")
        echo -e "${BLUE}Starting locally...${NC}"
        
        # Check Python
        if ! command -v python3 &> /dev/null; then
            echo -e "${RED}ERROR: Python3 not found!${NC}"
            exit 1
        fi
        
        # Install dependencies if needed
        if [ ! -d "venv" ]; then
            echo "Creating virtual environment..."
            python3 -m venv venv
        fi
        
        source venv/bin/activate
        pip install -q -r requirements.txt
        
        # Start Redis if available
        if command -v redis-server &> /dev/null; then
            redis-server --daemonize yes 2>/dev/null || true
        fi
        
        # Start the application
        echo ""
        echo -e "${GREEN}Starting GNOSIS Trading System...${NC}"
        python3 scripts/launch.py
        ;;
        
    "train")
        echo -e "${BLUE}Training models before starting...${NC}"
        
        # Check Python
        if ! command -v python3 &> /dev/null; then
            echo -e "${RED}ERROR: Python3 not found!${NC}"
            exit 1
        fi
        
        # Train models
        echo "Training ML models..."
        python3 scripts/quick_train.py
        
        # Then start
        echo ""
        echo -e "${GREEN}Starting GNOSIS Trading System...${NC}"
        python3 scripts/launch.py
        ;;
        
    "dry-run")
        echo -e "${YELLOW}Starting in DRY-RUN mode (no actual orders)...${NC}"
        
        python3 scripts/launch.py --dry-run
        ;;
        
    *)
        echo "Usage: $0 [docker|local|train|dry-run]"
        echo ""
        echo "Modes:"
        echo "  docker   - Start with Docker Compose (default)"
        echo "  local    - Start locally without Docker"
        echo "  train    - Train ML models first, then start"
        echo "  dry-run  - Run without executing actual orders"
        exit 1
        ;;
esac
