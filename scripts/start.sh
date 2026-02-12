#!/bin/bash
# =============================================================================
# GNOSIS Trading System - Quick Start Script
# =============================================================================
#
# Usage:
#   ./scripts/start.sh              # Start with Docker
#   ./scripts/start.sh --local      # Start locally (no Docker)
#   ./scripts/start.sh --train      # Train models first, then start
#   ./scripts/start.sh --dry-run    # Start in dry-run mode (no actual trades)
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║   ██████╗ ███╗   ██╗ ██████╗ ███████╗██╗███████╗             ║"
echo "║  ██╔════╝ ████╗  ██║██╔═══██╗██╔════╝██║██╔════╝             ║"
echo "║  ██║  ███╗██╔██╗ ██║██║   ██║███████╗██║███████╗             ║"
echo "║  ██║   ██║██║╚██╗██║██║   ██║╚════██║██║╚════██║             ║"
echo "║  ╚██████╔╝██║ ╚████║╚██████╔╝███████║██║███████║             ║"
echo "║   ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ ╚══════╝╚═╝╚══════╝             ║"
echo "║                                                               ║"
echo "║           Super Gnosis Elite Trading System                   ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Parse arguments
USE_DOCKER=true
TRAIN_FIRST=false
DRY_RUN=false
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            USE_DOCKER=false
            shift
            ;;
        --train)
            TRAIN_FIRST=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            EXTRA_ARGS="$EXTRA_ARGS --dry-run"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Check for .env file
if [ ! -f ".env" ]; then
    echo -e "${RED}Error: .env file not found!${NC}"
    echo "Please create a .env file with your API credentials."
    echo "You can copy from .env.example:"
    echo "  cp .env.example .env"
    exit 1
fi

# Load environment
export $(grep -v '^#' .env | xargs)

echo -e "${GREEN}✓ Configuration loaded${NC}"
echo ""

# Validate critical environment variables
if [ -z "$ALPACA_API_KEY" ] || [ -z "$ALPACA_SECRET_KEY" ]; then
    echo -e "${RED}Error: Alpaca API credentials not configured!${NC}"
    echo "Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env"
    exit 1
fi

echo -e "${GREEN}✓ Alpaca credentials found${NC}"

if [ "$USE_DOCKER" = true ]; then
    # ==========================================================================
    # Docker Mode
    # ==========================================================================
    echo -e "\n${BLUE}Starting with Docker...${NC}\n"
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}Error: Docker is not running!${NC}"
        echo "Please start Docker and try again."
        exit 1
    fi
    
    # Train models first if requested
    if [ "$TRAIN_FIRST" = true ]; then
        echo -e "${YELLOW}Training ML models first...${NC}"
        docker-compose run --rm app python scripts/quick_train.py
    fi
    
    # Start all services
    echo -e "${GREEN}Starting all services...${NC}"
    docker-compose up -d
    
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}GNOSIS Trading System is now running!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Services:"
    echo "  📊 Trading API:   http://localhost:8000"
    echo "  📊 API Docs:      http://localhost:8000/docs"
    echo "  📈 Grafana:       http://localhost:3000 (admin/gnosis123)"
    echo "  📉 Prometheus:    http://localhost:9090"
    echo ""
    echo "Commands:"
    echo "  View logs:        docker-compose logs -f app"
    echo "  Stop system:      docker-compose down"
    echo "  Restart:          docker-compose restart app"
    echo ""
    
else
    # ==========================================================================
    # Local Mode (no Docker)
    # ==========================================================================
    echo -e "\n${BLUE}Starting locally...${NC}\n"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: Python 3 not found!${NC}"
        exit 1
    fi
    
    # Create virtual environment if not exists
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    echo "Installing dependencies..."
    pip install -r requirements.txt -q
    
    # Create necessary directories
    mkdir -p logs data/training models/trained checkpoints
    
    # Train models first if requested
    if [ "$TRAIN_FIRST" = true ]; then
        echo -e "${YELLOW}Training ML models...${NC}"
        python scripts/quick_train.py
    fi
    
    # Start the trading system
    echo -e "\n${GREEN}Starting GNOSIS Trading System...${NC}\n"
    
    if [ "$DRY_RUN" = true ]; then
        python scripts/launch.py --dry-run
    else
        python scripts/launch.py
    fi
fi
